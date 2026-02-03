"""
음성 분석 API 라우터 (Voice Analysis Router)

보이스피싱 탐지를 위한 음성 분석 엔드포인트를 제공합니다.

시스템 아키텍처:
┌─────────────────────────────────────────────────────────────────┐
│ [클라이언트] POST /analyze (오디오 파일)                         │
│       ↓                                                          │
│ [FastAPI Router - voice_analysis.py]                            │
│       ↓                                                          │
│ ┌─────────────── 백그라운드 파이프라인 ──────────────┐          │
│ │ 1. 오디오 청크 분리 (5초 단위)                      │          │
│ │ 2. STT 전사 (faster-whisper)                        │          │
│ │ 3. 텍스트 누적                                      │          │
│ │ 4. 30초마다 LLM 분석 트리거                         │          │
│ │ 5. 최종 결과 저장                                   │          │
│ └────────────────────────────────────────────────────┘          │
│       ↓                                                          │
│ [클라이언트] GET /status/{run_id} (폴링)                        │
└─────────────────────────────────────────────────────────────────┘

엔드포인트:
- POST /analyze: 오디오 파일 분석 시작 (비동기)
- GET /status/{run_id}: 분석 상태 조회 (long-polling 지원)
- POST /predict_text: 텍스트 직접 분석 (STT 건너뛰기)

의존성:
- STT: src.services.stt_service.FasterWhisperEngine
- LLM: scripts/finetuning/inference.py (동적 임포트)

작성자: Voice Phishing Protector Team
최종 수정: 2026-02-01
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.config import settings
from src.core.exceptions import (
    AudioLoadError,
    VoicePhishingProtectorError,
)
from src.models.schemas import AnalysisStatus, AnalyzeResponse, PredictTextResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice Analysis"])

# ============================================================================
# 상태 저장소 (In-Memory State Store)
# ============================================================================
# 프로덕션 환경에서는 Redis/Memcached로 교체 권장
# 이유: 서버 재시작 시 상태 유실, 멀티 프로세스 환경에서 공유 불가
# Trade-off: 간단한 구현 vs 영속성/확장성
# ============================================================================

RUNS: Dict[str, Dict[str, Any]] = {}

# inference 모듈 캐시 (지연 로드)
_local_models = None

# STT 엔진 캐시 (지연 로드)
_stt_engine = None


# ============================================================================
# 모듈 지연 로딩 (Lazy Loading)
# ============================================================================


def _get_inference_module():
    """
    scripts/finetuning/inference.py 모듈을 지연 로드합니다.
    
    지연 로딩 이유:
    - 앱 시작 시간 단축 (모델 로드 약 3-5초 절약)
    - 메모리 절약 (사용하지 않는 경로에서는 로드 안 함)
    - 순환 의존성 방지
    
    캐싱 전략:
    - 최초 호출 시 1회만 로드, 이후 캐시된 모듈 반환
    - 앱 재시작 전까지 유지 (글로벌 변수)
    
    Returns:
        inference 모듈 또는 None (로드 실패 시)
    
    부수 효과:
        sys.path에 scripts/finetuning 경로 추가
    """
    global _local_models
    
    if _local_models is None:
        try:
            import sys
            
            # scripts/finetuning을 Python 모듈 검색 경로에 추가
            # 이유: inference.py가 패키지 구조가 아니므로 명시적 경로 주입 필요
            finetuning_path = settings.project_root / "scripts" / "finetuning"
            if str(finetuning_path) not in sys.path:
                sys.path.insert(0, str(finetuning_path))
            
            import inference as local_models
            _local_models = local_models
            logger.info("inference 모듈 로드 성공")
            
        except ImportError as e:
            logger.warning(f"inference 모듈 로드 실패: {e}")
            _local_models = None
        except Exception as e:
            logger.error(f"inference 모듈 초기화 에러: {e}")
            _local_models = None
    
    return _local_models


def _get_stt_engine():
    """
    STT (Speech-to-Text) 엔진을 지연 로드합니다.
    
    지연 로딩 이유:
    - 텍스트 직접 분석 (/predict_text)에서는 STT 불필요
    - 모델 초기화 비용 절감 (faster-whisper 로드 약 2-3초)
    
    엔진 선택:
    - faster-whisper (CTranslate2 기반)
    - 이유: PyTorch 네이티브보다 3-4배 빠름, 메모리 효율적
    
    캐싱 전략:
    - 최초 호출 시 1회만 초기화
    - 앱 재시작 전까지 재사용
    
    Returns:
        STTEngine 인스턴스 또는 None (로드 실패 시)
    
    에러 처리:
    - 로드 실패 시 None 반환 (치명적 에러 아님)
    - 파이프라인은 STT 없이도 진행 가능 (사용자가 텍스트 직접 입력 가능)
    """
    global _stt_engine
    
    if _stt_engine is None:
        try:
            from src.services.stt_service import STTConfig, get_stt_engine
            
            # STT 설정
            # model="medium": 정확도와 속도의 균형점 (tiny < base < small < medium < large)
            # device="auto": CUDA 사용 가능하면 GPU, 아니면 CPU
            # language="ko": 한국어 최적화
            stt_config = STTConfig(
                engine="faster_whisper",
                model="medium",
                device="auto",
                language="ko",
                vad_enabled=False,  # VAD 비활성화 (청크 단위 처리이므로 불필요)
            )
            
            engine = get_stt_engine(stt_config)
            engine.initialize()  # 모델 로드
            
            _stt_engine = engine
            logger.info(f"STT 엔진 초기화 성공: {stt_config.model} on {stt_config.device}")
            
        except ImportError as e:
            logger.warning(f"STT 모듈 import 실패: {e}")
            _stt_engine = None
        except Exception as e:
            logger.error(f"STT 엔진 초기화 실패: {e}")
            _stt_engine = None
    
    return _stt_engine


# ============================================================================
# 리소스 정리 유틸리티
# ============================================================================


def _cleanup_old_runs() -> None:
    """
    메모리에 저장된 오래된 분석 작업을 정리합니다.
    
    정리 전략:
    - FIFO (First In First Out): 가장 오래된 항목부터 제거
    - 트리거: max_runs 제한 도달 시
    
    메모리 관리 이유:
    - 장기 실행 서버에서 무제한 누적 방지
    - 메모리 누수 방지
    
    설정값:
    - settings.max_runs (기본값: 100)
    
    부수 효과:
    - RUNS 딕셔너리에서 항목 제거
    """
    if len(RUNS) >= settings.max_runs:
        # 딕셔너리는 Python 3.7+에서 삽입 순서 유지
        # next(iter(RUNS)): 첫 번째 키 = 가장 오래된 항목
        oldest_run_id = next(iter(RUNS))
        del RUNS[oldest_run_id]
        logger.info(f"오래된 run 정리: {oldest_run_id}")


def _cleanup_temp_audio_files(directory: str) -> None:
    """
    임시 오디오 파일들을 정리합니다.
    
    정리 대상:
    - *.wav, *.mp3, *.flac, *.ogg 파일
    
    정리 시점:
    - 새 분석 시작 전 (이전 세션의 임시 파일 제거)
    
    에러 처리:
    - 파일 사용 중 (PermissionError): 경고 로그, 계속 진행
    - OS 에러: 경고 로그, 계속 진행
    - 이유: 정리 실패가 분석 시작을 막아서는 안 됨
    
    Args:
        directory: 정리할 디렉토리 경로
    """
    dir_path = Path(directory)
    audio_patterns = ("*.wav", "*.mp3", "*.flac", "*.ogg")

    for pattern in audio_patterns:
        for file_path in dir_path.glob(pattern):
            try:
                file_path.unlink()
                logger.debug(f"임시 파일 제거: {file_path}")
            except PermissionError:
                # 파일이 다른 프로세스에서 사용 중
                logger.warning(f"임시 파일 제거 실패 (사용 중): {file_path}")
            except OSError as e:
                # 파일 시스템 에러 (권한, 경로 등)
                logger.warning(f"임시 파일 제거 실패 {file_path}: {e}")


# ============================================================================
# 오디오 전처리
# ============================================================================


def _convert_to_wav(src_path: str) -> str:
    """
    오디오 파일을 STT에 최적화된 WAV 포맷으로 변환합니다.
    
    변환 파라미터:
    - 샘플레이트: 16kHz (음성 인식 표준, 전화 품질)
    - 채널: 모노 (스테레오 불필요, 파일 크기 절반)
    - 비트레이트: 16-bit (충분한 품질)
    
    ffmpeg 사용 이유:
    - 범용적: 거의 모든 오디오 포맷 지원 (MP3, FLAC, OGG, AAC 등)
    - 안정적: 산업 표준 도구
    - 빠름: 네이티브 C 구현
    
    Trade-off:
    - 장점: 모든 입력 포맷 처리 가능
    - 단점: 외부 의존성 (ffmpeg 설치 필요)
    
    Args:
        src_path: 원본 오디오 파일 경로
    
    Returns:
        변환된 WAV 파일 경로 (임시 디렉토리)
    
    Raises:
        AudioLoadError: ffmpeg 변환 실패 또는 미설치
    """
    src = Path(src_path)
    out_path = Path(tempfile.gettempdir()) / f"{src.stem}_conv.wav"

    # ffmpeg 명령어 구성
    # -y: 기존 파일 덮어쓰기 (확인 없이)
    # -i: 입력 파일
    # -ar 16000: 샘플레이트 16kHz (Whisper 권장)
    # -ac 1: 모노 채널 (음성 인식에 스테레오 불필요)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_path),
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,  # 출력 억제
            stderr=subprocess.DEVNULL,  # 에러 출력 억제
        )
    except subprocess.CalledProcessError as e:
        raise AudioLoadError(
            f"ffmpeg 오디오 변환 실패: {src_path}",
            details={"return_code": e.returncode, "command": " ".join(cmd)},
        )
    except FileNotFoundError:
        raise AudioLoadError(
            "ffmpeg가 설치되어 있지 않습니다. 설치: https://ffmpeg.org/download.html",
            details={"command": "ffmpeg"},
        )

    return str(out_path)


# ============================================================================
# 분석 파이프라인
# ============================================================================


def analysis_pipeline(
    audio_path: str,
    run_id: str,
    system_prompt: Optional[str] = None,
) -> None:
    """
    백그라운드에서 실행되는 음성 분석 파이프라인.
    
    실행 흐름:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 오디오 청크 분리                                      │
    │    ├─> soundfile로 5초 단위 읽기                        │
    │    └─> 임시 파일로 저장                                 │
    │                                                          │
    │ 2. 각 청크 STT 전사                                      │
    │    ├─> faster-whisper 추론                              │
    │    └─> 결과 텍스트 누적                                 │
    │                                                          │
    │ 3. 중간 LLM 분석 (30초마다)                              │
    │    ├─> 누적 텍스트로 보이스피싱 분석                    │
    │    └─> 백그라운드 스레드에서 실행                       │
    │                                                          │
    │ 4. 최종 분석                                             │
    │    ├─> 전체 텍스트로 최종 LLM 분석                      │
    │    └─> 상태 "all_complete"로 변경                       │
    └─────────────────────────────────────────────────────────┘
    
    왜 청크 단위 처리:
    - 실시간 피드백: 전체 파일 전사 대기 없이 부분 결과 제공
    - 메모리 효율: 전체 오디오를 메모리에 로드하지 않음
    - 조기 탐지: 30초 시점에 이미 1차 분석 완료
    
    왜 폴백 처리:
    - 청크 처리 실패 시 전체 파일 전사로 자동 복구
    - Graceful degradation 원칙
    
    스레드 안전성:
    - 백그라운드 스레드에서 실행 (daemon=True)
    - RUNS 딕셔너리 접근: 단일 run_id만 수정하므로 경합 최소화
    - 주의: 프로덕션에서는 Lock 또는 async/await 사용 권장
    
    Args:
        audio_path: 오디오 파일 경로
        run_id: 실행 고유 식별자 (UUID)
        system_prompt: LLM 시스템 프롬프트 (선택)
    
    부수 효과:
        - RUNS[run_id] 상태 업데이트
        - 임시 청크 파일 생성 및 정리
        - inference.ASYNC_RUNS 업데이트 (LLM 분석 시)
    """
    try:
        RUNS[run_id]["status"] = "processing_stt"
        logger.info(f"=== 분석 파이프라인 시작: run_id={run_id} ===")
        
        # 모듈 로드
        inference = _get_inference_module()
        stt_engine = _get_stt_engine()
        
        logger.info(f"모듈 상태: inference={inference is not None}, stt_engine={stt_engine is not None}")

        # STT 엔진 사용 불가 시 경고
        if stt_engine is None:
            logger.warning("STT 엔진을 사용할 수 없습니다. 음성 전사가 건너뛰어집니다.")
            RUNS[run_id]["stt_result"] = "[STT 엔진 초기화 실패]"
            RUNS[run_id]["stt_done"] = True
            RUNS[run_id]["status"] = "all_complete"
            return

        # 청크 처리 설정
        chunk_sec = settings.chunk_duration_sec  # 5초 (기본값)
        incremental_text = ""  # 누적 텍스트
        chunk_counter = 0
        ran_llm_analysis = False

        # =====================================================================
        # 청크 기반 처리
        # =====================================================================
        try:
            import soundfile as sf

            # 오디오 파일 열기 (실패 시 WAV 변환 시도)
            try:
                sf_obj = sf.SoundFile(audio_path)
            except Exception as e:
                logger.info(f"오디오 파일 직접 열기 실패 ({e}), WAV 변환 시도...")
                conv_path = _convert_to_wav(audio_path)
                sf_obj = sf.SoundFile(conv_path)
                audio_path = conv_path

            sr = sf_obj.samplerate  # 샘플레이트
            frames_per_chunk = int(sr * chunk_sec)  # 5초 = 샘플레이트 * 5
            tmp_dir = tempfile.mkdtemp(prefix=f"stt_chunks_{run_id}_")
            chunk_files = []

            try:
                n = 1
                logger.info(f"청크 처리 시작: sr={sr}, frames_per_chunk={frames_per_chunk}")
                
                while True:
                    # 청크 읽기 (5초 단위)
                    frames = sf_obj.read(frames_per_chunk, always_2d=True)
                    if frames is None or getattr(frames, "size", 0) == 0:
                        logger.info(f"파일 끝 도달: 총 {chunk_counter}개 청크 처리됨")
                        break  # 파일 끝 도달

                    # 청크를 임시 WAV 파일로 저장
                    chunk_path = os.path.join(tmp_dir, f"chunk_{n:04d}.wav")
                    try:
                        sf.write(chunk_path, frames, sr)
                    except Exception as e:
                        logger.warning(f"청크 저장 실패 {chunk_path}: {e}")
                        n += 1
                        continue

                    # STT 전사
                    seg_text = ""
                    if stt_engine is not None:
                        try:
                            # FasterWhisperEngine.transcribe() 호출
                            # 반환값: TranscriptionResult(text, segments, engine, error)
                            result = stt_engine.transcribe(chunk_path)
                            seg_text = result.text.strip()
                            logger.info(f"청크 {n} 전사 완료: '{seg_text[:50]}...' ({len(seg_text)} chars)")
                        except Exception as e:
                            logger.warning(f"청크 전사 실패 {chunk_path}: {e}")

                    # 전사 텍스트 누적
                    if seg_text:
                        incremental_text = (incremental_text + " " + seg_text).strip()
                        RUNS[run_id]["stt_result"] = incremental_text

                    chunk_files.append(chunk_path)
                    chunk_counter += 1
                    n += 1
                    
                    logger.info(f"청크 카운터: {chunk_counter}, 다음 분석까지: {settings.analysis_interval_chunks - (chunk_counter % settings.analysis_interval_chunks)}")

                    # 실시간 시뮬레이션 제거 (빠른 테스트를 위해)
                    # 원래: time.sleep(chunk_sec)
                    # 이유: 디버깅 중에는 빠른 처리가 필요함
                    # TODO: 프로덕션에서 실시간 시뮬레이션이 필요하면 아래 주석 해제
                    # time.sleep(chunk_sec)

                    # 30초마다 중간 LLM 분석 (6개 청크 = 5초 * 6 = 30초)
                    if (chunk_counter % settings.analysis_interval_chunks) == 0:
                        logger.info(f"중간 LLM 분석 트리거: chunk_counter={chunk_counter}, text_len={len(incremental_text)}")
                        if inference is not None and incremental_text:
                            try:
                                RUNS[run_id]["status"] = "processing_llm"
                                
                                # PLM 모델 사전 로드 (없으면 자동 로드)
                                # 이유: start_async_analysis()는 PLM이 로드되어 있어야 함
                                if inference.MODELS.plm_model is None:
                                    logger.info("PLM 모델 로드 중...")
                                    try:
                                        inference.load_plm()
                                        logger.info("PLM 모델 로드 완료")
                                    except Exception as plm_err:
                                        logger.error(f"PLM 로드 실패: {plm_err}")
                                        raise
                                
                                # inference.start_async_analysis() 호출
                                # 참고: start_local_async_analysis (X)
                                logger.info("LLM 분석 시작...")
                                inference.start_async_analysis(
                                    conversation_text=incremental_text,
                                    system_prompt_path="./system_ko.txt",
                                    run_id=run_id
                                )
                                ran_llm_analysis = True
                                logger.info(f"LLM 분석 스레드 시작 완료: run_id={run_id}")
                                
                            except Exception as e:
                                logger.warning(f"중간 LLM 분석 시작 실패: {e}")
                            finally:
                                RUNS[run_id]["status"] = "processing_stt"
                        else:
                            logger.warning(f"LLM 분석 건너뜀: inference={inference is not None}, text_len={len(incremental_text)}")

                # while 루프 종료 (모든 청크 처리 완료)
                full_text = incremental_text
                logger.info(f"모든 청크 처리 완료: chunk_counter={chunk_counter}, full_text_len={len(full_text)}")

                # ================================================================
                # 최종 LLM 분석 (항상 실행)
                # ================================================================
                # 이전 로직: 청크 수가 6의 배수가 아닐 때만 실행
                # 문제점: 정확히 6개, 12개 등의 청크면 최종 분석 건너뜀
                # 수정: 텍스트가 있으면 항상 최종 분석 실행
                if inference is not None and full_text:
                    try:
                        RUNS[run_id]["status"] = "processing_llm"
                        logger.info(f"최종 LLM 분석 시작: {len(full_text)} chars, chunk_counter={chunk_counter}")
                        
                        # PLM 모델 사전 로드
                        if inference.MODELS.plm_model is None:
                            logger.info("PLM 모델 로드 중 (최종 분석)...")
                            try:
                                inference.load_plm()
                                logger.info("PLM 모델 로드 완료")
                            except Exception as plm_err:
                                logger.error(f"PLM 로드 실패: {plm_err}")
                                raise
                        
                        inference.start_async_analysis(
                            conversation_text=full_text,
                            system_prompt_path="./system_ko.txt",
                            run_id=run_id
                        )
                        ran_llm_analysis = True
                        logger.info(f"최종 LLM 분석 스레드 시작 완료: run_id={run_id}")
                    except Exception as e:
                        logger.error(f"최종 LLM 분석 실패: {e}")
                else:
                    logger.warning(f"최종 LLM 분석 건너뜀: inference={inference is not None}, full_text_len={len(full_text) if full_text else 0}")

                RUNS[run_id]["stt_done"] = True

            finally:
                # 리소스 정리 (무조건 실행)
                # 이유: 예외 발생해도 리소스는 반드시 해제해야 함
                try:
                    sf_obj.close()
                except Exception:
                    pass

                # 임시 청크 파일 정리
                for p in chunk_files:
                    try:
                        if os.path.exists(p):
                            os.unlink(p)
                    except Exception:
                        pass
                
                # 임시 디렉토리 제거
                try:
                    if os.path.isdir(tmp_dir):
                        os.rmdir(tmp_dir)
                except Exception:
                    pass

        except Exception as chunk_error:
            # ================================================================
            # 폴백: 청크 처리 실패 시 전체 파일 전사
            # ================================================================
            logger.warning(f"청크 처리 실패, 전체 파일 전사로 폴백: {chunk_error}")

            full_text = ""
            if stt_engine is not None:
                try:
                    # 전체 파일 직접 전사 시도
                    result = stt_engine.transcribe(audio_path)
                    full_text = result.text.strip()
                except Exception:
                    # WAV 변환 후 재시도
                    try:
                        conv_path = _convert_to_wav(audio_path)
                        result = stt_engine.transcribe(conv_path)
                        full_text = result.text.strip()
                    except Exception as e:
                        logger.error(f"전체 파일 전사 실패: {e}")

            RUNS[run_id]["stt_result"] = full_text or ""
            RUNS[run_id]["stt_done"] = True

            # 폴백 LLM 분석
            if inference is not None and full_text:
                try:
                    RUNS[run_id]["status"] = "processing_llm"
                    
                    # PLM 모델 사전 로드
                    if inference.MODELS.plm_model is None:
                        try:
                            inference.load_plm()
                        except Exception as plm_err:
                            logger.error(f"PLM 로드 실패: {plm_err}")
                            raise
                    
                    inference.start_async_analysis(
                        conversation_text=full_text,
                        system_prompt_path="./system_ko.txt",
                        run_id=run_id
                    )
                    ran_llm_analysis = True
                except Exception as e:
                    logger.warning(f"폴백 LLM 분석 실패: {e}")

        # 최종 상태 설정
        if ran_llm_analysis:
            # LLM 분석 진행 중 (백그라운드 스레드에서 완료 후 상태 변경)
            RUNS[run_id]["status"] = "processing_llm"
        else:
            # LLM 분석 없음 (inference 모듈 없거나 텍스트 없음)
            RUNS[run_id]["status"] = "all_complete"

    except VoicePhishingProtectorError as e:
        # 커스텀 예외: 예상된 비즈니스 로직 에러
        RUNS[run_id]["status"] = "error"
        RUNS[run_id]["llm_result"] = f"처리 오류: {e.message}"
        logger.error(f"분석 파이프라인 오류 (run_id={run_id}): {e}")
    except (OSError, IOError) as e:
        # 파일/IO 관련 시스템 오류
        RUNS[run_id]["status"] = "error"
        RUNS[run_id]["llm_result"] = f"파일 처리 오류: {e}"
        logger.error(f"분석 파이프라인 IO 오류 (run_id={run_id}): {e}")
    except Exception as e:
        # 예상치 못한 오류 (버그 가능성)
        RUNS[run_id]["status"] = "error"
        RUNS[run_id]["llm_result"] = f"예상치 못한 오류: {type(e).__name__}"
        logger.exception(f"분석 파이프라인 예외 (run_id={run_id}): {e}")


# ============================================================================
# API 엔드포인트
# ============================================================================


@router.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(
    file: UploadFile = File(...),
    sys_prompt: Optional[str] = Form(None),
):
    """
    오디오 파일 분석을 시작합니다 (비동기).
    
    동작 방식:
    1. 클라이언트가 오디오 파일 업로드
    2. 서버가 파일 저장 및 run_id 생성
    3. 백그라운드 스레드에서 분석 파이프라인 시작
    4. 즉시 run_id 반환 (클라이언트는 /status/{run_id}로 진행상황 폴링)
    
    비동기 패턴 이유:
    - 긴 처리 시간 (수 분): HTTP 타임아웃 방지
    - 클라이언트 블로킹 방지: UI 반응성 유지
    - 확장성: 동시 다중 분석 요청 처리
    
    파일 저장 위치:
    - 임시 디렉토리: pages/{run_id}.{ext}
    - 이유: 분석 후 자동 정리, 영구 저장 불필요
    
    Args:
        file: 업로드된 오디오 파일 (MP3, WAV, FLAC, OGG)
        sys_prompt: LLM 분석에 사용할 시스템 프롬프트 (선택)
    
    Returns:
        AnalyzeResponse: run_id와 메시지
    
    Raises:
        HTTPException 500: 파일 저장 실패
    """
    # 오래된 분석 작업 정리 (메모리 관리)
    _cleanup_old_runs()

    # 저장 디렉토리 준비
    pages_dir = os.path.join(os.getcwd(), "pages")
    os.makedirs(pages_dir, exist_ok=True)

    # 이전 세션의 임시 파일 정리
    _cleanup_temp_audio_files(pages_dir)

    # 파일 확장자 추출 및 검증
    orig_name = getattr(file, "filename", None) or "uploaded"
    _, ext = os.path.splitext(orig_name)
    if not ext:
        ext = ".wav"  # 기본 확장자

    # 고유 run_id 생성 (UUID4)
    # 이유: 충돌 없는 고유 식별자, 예측 불가능성 (보안)
    run_id = uuid.uuid4().hex
    tmp_audio_path = os.path.join(pages_dir, f"{run_id}{ext}")

    # 파일 저장
    try:
        # 파일 포인터를 처음으로 되돌림
        # 이유: FastAPI가 파일을 이미 읽었을 수 있음
        try:
            file.file.seek(0)
        except Exception:
            pass  # seek 실패는 치명적이지 않음

        with open(tmp_audio_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    # 저장된 파일 크기 확인 (로깅용)
    try:
        size = os.path.getsize(tmp_audio_path)
    except Exception:
        size = None

    logger.info(f"파일 저장 완료: {tmp_audio_path} (원본={orig_name}, 크기={size} bytes)")

    # 분석 상태 초기화
    RUNS[run_id] = {
        "status": "pending",  # pending -> processing_stt -> processing_llm -> all_complete
        "stt_result": "",
        "llm_result": "",
        "stt_done": False,
        "audio_path": tmp_audio_path,
        "created_at": time.time(),
    }

    # 백그라운드 스레드에서 분석 파이프라인 시작
    # daemon=True: 메인 프로세스 종료 시 자동 종료
    thread = threading.Thread(
        target=analysis_pipeline,
        args=(tmp_audio_path, run_id, sys_prompt),
        daemon=True,
    )
    thread.start()

    return AnalyzeResponse(run_id=run_id, message="분석이 시작되었습니다.")


@router.get("/status/{run_id}", response_model=AnalysisStatus)
async def get_status(
    run_id: str,
    wait: Optional[int] = None,
    timeout: int = 25,
):
    """
    분석 상태를 조회합니다 (long-polling 지원).
    
    동작 모드:
    1. 일반 모드 (wait=None): 즉시 현재 상태 반환
    2. Long-polling 모드 (wait=1): 완료될 때까지 대기 후 반환
    
    Long-polling 이유:
    - 클라이언트 폴링 간격 최적화 (네트워크 오버헤드 감소)
    - 서버 부하 감소 (빈번한 요청 방지)
    - 실시간성 향상 (완료 즉시 응답)
    
    상태 병합 로직:
    - RUNS (STT 상태) + inference.ASYNC_RUNS (LLM 상태) 병합
    - 이유: STT와 LLM이 독립적인 스레드에서 실행되므로 상태 통합 필요
    
    Args:
        run_id: 분석 작업 식별자
        wait: Long-polling 활성화 (값이 있으면 완료까지 대기)
        timeout: Long-polling 최대 대기 시간 (초, 기본 25초)
    
    Returns:
        AnalysisStatus: 현재 분석 상태 (stt_result, llm_result, status 등)
    
    Raises:
        HTTPException 404: run_id를 찾을 수 없는 경우
    """
    run_data = RUNS.get(run_id)

    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    # 디버그 로깅
    stt_len = len(run_data.get("stt_result", "") or "")
    llm_len = len(run_data.get("llm_result", "") or "")
    logger.debug(
        f"상태 조회: run_id={run_id}, status={run_data.get('status')}, "
        f"stt_len={stt_len}, llm_len={llm_len}"
    )

    inference = _get_inference_module()

    # ========================================================================
    # Long-polling 처리
    # ========================================================================
    # wait 파라미터가 있으면 완료될 때까지 대기
    # 최대 timeout초까지만 대기 (무한 대기 방지)
    if wait is not None:
        start_ts = time.time()
        
        while True:
            merged = dict(run_data)

            # inference 모듈의 LLM 분석 상태 병합
            if inference is not None:
                # inference.ASYNC_RUNS에서 이 run_id의 상태 가져오기
                # 참고: LOCAL_ASYNC (X), ASYNC_RUNS (O)
                async_runs = getattr(inference, "ASYNC_RUNS", {}) or {}
                if run_id in async_runs:
                    merged.update(async_runs[run_id])

            # 완료 조건 체크
            cur_status = merged.get("status")
            if cur_status in ("all_complete", "error"):
                break  # 분석 완료 또는 에러 발생
            
            # 타임아웃 체크
            if (time.time() - start_ts) >= float(timeout):
                logger.warning(f"Long-polling 타임아웃: run_id={run_id}")
                break

            # 0.5초 대기 후 재확인
            # 이유: CPU 사용률 최소화, 충분히 빠른 응답성
            time.sleep(0.5)

    # ========================================================================
    # 최종 상태 병합 및 반환
    # ========================================================================
    merged = dict(run_data)

    if inference is not None:
        async_runs = getattr(inference, "ASYNC_RUNS", {}) or {}
        if run_id in async_runs:
            llm_entry = async_runs[run_id]
            merged.update(llm_entry)

            # LLM 분석 완료 시 결과 포맷팅
            if llm_entry.get("status") == "complete":
                merged["llm_result"] = _format_llm_result(llm_entry)
                merged["status"] = "all_complete"

    # RUNS 딕셔너리에 최종 결과 저장
    # 이유: 다음 조회 시 병합 과정 생략 가능
    if merged.get("llm_result"):
        RUNS[run_id]["llm_result"] = merged.get("llm_result")

    if merged.get("status") == "all_complete" and RUNS[run_id].get("stt_done"):
        RUNS[run_id]["status"] = "all_complete"

    return AnalysisStatus(**merged)


@router.post("/predict_text", response_model=PredictTextResponse)
async def predict_text(
    run_id: str = Form(...),
    text: str = Form(...),
):
    """
    텍스트를 직접 분석합니다 (STT 건너뛰기).
    
    사용 사례:
    - 이미 전사된 텍스트가 있는 경우
    - STT 없이 LLM 분석만 필요한 경우
    - 테스트 및 디버깅
    
    Args:
        run_id: 분석 작업 식별자
        text: 분석할 텍스트 (대화 내용)
    
    Returns:
        PredictTextResponse: run_id와 메시지
    
    Raises:
        HTTPException 503: inference 모듈을 사용할 수 없는 경우
        HTTPException 500: 분석 시작 실패
    """
    inference = _get_inference_module()

    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="inference 모듈을 사용할 수 없습니다. 모델 파일을 확인하세요."
        )

    try:
        # PLM 모델 로드 (필수)
        # 이유: start_async_analysis()는 PLM이 로드되어 있어야 동작
        if inference.MODELS.plm_model is None:
            try:
                logger.info("PLM 모델 로드 중...")
                inference.load_plm()
                logger.info("PLM 모델 로드 완료")
            except Exception as plm_err:
                logger.error(f"PLM 로드 실패: {plm_err}")
                raise HTTPException(
                    status_code=500,
                    detail=f"PLM 모델 로드 실패: {plm_err}"
                )

        # LLM 분석 시작
        # 참고: start_local_async_analysis (X), start_async_analysis (O)
        inference.start_async_analysis(
            conversation_text=text,
            system_prompt_path="./system_ko.txt",
            run_id=run_id
        )

    except Exception as e:
        logger.error(f"텍스트 분석 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 시작 실패: {e}")

    return PredictTextResponse(run_id=run_id, message="분석이 시작되었습니다.")


# ============================================================================
# 헬퍼 함수
# ============================================================================


def _format_llm_result(llm_entry: Dict[str, Any]) -> str:
    """
    LLM 분석 결과를 사람이 읽기 쉬운 형식으로 포맷팅합니다.
    
    입력 형식 (inference.ASYNC_RUNS[run_id]):
        {
            "PLM_risk_score": 0.85,
            "LLM_risk_score": 0.72,
            "comprehensive_risk_score": 0.785,
            "reasoning": "...",
            "key_evidence": ["...", "..."],
            "llm_text": "{...}"  # 원본 LLM 출력 (JSON)
        }
    
    출력 형식:
        종합 점수: 0.785
        PLM 점수: 0.850
        LLM 점수: 0.720
        설명: ...
        증거: ..., ...
    
    Args:
        llm_entry: inference.ASYNC_RUNS에서 가져온 분석 결과
    
    Returns:
        포맷팅된 결과 문자열
    
    에러 처리:
        파싱 실패 시 원본 데이터를 JSON 문자열로 반환 (최소한의 정보 제공)
    """
    llm_text = llm_entry.get("llm_text") or ""
    reasoning = llm_entry.get("reasoning") or ""

    try:
        # JSON 파싱 시도 (llm_text는 LLM이 생성한 원본 JSON)
        parsed = None
        if llm_text:
            try:
                parsed = json.loads(llm_text)
            except json.JSONDecodeError:
                pass  # JSON이 아닌 경우 무시

        if isinstance(parsed, dict):
            parts = []

            # 점수 정보 추가
            comp = llm_entry.get("comprehensive_risk_score")
            if comp is not None:
                parts.append(f"종합 점수: {float(comp):.3f}")

            plm = llm_entry.get("PLM_risk_score")
            if plm is not None:
                parts.append(f"PLM 점수: {float(plm):.3f}")

            llm_sc = llm_entry.get("LLM_risk_score")
            if llm_sc is not None:
                parts.append(f"LLM 점수: {float(llm_sc):.3f}")

            # 설명 추가
            parsed_reason = parsed.get("reasoning")
            if parsed_reason:
                parts.append(f"설명: {parsed_reason}")
            elif reasoning:
                parts.append(f"설명: {reasoning}")

            # 근거 추가
            ev = parsed.get("key_evidence")
            if ev:
                parts.append("증거: " + ", ".join(map(str, ev)))

            if parts:
                return "\n".join(parts)
            
            # parts가 비어있으면 원본 JSON 반환
            return json.dumps(parsed, ensure_ascii=False)

        # JSON 파싱 실패 시 원본 텍스트 반환
        return llm_text or reasoning or json.dumps(llm_entry, ensure_ascii=False)

    except Exception as e:
        # 포맷팅 실패 시 최소한의 정보라도 반환
        logger.warning(f"LLM 결과 포맷팅 실패: {e}")
        return llm_text or reasoning or str(llm_entry)

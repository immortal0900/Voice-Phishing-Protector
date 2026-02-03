"""
STT 서비스 모듈 (Speech-to-Text Service)

여러 STT 엔진을 통합하여 일관된 인터페이스를 제공합니다.
Strategy 패턴을 적용하여 엔진 교체가 용이합니다.

해결 이슈:
- RC-011: STT 로직 중복 → 단일 서비스로 통합
- RD-008: 긴 함수 분리 → 책임별 클래스 분리

지원 엔진:
- FasterWhisper: 빠르고 정확한 로컬 STT (GPU 권장)
- GoogleSTT: Google Web API 기반 (네트워크 필요)

사용 예시:
    from src.services.stt_service import get_stt_engine, STTConfig

    config = STTConfig(engine="faster_whisper", model="small")
    engine = get_stt_engine(config)
    text = engine.transcribe("audio.wav")
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """전사 세그먼트 결과

    Attributes:
        index: 세그먼트 인덱스
        start: 시작 시간 (초)
        end: 종료 시간 (초)
        text: 전사된 텍스트
        speaker: 화자 ID (화자 분리 시)
        confidence: 신뢰도 (지원하는 엔진만)
    """

    index: int
    start: float
    end: float
    text: str
    speaker: str = "speaker_1"
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """전사 결과

    Attributes:
        text: 전체 전사 텍스트
        segments: 세그먼트별 결과
        duration_sec: 오디오 길이 (초)
        language: 감지된 언어
        engine: 사용된 엔진명
        error: 에러 메시지 (실패 시)
    """

    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration_sec: float = 0.0
    language: str = "ko"
    engine: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """전사 성공 여부"""
        return self.error is None and bool(self.text)


@dataclass
class STTConfig:
    """STT 설정

    Attributes:
        engine: 사용할 엔진 ("faster_whisper", "google")
        model: 모델 크기/이름 (faster_whisper 전용)
        device: 실행 디바이스 ("cuda", "cpu", "auto")
        compute_type: 연산 타입 ("float16", "float32", "int8")
        language: 인식 언어
        chunk_duration_sec: 청크 길이 (초)
        vad_enabled: VAD(Voice Activity Detection) 활성화
    """

    engine: Literal["faster_whisper", "google"] = "faster_whisper"
    model: str = "small"
    device: str = "auto"
    compute_type: str = "float16"
    language: str = "ko"
    chunk_duration_sec: int = 5
    vad_enabled: bool = True

    @classmethod
    def from_settings(cls) -> "STTConfig":
        """settings에서 설정을 로드합니다."""
        return cls(
            model=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            language="ko",
            chunk_duration_sec=settings.chunk_duration_sec,
        )


class STTEngine(ABC):
    """STT 엔진 추상 베이스 클래스

    새로운 STT 엔진을 추가하려면 이 클래스를 상속받아 구현하세요.
    """

    def __init__(self, config: STTConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """엔진을 초기화합니다.

        무거운 모델 로딩은 여기서 수행됩니다.
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """오디오 파일을 전사합니다.

        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (None이면 자동 감지 또는 기본값)

        Returns:
            TranscriptionResult 객체
        """
        pass

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """오디오 청크를 전사합니다.

        실시간 처리용 메서드입니다.

        Args:
            audio_data: 오디오 바이트 데이터
            sample_rate: 샘플레이트

        Returns:
            전사된 텍스트
        """
        pass

    def ensure_initialized(self) -> None:
        """초기화 여부를 확인하고 필요 시 초기화합니다."""
        if not self._initialized:
            self.initialize()
            self._initialized = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"


class FasterWhisperEngine(STTEngine):
    """Faster Whisper 기반 STT 엔진

    CTranslate2 기반으로 빠르고 효율적인 전사를 제공합니다.
    GPU 사용 시 실시간보다 빠른 처리가 가능합니다.

    Attributes:
        model: WhisperModel 인스턴스
    """

    def __init__(self, config: STTConfig):
        super().__init__(config)
        self.model = None

    def initialize(self) -> None:
        """Faster Whisper 모델을 로드합니다."""
        try:
            from faster_whisper import WhisperModel

            device = self.config.device
            if device == "auto":
                # CUDA 사용 가능 여부 확인
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            logger.info(
                f"Faster Whisper 초기화: model={self.config.model}, "
                f"device={device}, compute_type={self.config.compute_type}"
            )

            self.model = WhisperModel(
                self.config.model,
                device=device,
                compute_type=self.config.compute_type if device == "cuda" else "float32",
            )
            self._initialized = True

        except ImportError:
            raise ImportError(
                "faster-whisper 패키지가 설치되지 않았습니다: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"Faster Whisper 초기화 실패: {e}")
            raise

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """오디오 파일을 전사합니다."""
        self.ensure_initialized()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return TranscriptionResult(
                text="",
                engine=self.__class__.__name__,
                error=f"파일을 찾을 수 없습니다: {audio_path}",
            )

        try:
            lang = language or self.config.language
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=5,
                vad_filter=self.config.vad_enabled,
            )

            segments: List[TranscriptionSegment] = []
            texts: List[str] = []

            for idx, seg in enumerate(segments_iter):
                text = (seg.text or "").strip()
                if text:
                    segments.append(
                        TranscriptionSegment(
                            index=idx,
                            start=seg.start,
                            end=seg.end,
                            text=text,
                        )
                    )
                    texts.append(text)

            full_text = " ".join(texts)

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                duration_sec=info.duration if hasattr(info, "duration") else 0.0,
                language=info.language if hasattr(info, "language") else lang,
                engine=self.__class__.__name__,
            )

        except Exception as e:
            logger.error(f"전사 실패 ({audio_path}): {e}")
            return TranscriptionResult(
                text="",
                engine=self.__class__.__name__,
                error=str(e),
            )

    def transcribe_chunk(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """오디오 청크를 전사합니다."""
        self.ensure_initialized()

        # 임시 파일로 저장 후 전사
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                # soundfile로 WAV 저장
                import numpy as np
                import soundfile as sf

                # bytes를 numpy array로 변환 (16-bit PCM 가정)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_array /= 32768.0  # 정규화

                sf.write(tmp_path, audio_array, sample_rate)

                result = self.transcribe(tmp_path)
                return result.text

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


class GoogleSTTEngine(STTEngine):
    """Google Web API 기반 STT 엔진

    SpeechRecognition 라이브러리를 통해 Google의 무료 STT API를 사용합니다.

    주의:
    - 네트워크 연결 필요
    - 요청 제한이 있을 수 있음
    - 긴 오디오는 청크로 분할 필요
    """

    def __init__(self, config: STTConfig):
        super().__init__(config)
        self.recognizer = None

    def initialize(self) -> None:
        """SpeechRecognition 인식기를 초기화합니다."""
        try:
            import speech_recognition as sr

            self.recognizer = sr.Recognizer()
            self._initialized = True
            logger.info("Google STT 엔진 초기화 완료")

        except ImportError:
            raise ImportError(
                "SpeechRecognition 패키지가 설치되지 않았습니다: "
                "pip install SpeechRecognition"
            )

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """오디오 파일을 전사합니다."""
        self.ensure_initialized()

        import speech_recognition as sr
        from pydub import AudioSegment

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return TranscriptionResult(
                text="",
                engine=self.__class__.__name__,
                error=f"파일을 찾을 수 없습니다: {audio_path}",
            )

        try:
            # pydub로 오디오 로드 및 전처리
            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_frame_rate(16000).set_channels(1)
            duration_sec = len(audio) / 1000.0

            # 청크로 분할 (긴 오디오 처리)
            chunk_ms = self.config.chunk_duration_sec * 1000
            segments: List[TranscriptionSegment] = []
            texts: List[str] = []

            for idx, start_ms in enumerate(range(0, len(audio), chunk_ms)):
                chunk = audio[start_ms : start_ms + chunk_ms]
                if len(chunk) < 500:  # 0.5초 미만 스킵
                    continue

                # WAV로 변환하여 인식
                with io.BytesIO() as buf:
                    chunk.export(buf, format="wav")
                    buf.seek(0)

                    with sr.AudioFile(buf) as source:
                        audio_data = self.recognizer.record(source)

                try:
                    lang = language or f"{self.config.language}-KR"
                    if not "-" in lang:
                        lang = f"{lang}-KR"

                    text = self.recognizer.recognize_google(audio_data, language=lang)
                    text = text.strip()

                    if text:
                        start_sec = start_ms / 1000.0
                        end_sec = min((start_ms + chunk_ms) / 1000.0, duration_sec)

                        segments.append(
                            TranscriptionSegment(
                                index=idx,
                                start=start_sec,
                                end=end_sec,
                                text=text,
                            )
                        )
                        texts.append(text)

                except sr.UnknownValueError:
                    # 음성 인식 불가 (무음 등)
                    pass
                except sr.RequestError as e:
                    logger.warning(f"Google STT 요청 오류: {e}")

            full_text = " ".join(texts)

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                duration_sec=duration_sec,
                language=self.config.language,
                engine=self.__class__.__name__,
            )

        except Exception as e:
            logger.error(f"전사 실패 ({audio_path}): {e}")
            return TranscriptionResult(
                text="",
                engine=self.__class__.__name__,
                error=str(e),
            )

    def transcribe_chunk(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """오디오 청크를 전사합니다."""
        self.ensure_initialized()

        import speech_recognition as sr
        from pydub import AudioSegment

        try:
            # bytes를 AudioSegment로 변환
            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=sample_rate,
                channels=1,
            )

            with io.BytesIO() as buf:
                audio.export(buf, format="wav")
                buf.seek(0)

                with sr.AudioFile(buf) as source:
                    audio_sr = self.recognizer.record(source)

            lang = f"{self.config.language}-KR"
            return self.recognizer.recognize_google(audio_sr, language=lang).strip()

        except Exception as e:
            logger.warning(f"청크 전사 실패: {e}")
            return ""


def get_stt_engine(config: Optional[STTConfig] = None) -> STTEngine:
    """팩토리 함수: 설정에 따라 적절한 STT 엔진을 반환합니다.

    Args:
        config: STT 설정 (None이면 기본 설정 사용)

    Returns:
        STTEngine 구현체

    Example:
        >>> engine = get_stt_engine()
        >>> result = engine.transcribe("audio.wav")
        >>> print(result.text)
    """
    if config is None:
        config = STTConfig.from_settings()

    if config.engine == "faster_whisper":
        return FasterWhisperEngine(config)
    elif config.engine == "google":
        return GoogleSTTEngine(config)
    else:
        raise ValueError(
            f"알 수 없는 STT 엔진: {config.engine}. "
            "지원 엔진: faster_whisper, google"
        )


# 싱글톤 인스턴스 (선택적 사용)
_default_engine: Optional[STTEngine] = None


def get_default_engine() -> STTEngine:
    """기본 STT 엔진을 반환합니다.

    싱글톤 패턴으로 한 번만 생성됩니다.
    모델 로딩 비용을 절감할 수 있습니다.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = get_stt_engine()
    return _default_engine


if __name__ == "__main__":
    print("=== STT 서비스 테스트 ===")

    # 설정 출력
    config = STTConfig.from_settings()
    print(f"기본 설정: {config}")

    # Faster Whisper 엔진 테스트
    try:
        engine = get_stt_engine(STTConfig(engine="faster_whisper", model="tiny"))
        print(f"Faster Whisper 엔진 생성됨: {engine}")
    except Exception as e:
        print(f"Faster Whisper 테스트 실패: {e}")

"""
공통 유틸리티 모듈 (Common Utilities)

여러 모듈에서 공통으로 사용하는 유틸리티 함수들을 제공합니다.

해결 이슈:
- RC-013: JSON 파싱 로직 중복 → 단일 모듈로 통합

사용 예시:
    from src.core.utils import parse_llm_json_response

    result = parse_llm_json_response(llm_output)
    print(result["risk_score"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class LLMParseResult:
    """LLM 응답 파싱 결과

    Attributes:
        risk_score: 위험 점수 (0.0 ~ 1.0 또는 0 ~ 100)
        reasoning: 판단 근거
        key_evidence: 핵심 증거 목록
        classification: 분류 결과 (예: "보이스피싱", "일상 대화")
        is_phishing: 보이스피싱 여부
        raw_data: 파싱된 원본 딕셔너리
        parse_success: 파싱 성공 여부
    """

    risk_score: float
    reasoning: str
    key_evidence: List[str]
    classification: str
    is_phishing: bool
    raw_data: Dict[str, Any]
    parse_success: bool


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """텍스트에서 JSON 객체를 추출합니다.

    LLM 응답에는 JSON 외에 추가 텍스트가 포함될 수 있으므로,
    정규표현식으로 첫 번째 {...} 블록을 찾아 파싱합니다.

    Args:
        text: LLM 응답 텍스트

    Returns:
        파싱된 딕셔너리 또는 None (파싱 실패 시)

    Note:
        이 함수가 없을 경우, LLM이 "JSON 외에 설명을 덧붙인 경우"
        전체 파싱이 실패하여 서비스 오류가 발생합니다.
    """
    if not text or not isinstance(text, str):
        return None

    # 중괄호로 둘러싸인 JSON 블록 찾기 (중첩 지원)
    # re.DOTALL로 줄바꿈도 포함하여 매칭
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 중첩된 JSON에서 마지막 닫는 괄호가 잘린 경우 복구 시도
        # 열린 괄호 수와 닫힌 괄호 수 맞추기
        open_count = json_str.count("{")
        close_count = json_str.count("}")
        if open_count > close_count:
            json_str += "}" * (open_count - close_count)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return None


def extract_score_from_text(text: str) -> Optional[float]:
    """텍스트에서 숫자 점수를 추출합니다.

    JSON 파싱이 실패했을 때 폴백으로 사용합니다.
    0-1 범위 또는 0-100 범위의 숫자를 찾습니다.

    Args:
        text: LLM 응답 텍스트

    Returns:
        추출된 점수 (0.0 ~ 1.0 범위로 정규화) 또는 None
    """
    if not text:
        return None

    # 패턴 1: 소수점 포함 0-1 범위 (예: 0.85, 0.7)
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if match:
        score = float(match.group(1))
        if 0.0 <= score <= 1.0:
            return score

    # 패턴 2: 0-100 범위 정수/소수 (예: 85, 75.5)
    match = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", text)
    if match:
        score = float(match.group(1))
        if 0 <= score <= 100:
            return score / 100.0  # 0-1 범위로 정규화

    return None


def parse_llm_json_response(
    response: Union[str, Dict[str, Any]],
    default_score: float = 0.0,
) -> LLMParseResult:
    """LLM 응답을 파싱하여 구조화된 결과를 반환합니다.

    여러 스키마 형식을 지원합니다:
    1. 신규 스키마: classification, risk_score, reasoning, key_evidence
    2. 앙상블 스키마: comprehensive_risk_score, PLM_risk_score, LLM_risk_score
    3. 레거시 스키마: phishing, score, reason

    Args:
        response: LLM 응답 (문자열 또는 이미 파싱된 딕셔너리)
        default_score: 파싱 실패 시 기본 점수

    Returns:
        LLMParseResult 객체

    Example:
        >>> result = parse_llm_json_response('{"risk_score": 0.85, "reasoning": "긴급 송금 요청"}')
        >>> print(result.risk_score)  # 0.85
        >>> print(result.is_phishing)  # True (0.5 이상)
    """
    parsed: Optional[Dict[str, Any]] = None
    raw_text = ""

    # 입력 타입에 따른 처리
    if isinstance(response, dict):
        parsed = response
    elif isinstance(response, str):
        raw_text = response.strip()
        parsed = extract_json_from_text(raw_text)

    # 파싱 실패 시 폴백
    if parsed is None:
        fallback_score = extract_score_from_text(raw_text) if raw_text else None
        return LLMParseResult(
            risk_score=fallback_score if fallback_score is not None else default_score,
            reasoning=raw_text[:500] if raw_text else "",
            key_evidence=[],
            classification="unknown",
            is_phishing=(fallback_score or default_score) >= 0.5,
            raw_data={},
            parse_success=False,
        )

    # 스키마에 따른 필드 추출
    risk_score = _extract_risk_score(parsed)
    reasoning = _extract_reasoning(parsed)
    key_evidence = _extract_key_evidence(parsed)
    classification = _extract_classification(parsed)

    # 점수 정규화 (0-1 범위)
    if risk_score > 1.0:
        risk_score = risk_score / 100.0
    risk_score = max(0.0, min(1.0, risk_score))

    # 피싱 여부 판정
    is_phishing = _determine_phishing(parsed, risk_score, classification)

    return LLMParseResult(
        risk_score=risk_score,
        reasoning=reasoning,
        key_evidence=key_evidence,
        classification=classification,
        is_phishing=is_phishing,
        raw_data=parsed,
        parse_success=True,
    )


def _extract_risk_score(data: Dict[str, Any]) -> float:
    """딕셔너리에서 위험 점수를 추출합니다."""
    # 우선순위: comprehensive_risk_score > risk_score > score
    for key in ("comprehensive_risk_score", "risk_score", "score"):
        if key in data:
            try:
                return float(data[key])
            except (ValueError, TypeError):
                continue
    return 0.0


def _extract_reasoning(data: Dict[str, Any]) -> str:
    """딕셔너리에서 판단 근거를 추출합니다."""
    for key in ("reasoning", "reason", "explanation", "rationale"):
        if key in data and data[key]:
            return str(data[key]).strip()
    return ""


def _extract_key_evidence(data: Dict[str, Any]) -> List[str]:
    """딕셔너리에서 핵심 증거 목록을 추출합니다."""
    for key in ("key_evidence", "evidence", "key_points"):
        if key in data:
            value = data[key]
            if isinstance(value, list):
                return [str(item) for item in value if item]
            elif isinstance(value, str):
                return [value] if value else []
    return []


def _extract_classification(data: Dict[str, Any]) -> str:
    """딕셔너리에서 분류 결과를 추출합니다."""
    for key in ("classification", "category", "label", "type"):
        if key in data and data[key]:
            return str(data[key]).strip()
    return "unknown"


def _determine_phishing(
    data: Dict[str, Any],
    risk_score: float,
    classification: str,
) -> bool:
    """보이스피싱 여부를 판정합니다."""
    # 명시적 phishing 필드가 있으면 우선
    if "phishing" in data:
        return bool(data["phishing"])

    # is_above_threshold 필드
    if "is_above_threshold" in data:
        return bool(data["is_above_threshold"])

    # 분류 문자열에 "보이스피싱" 포함
    if "보이스피싱" in classification:
        return True

    # 점수 기반 판정 (임계값 0.5)
    return risk_score >= 0.5


def format_analysis_result(
    plm_score: float,
    llm_score: float,
    reasoning: str = "",
    key_evidence: Optional[List[str]] = None,
    plm_weight: float = 0.5,
    llm_weight: float = 0.5,
) -> Dict[str, Any]:
    """PLM과 LLM 점수를 앙상블하여 최종 분석 결과를 포맷팅합니다.

    Args:
        plm_score: PLM 모델의 피싱 확률 (0.0 ~ 1.0)
        llm_score: LLM 모델의 위험 점수 (0.0 ~ 1.0)
        reasoning: LLM의 판단 근거
        key_evidence: 핵심 증거 목록
        plm_weight: PLM 점수 가중치
        llm_weight: LLM 점수 가중치

    Returns:
        표준화된 분석 결과 딕셔너리

    Note:
        키 이름은 RC-009 오타 수정 후 `LLM_risk_score`를 사용합니다.
        (기존 `LLM_risk_socre` 오타 폐기)
    """
    # 점수 정규화
    plm_score = max(0.0, min(1.0, float(plm_score)))
    llm_score = max(0.0, min(1.0, float(llm_score)))

    # 가중치 정규화
    total_weight = plm_weight + llm_weight
    if total_weight > 0:
        plm_weight = plm_weight / total_weight
        llm_weight = llm_weight / total_weight
    else:
        plm_weight = llm_weight = 0.5

    # 앙상블 점수 계산
    comprehensive_score = (plm_score * plm_weight) + (llm_score * llm_weight)

    return {
        "PLM_risk_score": plm_score,
        "LLM_risk_score": llm_score,  # RC-009: 오타 수정됨
        "comprehensive_risk_score": comprehensive_score,
        "reasoning": reasoning,
        "key_evidence": key_evidence or [],
        "is_phishing": comprehensive_score >= 0.5,
    }


def html_escape(text: str) -> str:
    """HTML 특수문자를 이스케이프합니다.

    Args:
        text: 원본 텍스트

    Returns:
        이스케이프된 텍스트
    """
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
        .replace("\n", "<br/>")
    )


def truncate_text(text: str, max_length: int = 4000, suffix: str = "...") -> str:
    """텍스트를 지정된 길이로 잘라냅니다.

    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 잘린 경우 붙일 접미사

    Returns:
        잘린 텍스트
    """
    if not text or len(text) <= max_length:
        return text or ""

    return text[: max_length - len(suffix)] + suffix


def simple_dedup(old_text: str, new_text: str, overlap_chars: int = 200) -> str:
    """두 텍스트를 중복 제거하며 병합합니다.

    실시간 STT에서 청크 간 중복된 부분을 제거할 때 사용합니다.

    Args:
        old_text: 기존 텍스트
        new_text: 새로운 텍스트
        overlap_chars: 중복 검사할 문자 수

    Returns:
        병합된 텍스트
    """
    if not old_text:
        return new_text
    if not new_text:
        return old_text

    pivot = old_text[-overlap_chars:] if len(old_text) > overlap_chars else old_text
    if pivot and new_text.startswith(pivot):
        return old_text + new_text[len(pivot) :]

    return (old_text + " " + new_text).strip()


# =============================================================================
# 임시 파일/리소스 관리
# =============================================================================


@contextmanager
def temp_audio_file(
    suffix: str = ".wav",
    prefix: str = "audio_",
    cleanup: bool = True,
) -> Generator[Path, None, None]:
    """임시 오디오 파일을 안전하게 관리하는 컨텍스트 매니저.

    블록 종료 시 자동으로 파일을 삭제합니다 (cleanup=True인 경우).
    이 함수가 없을 경우, 임시 파일이 쌓여 디스크 공간을 소비하거나
    예외 발생 시 파일이 삭제되지 않는 리소스 누수가 발생합니다.

    Args:
        suffix: 파일 확장자 (기본 .wav)
        prefix: 파일명 접두사
        cleanup: 종료 시 파일 삭제 여부

    Yields:
        Path: 임시 파일 경로

    Example:
        >>> with temp_audio_file(suffix=".wav") as audio_path:
        ...     # audio_path에 오디오 저장
        ...     save_audio(audio_path)
        ...     # 처리
        ...     result = process_audio(audio_path)
        >>> # 블록 종료 시 파일 자동 삭제

    Note:
        Windows에서는 열려 있는 파일 삭제 시 PermissionError가 발생할 수 있습니다.
        이 경우 경고 로그만 남기고 진행합니다.
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # 파일 디스크립터 즉시 닫기

    temp_path_obj = Path(temp_path)

    try:
        yield temp_path_obj
    finally:
        if cleanup and temp_path_obj.exists():
            try:
                temp_path_obj.unlink()
                logger.debug(f"임시 파일 삭제됨: {temp_path_obj}")
            except PermissionError:
                # Windows에서 파일이 아직 사용 중일 수 있음
                logger.warning(f"임시 파일 삭제 실패 (사용 중): {temp_path_obj}")
            except OSError as e:
                logger.warning(f"임시 파일 삭제 실패: {temp_path_obj} - {e}")


@contextmanager
def temp_directory(
    prefix: str = "vpp_",
    cleanup: bool = True,
) -> Generator[Path, None, None]:
    """임시 디렉토리를 안전하게 관리하는 컨텍스트 매니저.

    Args:
        prefix: 디렉토리명 접두사
        cleanup: 종료 시 디렉토리 삭제 여부

    Yields:
        Path: 임시 디렉토리 경로

    Example:
        >>> with temp_directory() as tmp_dir:
        ...     file1 = tmp_dir / "data.json"
        ...     file1.write_text('{"key": "value"}')
        >>> # 블록 종료 시 디렉토리와 모든 내용 삭제
    """
    import shutil

    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))

    try:
        yield temp_dir
    finally:
        if cleanup and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"임시 디렉토리 삭제됨: {temp_dir}")
            except PermissionError:
                logger.warning(f"임시 디렉토리 삭제 실패 (사용 중): {temp_dir}")
            except OSError as e:
                logger.warning(f"임시 디렉토리 삭제 실패: {temp_dir} - {e}")


def cleanup_old_temp_files(
    pattern: str = "audio_*.wav",
    max_age_seconds: int = 3600,
    temp_dir: Optional[Path] = None,
) -> int:
    """오래된 임시 파일들을 정리합니다.

    정기적으로 호출하여 누적된 임시 파일을 정리합니다.

    Args:
        pattern: 삭제할 파일 glob 패턴
        max_age_seconds: 이 시간(초)보다 오래된 파일만 삭제
        temp_dir: 검사할 디렉토리 (기본: 시스템 임시 디렉토리)

    Returns:
        삭제된 파일 수

    Example:
        >>> # 1시간 이상 된 임시 오디오 파일 정리
        >>> deleted = cleanup_old_temp_files("audio_*.wav", max_age_seconds=3600)
        >>> print(f"{deleted}개 파일 삭제됨")
    """
    import time

    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    current_time = time.time()
    deleted_count = 0

    for file_path in temp_dir.glob(pattern):
        try:
            if file_path.is_file():
                age = current_time - file_path.stat().st_mtime
                if age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"오래된 임시 파일 삭제: {file_path}")
        except OSError:
            continue

    if deleted_count > 0:
        logger.info(f"임시 파일 정리 완료: {deleted_count}개 삭제됨")

    return deleted_count


if __name__ == "__main__":
    # 테스트
    test_responses = [
        '{"risk_score": 0.85, "reasoning": "계좌이체 요청", "key_evidence": ["긴급", "비밀"]}',
        '{"classification": "보이스피싱", "score": 75, "reason": "검찰 사칭"}',
        "분석 결과: 위험 점수 0.6입니다.",
        '일반 텍스트와 {"phishing": true, "score": 0.9} JSON 혼합',
    ]

    print("=== JSON 파싱 테스트 ===")
    for resp in test_responses:
        result = parse_llm_json_response(resp)
        print(f"Input: {resp[:50]}...")
        print(f"  Score: {result.risk_score}, Phishing: {result.is_phishing}")
        print(f"  Parse Success: {result.parse_success}")
        print()

"""
커스텀 예외 클래스 모듈 (Custom Exceptions)

애플리케이션 전반에서 사용되는 구체적인 예외 클래스들을 정의합니다.

해결 이슈:
- RC-006: 광범위한 except Exception 사용 → 구체적인 예외 타입으로 개선

사용 예시:
    from src.core.exceptions import STTError, LLMError

    try:
        result = stt_engine.transcribe(audio)
    except STTError as e:
        logger.error(f"STT 실패: {e}")
"""

from __future__ import annotations


# =============================================================================
# 기본 예외 클래스
# =============================================================================


class VoicePhishingProtectorError(Exception):
    """애플리케이션 기본 예외 클래스

    모든 커스텀 예외의 부모 클래스입니다.
    이 클래스를 상속받아 구체적인 예외를 정의합니다.

    Attributes:
        message: 에러 메시지
        details: 추가 상세 정보 (선택)
    """

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# STT 관련 예외
# =============================================================================


class STTError(VoicePhishingProtectorError):
    """STT(Speech-to-Text) 처리 중 발생하는 예외

    음성 파일 변환, 모델 로드, 전사 실패 등의 상황에서 발생합니다.
    """

    pass


class AudioLoadError(STTError):
    """오디오 파일 로드 실패

    파일이 존재하지 않거나 지원하지 않는 형식일 때 발생합니다.
    """

    pass


class TranscriptionError(STTError):
    """음성 전사 실패

    Whisper 또는 Google STT 처리 중 오류가 발생했을 때 발생합니다.
    """

    pass


class ModelLoadError(STTError):
    """STT 모델 로드 실패

    faster-whisper 또는 다른 STT 모델을 불러올 수 없을 때 발생합니다.
    """

    pass


# =============================================================================
# LLM 관련 예외
# =============================================================================


class LLMError(VoicePhishingProtectorError):
    """LLM 처리 중 발생하는 예외

    Ollama, OpenAI 등 LLM API 호출 실패 시 발생합니다.
    """

    pass


class LLMConnectionError(LLMError):
    """LLM 서비스 연결 실패

    Ollama 서버가 실행되지 않았거나 네트워크 문제 시 발생합니다.
    """

    pass


class LLMResponseError(LLMError):
    """LLM 응답 처리 실패

    응답 파싱 실패, 예상치 못한 형식 등의 경우 발생합니다.
    """

    pass


class LLMTimeoutError(LLMError):
    """LLM 응답 타임아웃

    설정된 시간 내에 응답을 받지 못했을 때 발생합니다.
    """

    pass


# =============================================================================
# 얼굴 인식 관련 예외
# =============================================================================


class FaceRecognitionError(VoicePhishingProtectorError):
    """얼굴 인식 처리 중 발생하는 예외

    얼굴 탐지, 임베딩 추출, 매칭 등에서 발생합니다.
    """

    pass


class FaceDetectionError(FaceRecognitionError):
    """얼굴 탐지 실패

    YOLO 모델로 얼굴을 찾을 수 없거나 탐지 실패 시 발생합니다.
    """

    pass


class FaceEncodingError(FaceRecognitionError):
    """얼굴 임베딩 추출 실패

    ArcFace 모델로 임베딩을 생성할 수 없을 때 발생합니다.
    """

    pass


class FaceRegistrationError(FaceRecognitionError):
    """얼굴 등록 실패

    이미 등록된 이름이거나 저장 실패 시 발생합니다.
    """

    pass


class FaceNotFoundError(FaceRecognitionError):
    """등록된 얼굴을 찾을 수 없음

    삭제 또는 조회 시 해당 이름이 없을 때 발생합니다.
    """

    pass


# =============================================================================
# 이미지 처리 관련 예외
# =============================================================================


class ImageProcessingError(VoicePhishingProtectorError):
    """이미지 처리 중 발생하는 예외"""

    pass


class ImageDecodeError(ImageProcessingError):
    """이미지 디코딩 실패

    Base64 디코딩 또는 OpenCV imdecode 실패 시 발생합니다.
    """

    pass


class ImageFormatError(ImageProcessingError):
    """지원하지 않는 이미지 형식

    허용되지 않은 확장자나 손상된 파일일 때 발생합니다.
    """

    pass


# =============================================================================
# 설정/리소스 관련 예외
# =============================================================================


class ConfigurationError(VoicePhishingProtectorError):
    """설정 관련 예외

    필수 환경 변수 누락, 잘못된 설정값 등에서 발생합니다.
    """

    pass


class ResourceNotFoundError(VoicePhishingProtectorError):
    """리소스를 찾을 수 없음

    모델 파일, 데이터 파일 등이 존재하지 않을 때 발생합니다.
    """

    pass


class ResourceBusyError(VoicePhishingProtectorError):
    """리소스가 사용 중

    모델이 이미 로드 중이거나 파일이 잠겨있을 때 발생합니다.
    """

    pass


# =============================================================================
# API 관련 예외
# =============================================================================


class APIError(VoicePhishingProtectorError):
    """API 처리 중 발생하는 예외"""

    pass


class InvalidInputError(APIError):
    """잘못된 입력값

    필수 파라미터 누락, 형식 오류 등에서 발생합니다.
    """

    pass


class RunNotFoundError(APIError):
    """분석 작업을 찾을 수 없음

    존재하지 않는 run_id로 조회 시 발생합니다.
    """

    pass


# =============================================================================
# 유틸리티 함수
# =============================================================================


def handle_exception(
    exc: Exception,
    default_message: str = "처리 중 오류가 발생했습니다.",
) -> VoicePhishingProtectorError:
    """일반 예외를 커스텀 예외로 변환합니다.

    이미 커스텀 예외인 경우 그대로 반환하고,
    일반 Exception인 경우 VoicePhishingProtectorError로 래핑합니다.

    Args:
        exc: 원본 예외
        default_message: 기본 에러 메시지

    Returns:
        VoicePhishingProtectorError: 변환된 예외

    Example:
        try:
            risky_operation()
        except Exception as e:
            raise handle_exception(e, "작업 실패")
    """
    if isinstance(exc, VoicePhishingProtectorError):
        return exc

    return VoicePhishingProtectorError(
        message=default_message,
        details={"original_error": str(exc), "error_type": type(exc).__name__},
    )


if __name__ == "__main__":
    # 예외 테스트
    print("=== 커스텀 예외 테스트 ===")

    try:
        raise STTError("Whisper 모델 로드 실패", {"model": "large-v3"})
    except STTError as e:
        print(f"STTError: {e}")

    try:
        raise LLMConnectionError("Ollama 서버에 연결할 수 없습니다.")
    except LLMError as e:
        print(f"LLMError: {e}")

    try:
        raise FaceNotFoundError("홍길동", {"action": "delete"})
    except FaceRecognitionError as e:
        print(f"FaceRecognitionError: {e}")

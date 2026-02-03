"""
서비스 계층 단위 테스트 (Service Layer Unit Tests)

pytest를 사용한 서비스 모듈 테스트입니다.

실행 방법:
    pytest tests/test_services.py -v
    pytest tests/test_services.py -v -k "test_llm"
    pytest tests/test_services.py -v --cov=src

참고:
    - 외부 의존성(Ollama, 모델 파일 등)은 mock으로 대체합니다.
    - 실제 API 테스트는 tests/test_api.py에서 수행합니다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """테스트용 설정 객체"""
    from src.core.config import Settings

    return Settings(
        api_host="127.0.0.1",
        api_port=8000,
        ollama_url="http://localhost:11434",
        stt_model="tiny",
        similarity_threshold=0.70,
    )


@pytest.fixture
def sample_text():
    """테스트용 샘플 텍스트"""
    return "안녕하세요. 저는 검찰 수사관입니다. 계좌 확인이 필요합니다."


@pytest.fixture
def sample_normal_text():
    """일반 대화 샘플 텍스트"""
    return "오늘 날씨가 좋네요. 점심은 뭐 드실 거예요?"


# =============================================================================
# Utils 테스트
# =============================================================================


class TestUtils:
    """공통 유틸리티 테스트"""

    def test_extract_json_from_text_valid(self):
        """유효한 JSON 추출 테스트"""
        from src.core.utils import extract_json_from_text

        text = '분석 결과입니다. {"risk_score": 0.85, "reasoning": "의심스러운 패턴"}'
        result = extract_json_from_text(text)

        assert result is not None
        assert result["risk_score"] == 0.85
        assert "reasoning" in result

    def test_extract_json_from_text_invalid(self):
        """JSON이 없는 텍스트 테스트"""
        from src.core.utils import extract_json_from_text

        text = "이것은 일반 텍스트입니다."
        result = extract_json_from_text(text)

        assert result is None

    def test_extract_score_from_text(self):
        """텍스트에서 점수 추출 테스트"""
        from src.core.utils import extract_score_from_text

        # 0-1 범위 소수
        assert extract_score_from_text("위험 점수: 0.85") == 0.85

        # 0-100 범위 정수 (정규화)
        score = extract_score_from_text("위험도는 75점입니다")
        assert score == 0.75

    def test_parse_llm_json_response(self):
        """LLM 응답 파싱 테스트"""
        from src.core.utils import parse_llm_json_response

        response = '{"risk_score": 0.9, "reasoning": "계좌이체 요청", "key_evidence": ["긴급", "비밀"]}'
        result = parse_llm_json_response(response)

        assert result.parse_success is True
        assert result.risk_score == 0.9
        assert result.is_phishing is True
        assert len(result.key_evidence) == 2

    def test_parse_llm_json_response_fallback(self):
        """파싱 실패 시 폴백 테스트"""
        from src.core.utils import parse_llm_json_response

        response = "이것은 JSON이 아닌 응답입니다. 위험도: 0.6"
        result = parse_llm_json_response(response)

        assert result.parse_success is False
        assert result.risk_score == 0.6  # 폴백 점수 추출

    def test_format_analysis_result(self):
        """분석 결과 포맷팅 테스트"""
        from src.core.utils import format_analysis_result

        result = format_analysis_result(
            plm_score=0.8,
            llm_score=0.9,
            reasoning="의심스러운 패턴 감지",
            key_evidence=["계좌이체", "긴급"],
        )

        assert "PLM_risk_score" in result
        assert "LLM_risk_score" in result  # 오타 수정 확인
        assert "comprehensive_risk_score" in result
        assert result["is_phishing"] is True

    def test_html_escape(self):
        """HTML 이스케이프 테스트"""
        from src.core.utils import html_escape

        text = '<script>alert("XSS")</script>'
        escaped = html_escape(text)

        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "<script>" not in escaped

    def test_temp_audio_file_context_manager(self):
        """임시 오디오 파일 컨텍스트 매니저 테스트"""
        from src.core.utils import temp_audio_file

        with temp_audio_file(suffix=".wav", cleanup=True) as temp_path:
            assert temp_path.suffix == ".wav"
            # 파일에 데이터 쓰기
            temp_path.write_bytes(b"test audio data")
            assert temp_path.exists()

        # 컨텍스트 종료 후 파일 삭제 확인
        assert not temp_path.exists()


# =============================================================================
# 예외 클래스 테스트
# =============================================================================


class TestExceptions:
    """커스텀 예외 클래스 테스트"""

    def test_voice_phishing_protector_error(self):
        """기본 예외 클래스 테스트"""
        from src.core.exceptions import VoicePhishingProtectorError

        error = VoicePhishingProtectorError(
            message="테스트 에러",
            details={"code": 123},
        )

        assert error.message == "테스트 에러"
        assert error.details["code"] == 123
        assert "테스트 에러" in str(error)

    def test_stt_error_hierarchy(self):
        """STT 예외 계층 테스트"""
        from src.core.exceptions import (
            AudioLoadError,
            ModelLoadError,
            STTError,
            TranscriptionError,
        )

        assert issubclass(AudioLoadError, STTError)
        assert issubclass(TranscriptionError, STTError)
        assert issubclass(ModelLoadError, STTError)

    def test_llm_error_hierarchy(self):
        """LLM 예외 계층 테스트"""
        from src.core.exceptions import (
            LLMConnectionError,
            LLMError,
            LLMResponseError,
            LLMTimeoutError,
        )

        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMResponseError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)


# =============================================================================
# LLM 서비스 테스트
# =============================================================================


class TestLLMService:
    """LLM 서비스 테스트"""

    def test_get_llm_client_ollama(self):
        """Ollama 클라이언트 생성 테스트"""
        from src.services.llm_service import get_llm_client

        client = get_llm_client(provider="ollama")
        assert client is not None
        assert hasattr(client, "chat")

    def test_get_llm_client_openai(self):
        """OpenAI 클라이언트 생성 테스트"""
        from src.services.llm_service import get_llm_client

        client = get_llm_client(provider="openai")
        assert client is not None
        assert hasattr(client, "chat")

    def test_get_llm_client_invalid_provider(self):
        """잘못된 provider 테스트"""
        from src.services.llm_service import get_llm_client

        with pytest.raises(ValueError):
            get_llm_client(provider="invalid_provider")

    @patch("requests.post")
    def test_ollama_chat_success(self, mock_post):
        """Ollama 채팅 성공 테스트"""
        from src.services.llm_service import OllamaClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "테스트 응답입니다."}
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.chat("안녕하세요")

        assert result == "테스트 응답입니다."
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_ollama_chat_connection_error(self, mock_post):
        """Ollama 연결 실패 테스트"""
        import requests

        from src.core.exceptions import LLMConnectionError
        from src.services.llm_service import OllamaClient

        mock_post.side_effect = requests.exceptions.ConnectionError()

        client = OllamaClient()
        with pytest.raises(LLMConnectionError):
            client.chat("안녕하세요")


# =============================================================================
# 로깅 테스트
# =============================================================================


class TestLogging:
    """로깅 모듈 테스트"""

    def test_setup_logging(self):
        """로깅 설정 테스트"""
        from src.core.logging import setup_logging

        # 예외 없이 실행되어야 함
        setup_logging(level="DEBUG", json_format=False)

    def test_get_logger(self):
        """로거 가져오기 테스트"""
        from src.core.logging import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_colored_formatter(self):
        """컬러 포매터 테스트"""
        import logging

        from src.core.logging import ColoredFormatter

        formatter = ColoredFormatter(
            fmt="%(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="테스트 메시지",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "테스트 메시지" in formatted

    def test_json_formatter(self):
        """JSON 포매터 테스트"""
        import json
        import logging

        from src.core.logging import JSONFormatter

        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="테스트 메시지",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "테스트 메시지"
        assert "timestamp" in parsed


# =============================================================================
# 설정 테스트
# =============================================================================


class TestConfig:
    """설정 모듈 테스트"""

    def test_settings_defaults(self):
        """기본 설정값 테스트"""
        from src.core.config import Settings

        settings = Settings()

        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.similarity_threshold == 0.70

    def test_settings_env_override(self, monkeypatch):
        """환경 변수 오버라이드 테스트"""
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.85")

        from src.core.config import Settings

        settings = Settings()

        assert settings.api_port == 9000
        assert settings.similarity_threshold == 0.85

    def test_project_root_path(self):
        """프로젝트 루트 경로 테스트"""
        from src.core.config import settings

        assert settings.project_root.exists()

    def test_models_dir_path(self):
        """모델 디렉토리 경로 테스트"""
        from src.core.config import settings

        models_dir = settings.models_dir
        assert "trained_models" in str(models_dir)


# =============================================================================
# API 엔드포인트 테스트 (간단한 통합 테스트)
# =============================================================================


class TestAPIEndpoints:
    """API 엔드포인트 기본 테스트"""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트"""
        from fastapi.testclient import TestClient

        from main import app

        return TestClient(app)

    def test_health_check(self, client):
        """헬스 체크 엔드포인트 테스트"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint(self, client):
        """헬스 엔드포인트 테스트"""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_face_list_empty(self, client):
        """빈 얼굴 목록 조회 테스트"""
        response = client.get("/face_list")

        # 서비스가 초기화되지 않아도 빈 목록 반환
        assert response.status_code == 200
        data = response.json()
        assert "faces" in data
        assert "count" in data

    def test_status_not_found(self, client):
        """존재하지 않는 run_id 조회 테스트"""
        response = client.get("/status/nonexistent-id")

        assert response.status_code == 404


# =============================================================================
# Markers
# =============================================================================


# 느린 테스트 (실제 모델 로드 필요)
slow = pytest.mark.slow

# 외부 서비스 필요 (Ollama 등)
external = pytest.mark.external

# 통합 테스트
integration = pytest.mark.integration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

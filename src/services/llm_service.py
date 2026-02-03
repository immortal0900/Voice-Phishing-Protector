"""
LLM 서비스 모듈 (LLM Service)

Strategy 패턴을 적용하여 다양한 LLM 백엔드(Ollama, OpenAI)를 지원합니다.

해결 이슈:
- SC-003: LLM 클라이언트 확장성 부족 → Strategy 패턴 적용
- RC-012: LLM 호출 코드 중복 → 단일 서비스로 통합
- OP-003: 에러 복구 전략 부재 → tenacity로 재시도 로직 추가

사용 예시:
    from src.services.llm_service import get_llm_client

    client = get_llm_client("ollama")  # 또는 "openai"
    response = client.chat(system_prompt, user_prompt)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """LLM 클라이언트 관련 예외"""

    pass


class LLMConnectionError(LLMClientError):
    """LLM 서버 연결 오류"""

    pass


class LLMResponseError(LLMClientError):
    """LLM 응답 파싱 오류"""

    pass


class LLMClient(ABC):
    """LLM 클라이언트 추상 베이스 클래스

    Strategy 패턴의 Strategy 인터페이스 역할을 합니다.
    새로운 LLM 백엔드를 추가하려면 이 클래스를 상속받아 구현하세요.

    Attributes:
        model: 사용할 모델명
        temperature: 생성 온도 (낮을수록 결정적)
        timeout: 요청 타임아웃 (초)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        timeout: int = 120,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    @abstractmethod
    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """LLM과 대화하여 응답을 반환합니다.

        Args:
            user_prompt: 사용자 프롬프트 (필수)
            system_prompt: 시스템 프롬프트 (선택)
            max_tokens: 최대 생성 토큰 수

        Returns:
            LLM 응답 텍스트

        Raises:
            LLMConnectionError: 서버 연결 실패
            LLMResponseError: 응답 파싱 실패
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """LLM 서버가 사용 가능한지 확인합니다.

        Returns:
            서버 가용 여부
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


class OllamaClient(LLMClient):
    """Ollama 기반 LLM 클라이언트

    로컬에서 실행되는 Ollama 서버와 통신합니다.
    Gemma, Llama 등 다양한 오픈소스 모델을 지원합니다.

    Attributes:
        base_url: Ollama 서버 URL
        session: HTTP 세션 (연결 재사용)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: Optional[int] = None,
    ):
        super().__init__(
            model=model or settings.ollama_model,
            temperature=temperature,
            timeout=timeout or settings.ollama_timeout,
        )
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry_min_wait,
            max=settings.retry_max_wait,
        ),
        reraise=True,
    )
    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """Ollama API를 통해 채팅 응답을 받습니다.

        /api/chat 엔드포인트를 사용하여 메시지 기반 대화를 수행합니다.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Ollama 응답 구조에서 메시지 추출
            content = data.get("message", {}).get("content", "")
            if not content and isinstance(data.get("messages"), list):
                # 대체 응답 구조
                content = data["messages"][-1].get("content", "")

            return content.strip()

        except requests.Timeout as e:
            logger.error(f"Ollama 타임아웃: {e}")
            raise LLMConnectionError(f"Ollama 서버 타임아웃: {self.base_url}") from e
        except requests.RequestException as e:
            logger.error(f"Ollama 연결 오류: {e}")
            raise LLMConnectionError(f"Ollama 서버 연결 실패: {e}") from e
        except (KeyError, ValueError) as e:
            logger.error(f"Ollama 응답 파싱 오류: {e}")
            raise LLMResponseError(f"응답 파싱 실패: {e}") from e

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Ollama /api/generate 엔드포인트를 사용합니다.

        단일 프롬프트 생성에 적합합니다.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.RequestException as e:
            logger.error(f"Ollama generate 오류: {e}")
            raise LLMConnectionError(f"Ollama 서버 연결 실패: {e}") from e

    def is_available(self) -> bool:
        """Ollama 서버 가용성을 확인합니다."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False


class OpenAIClient(LLMClient):
    """OpenAI API 기반 LLM 클라이언트

    OpenAI의 GPT 모델들을 지원합니다.
    OPENAI_API_KEY 환경변수가 설정되어야 합니다.

    Attributes:
        api_key: OpenAI API 키
        client: OpenAI 클라이언트 인스턴스
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 60,
    ):
        super().__init__(
            model=model or settings.openai_model,
            temperature=temperature,
            timeout=timeout,
        )
        self.api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise LLMClientError(
                "OpenAI API 키가 설정되지 않았습니다. "
                "OPENAI_API_KEY 환경변수를 설정하거나 .env 파일에 추가하세요."
            )

        # 지연 임포트 (OpenAI가 설치되지 않았을 수 있음)
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise LLMClientError("openai 패키지가 설치되지 않았습니다: pip install openai")

    @retry(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry_min_wait,
            max=settings.retry_max_wait,
        ),
        reraise=True,
    )
    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """OpenAI Chat Completions API를 통해 응답을 받습니다."""
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return (content or "").strip()

        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            raise LLMConnectionError(f"OpenAI API 호출 실패: {e}") from e

    def is_available(self) -> bool:
        """OpenAI API 키가 유효한지 확인합니다."""
        if not self.api_key:
            return False
        try:
            # 간단한 모델 목록 조회로 키 유효성 확인
            self.client.models.list()
            return True
        except Exception:
            return False


class MockLLMClient(LLMClient):
    """테스트용 Mock LLM 클라이언트

    테스트 환경에서 실제 LLM 호출 없이 사용할 수 있습니다.
    """

    def __init__(
        self,
        model: str = "mock-model",
        default_response: str = '{"risk_score": 0.5, "reasoning": "테스트 응답"}',
    ):
        super().__init__(model=model)
        self.default_response = default_response
        self.call_history: List[Dict[str, Any]] = []

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """Mock 응답을 반환합니다."""
        self.call_history.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.default_response

    def is_available(self) -> bool:
        return True


def get_llm_client(
    provider: str = "ollama",
    **kwargs: Any,
) -> LLMClient:
    """팩토리 함수: 설정에 따라 적절한 LLM 클라이언트를 반환합니다.

    Strategy 패턴의 Context 역할을 합니다.

    Args:
        provider: LLM 제공자 ("ollama", "openai", "mock")
        **kwargs: 클라이언트 생성자에 전달할 추가 인자

    Returns:
        LLMClient 구현체

    Raises:
        ValueError: 알 수 없는 provider

    Example:
        >>> client = get_llm_client("ollama", model="gemma2:9b")
        >>> response = client.chat("안녕하세요")
    """
    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(
            f"알 수 없는 LLM provider: {provider}. "
            "지원 provider: ollama, openai, mock"
        )


def get_llm_client_from_spec(model_spec: str, **kwargs: Any) -> LLMClient:
    """모델 스펙 문자열에서 LLM 클라이언트를 생성합니다.

    레거시 호환성을 위한 함수입니다.

    Args:
        model_spec: "provider:model" 형식 (예: "ollama:gemma2:9b", "openai:gpt-4o-mini")
        **kwargs: 추가 설정

    Returns:
        LLMClient 구현체

    Example:
        >>> client = get_llm_client_from_spec("ollama:gemma2:9b")
        >>> client = get_llm_client_from_spec("openai:gpt-4o-mini")
    """
    parts = model_spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"잘못된 model_spec 형식: {model_spec}. "
            "'provider:model' 형식이어야 합니다."
        )

    provider, model = parts
    return get_llm_client(provider, model=model, **kwargs)


# 편의를 위한 싱글톤 인스턴스 (선택적 사용)
_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """기본 LLM 클라이언트를 반환합니다.

    싱글톤 패턴으로 한 번만 생성됩니다.
    """
    global _default_client
    if _default_client is None:
        _default_client = get_llm_client("ollama")
    return _default_client


if __name__ == "__main__":
    # 간단한 테스트
    print("=== LLM 서비스 테스트 ===")

    # Mock 클라이언트 테스트
    mock = get_llm_client("mock")
    response = mock.chat("테스트 프롬프트")
    print(f"Mock 응답: {response}")

    # Ollama 가용성 확인
    try:
        ollama = get_llm_client("ollama")
        if ollama.is_available():
            print("Ollama 서버: 사용 가능")
            response = ollama.chat("안녕하세요. 간단히 인사해주세요.", max_tokens=50)
            print(f"Ollama 응답: {response[:100]}...")
        else:
            print("Ollama 서버: 사용 불가")
    except Exception as e:
        print(f"Ollama 테스트 실패: {e}")

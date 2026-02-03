"""
설정 관리 모듈 (Configuration Management)

환경변수와 기본값을 통합 관리합니다.
pydantic-settings를 사용하여 타입 안전성과 검증을 제공합니다.

해결 이슈:
- RC-003: 경로 하드코딩 → pathlib.Path 사용
- RC-004: 하드코딩된 URL → 환경변수로 분리
- RD-001~003: 매직 넘버/문자열 → 설정 상수화

사용 예시:
    from src.core.config import settings

    print(settings.api_port)  # 8000
    print(settings.yolo_model_path)  # Path 객체
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    애플리케이션 설정

    모든 설정은 환경변수로 오버라이드 가능합니다.
    환경변수명은 대문자로 변환됩니다. (예: api_port → API_PORT)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # API 설정
    # =========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI 서버 바인딩 호스트"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="FastAPI 서버 포트"
    )
    backend_url: str = Field(
        default="http://127.0.0.1:8000",
        description="프론트엔드에서 접근할 백엔드 URL"
    )

    # =========================================================================
    # Ollama 설정 (로컬 LLM)
    # =========================================================================
    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama 서버 URL"
    )
    ollama_model: str = Field(
        default="gemma2:9b",
        description="Ollama에서 사용할 모델명"
    )
    ollama_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM 생성 온도 (낮을수록 결정적)"
    )
    ollama_timeout: int = Field(
        default=120,
        ge=10,
        description="Ollama 요청 타임아웃 (초)"
    )

    # =========================================================================
    # OpenAI 설정 (선택적)
    # =========================================================================
    openai_api_key: str = Field(
        default="",
        description="OpenAI API 키 (선택적)"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI 모델명"
    )

    # =========================================================================
    # STT 설정 (Speech-to-Text)
    # =========================================================================
    stt_model: str = Field(
        default="small",
        description="Whisper 모델 크기 (tiny, base, small, medium, large)"
    )
    stt_device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="STT 모델 실행 디바이스"
    )
    stt_compute_type: str = Field(
        default="float16",
        description="STT 연산 타입 (float16, float32, int8)"
    )
    chunk_duration_sec: int = Field(
        default=5,
        ge=1,
        le=30,
        description="오디오 청크 길이 (초)"
    )
    analysis_interval_chunks: int = Field(
        default=6,
        ge=1,
        description="LLM 분석 트리거 간격 (청크 수, 6 = 30초마다)"
    )

    # =========================================================================
    # 얼굴 인식 설정
    # =========================================================================
    similarity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="얼굴 유사도 임계값 (높을수록 엄격)"
    )
    yolo_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="YOLO 탐지 신뢰도 임계값"
    )

    # =========================================================================
    # 경로 설정 (pathlib 사용으로 크로스플랫폼 호환)
    # =========================================================================
    # 프로젝트 루트는 이 파일 기준으로 3단계 상위
    @property
    def project_root(self) -> Path:
        """프로젝트 루트 디렉토리"""
        return Path(__file__).parent.parent.parent.resolve()

    # 모델 파일명 설정
    yolo_model_name: str = Field(
        default="yolov8l_100e.pt",
        description="YOLO 모델 파일명"
    )
    plm_model_file: str = Field(
        default="koelectra_base_v3_finetuned.safetensors",
        description="PLM 모델 파일명 (.safetensors)"
    )
    llm_model_file: str = Field(
        default="gemma_2b_finetuned.safetensors",
        description="LLM 모델 파일명 (.safetensors)"
    )
    whisper_model_file: str = Field(
        default="whisper_meium_finetuned.safetensors",
        description="Whisper STT 모델 파일명 (.safetensors)"
    )
    face_embeddings_file: str = Field(
        default="known_face_embeddings.pkl",
        description="얼굴 임베딩 캐시 파일명"
    )

    @property
    def models_dir(self) -> Path:
        """모델 저장 디렉토리"""
        return self.project_root / "models"

    @property
    def cache_dir(self) -> Path:
        """런타임 캐시 디렉토리"""
        return self.project_root / "cache"

    @property
    def yolo_model_path(self) -> Path:
        """YOLO 모델 전체 경로"""
        return self.models_dir / self.yolo_model_name

    @property
    def plm_model_path(self) -> Path:
        """PLM 모델 전체 경로"""
        return self.models_dir / self.plm_model_file

    @property
    def llm_model_path(self) -> Path:
        """LLM 모델 전체 경로"""
        return self.models_dir / self.llm_model_file

    @property
    def whisper_model_path(self) -> Path:
        """Whisper STT 모델 전체 경로"""
        return self.models_dir / self.whisper_model_file

    @property
    def face_embeddings_cache(self) -> Path:
        """얼굴 임베딩 캐시 전체 경로"""
        return self.cache_dir / self.face_embeddings_file

    @property
    def face_images_dir(self) -> Path:
        """얼굴 이미지 저장 디렉토리"""
        return self.project_root / "data" / "face_images"

    @property
    def data_dir(self) -> Path:
        """데이터 디렉토리"""
        return self.project_root / "data"

    # =========================================================================
    # 보이스피싱 탐지 설정
    # =========================================================================
    phishing_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="피싱 판정 임계값"
    )
    plm_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="PLM 점수 가중치 (앙상블)"
    )
    llm_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM 점수 가중치 (앙상블)"
    )

    # =========================================================================
    # 로깅 설정
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="로깅 레벨"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷"
    )
    log_file: str = Field(
        default="",
        description="로그 파일 경로 (빈 문자열이면 콘솔만 출력)"
    )

    # =========================================================================
    # 캐시/세션 설정
    # =========================================================================
    max_runs: int = Field(
        default=100,
        ge=10,
        description="최대 동시 분석 세션 수"
    )
    run_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="분석 세션 TTL (초)"
    )

    # =========================================================================
    # 재시도 설정
    # =========================================================================
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        description="API 호출 최대 재시도 횟수"
    )
    retry_min_wait: float = Field(
        default=2.0,
        ge=0.1,
        description="재시도 최소 대기 시간 (초)"
    )
    retry_max_wait: float = Field(
        default=10.0,
        ge=1.0,
        description="재시도 최대 대기 시간 (초)"
    )


# 싱글톤 인스턴스: 앱 전체에서 이 인스턴스를 import하여 사용
settings = Settings()


if __name__ == "__main__":
    # 설정값 확인용 스크립트
    print("=== Voice Phishing Protector Settings ===")
    print(f"Project Root: {settings.project_root}")
    print(f"API: {settings.api_host}:{settings.api_port}")
    print(f"Backend URL: {settings.backend_url}")
    print(f"Ollama: {settings.ollama_url} ({settings.ollama_model})")
    print(f"STT Model: {settings.stt_model} on {settings.stt_device}")
    print(f"YOLO Model Path: {settings.yolo_model_path}")
    print(f"PLM Model Path: {settings.plm_model_path}")
    print(f"Phishing Threshold: {settings.phishing_threshold}")
    print(f"Log Level: {settings.log_level}")

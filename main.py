"""
Voice Phishing Protector - FastAPI 애플리케이션 엔트리포인트

실시간 보이스피싱 탐지 및 얼굴 인식 서비스의 메인 진입점입니다.

해결 이슈:
- SC-001: main.py SRP 위반 → 라우터/서비스 분리로 641줄 → ~60줄 경량화
- SC-004: 직접 구현체 참조 → 서비스 모듈로 위임
- SC-005: 전역 상태 의존 → 라우터 모듈로 캡슐화

실행 방법:
    # 직접 실행
    python main.py

    # uvicorn 사용
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # uv 사용
    uv run python main.py
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# API 라우터 임포트
from src.api import face_recognition as face_router
from src.api import voice_analysis as voice_router

# 설정 로드 시도 (pydantic-settings 미설치 시 기본값 사용)
try:
    from src.core.config import settings

    API_HOST = settings.api_host
    API_PORT = settings.api_port
except ImportError:
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))


# =============================================================================
# FastAPI 애플리케이션 생성
# =============================================================================

app = FastAPI(
    title="Voice Phishing Protector API",
    description="실시간 보이스피싱 탐지 및 얼굴 인식 서비스",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 미들웨어 설정
# NOTE: 프로덕션에서는 allow_origins를 특정 도메인으로 제한하세요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 라우터 등록
# =============================================================================

# 음성 분석 API (/analyze, /status, /predict_text)
app.include_router(voice_router.router, prefix="", tags=["Voice Analysis"])

# 얼굴 인식 API (/face_recognize, /face_register, /face_list)
app.include_router(face_router.router, prefix="", tags=["Face Recognition"])


# =============================================================================
# 루트 엔드포인트
# =============================================================================


@app.get("/")
def health_check():
    """서버 상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "service": "Voice Phishing Protector",
        "version": "1.0.0",
        "message": "API가 정상적으로 실행 중입니다.",
    }


@app.get("/health")
def health():
    """헬스 체크 엔드포인트 (컨테이너/로드밸런서용)"""
    return {"status": "ok"}


# =============================================================================
# 메인 실행
# =============================================================================


def run_server():
    """서버를 실행합니다. CLI 스크립트 엔트리포인트로 사용됩니다."""
    import uvicorn

    print(f"[INFO] Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)


if __name__ == "__main__":
    import uvicorn

    print(f"[INFO] Starting uvicorn on {API_HOST}:{API_PORT}")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT)

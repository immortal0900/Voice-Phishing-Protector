"""
Pytest 설정 및 공통 Fixtures

이 파일은 모든 테스트 모듈에서 자동으로 로드됩니다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 환경 설정
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """테스트 환경 설정 (세션 단위)"""
    # 테스트용 환경 변수 설정
    os.environ.setdefault("API_HOST", "127.0.0.1")
    os.environ.setdefault("API_PORT", "8000")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

    yield

    # 정리 작업 (필요시)


# =============================================================================
# 공통 Fixtures
# =============================================================================


@pytest.fixture
def sample_audio_bytes():
    """테스트용 더미 오디오 바이트"""
    # WAV 헤더 + 무음 데이터
    import struct

    # 최소 WAV 파일 헤더 (44 bytes)
    sample_rate = 16000
    num_samples = 16000  # 1초
    bits_per_sample = 16
    num_channels = 1

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    # RIFF 헤더
    wav_data = b"RIFF"
    wav_data += struct.pack("<I", 36 + data_size)  # 파일 크기
    wav_data += b"WAVE"

    # fmt 서브청크
    wav_data += b"fmt "
    wav_data += struct.pack("<I", 16)  # 서브청크 크기
    wav_data += struct.pack("<H", 1)  # PCM 포맷
    wav_data += struct.pack("<H", num_channels)
    wav_data += struct.pack("<I", sample_rate)
    wav_data += struct.pack("<I", byte_rate)
    wav_data += struct.pack("<H", block_align)
    wav_data += struct.pack("<H", bits_per_sample)

    # data 서브청크
    wav_data += b"data"
    wav_data += struct.pack("<I", data_size)
    wav_data += b"\x00" * data_size  # 무음 데이터

    return wav_data


@pytest.fixture
def sample_image_bytes():
    """테스트용 더미 이미지 바이트 (1x1 검은색 PNG)"""
    # 최소 PNG 파일
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def temp_dir(tmp_path):
    """임시 디렉토리 (테스트 종료 후 자동 삭제)"""
    return tmp_path


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio_bytes):
    """임시 오디오 파일"""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(sample_audio_bytes)
    return audio_path


@pytest.fixture
def temp_image_file(tmp_path, sample_image_bytes):
    """임시 이미지 파일"""
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(sample_image_bytes)
    return image_path


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_ollama_response():
    """Ollama API 응답 모킹"""
    return {
        "model": "gemma3:4b",
        "response": '{"risk_score": 0.85, "reasoning": "검찰 사칭 의심", "key_evidence": ["계좌이체", "긴급"]}',
        "done": True,
    }


@pytest.fixture
def mock_stt_result():
    """STT 결과 모킹"""
    return "안녕하세요. 저는 검찰 수사관입니다. 계좌 확인이 필요합니다."


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config):
    """pytest 마커 등록"""
    config.addinivalue_line("markers", "slow: 느린 테스트 (실제 모델 로드)")
    config.addinivalue_line("markers", "external: 외부 서비스 필요 (Ollama 등)")
    config.addinivalue_line("markers", "integration: 통합 테스트")
    config.addinivalue_line("markers", "gpu: GPU 필요")


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """테스트 수집 후 처리"""
    # 마커가 없는 테스트에 기본 마커 추가
    for item in items:
        if "slow" not in item.keywords:
            item.add_marker(pytest.mark.quick)

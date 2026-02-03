"""
로깅 설정 모듈 (Logging Configuration)

애플리케이션 전반에 사용되는 구조화된 로깅 시스템을 제공합니다.

해결 이슈:
- OP-010: print() → logging 모듈 전환
- OP-011: 구조화된 로깅 포맷

사용 예시:
    from src.core.logging import setup_logging, get_logger

    # 앱 시작 시 한 번 호출
    setup_logging(level="DEBUG")

    # 각 모듈에서 로거 가져오기
    logger = get_logger(__name__)
    logger.info("서버가 시작되었습니다", extra={"port": 8000})
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 컬러 코드 (터미널 출력용)
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """터미널 출력용 컬러 포매터

    로그 레벨에 따라 다른 색상을 적용하여 가독성을 높입니다.
    """

    def format(self, record: logging.LogRecord) -> str:
        # 컬러 적용
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = (
                f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
            )

        # 모듈명 짧게 표시 (마지막 부분만)
        if "." in record.name:
            record.name = record.name.rsplit(".", 1)[-1]

        result = super().format(record)

        # 원래 레벨명 복원 (다음 로그를 위해)
        record.levelname = levelname
        return result


class JSONFormatter(logging.Formatter):
    """JSON 형식 포매터

    로그를 JSON 형식으로 출력하여 로그 수집 시스템(ELK 등)과 연동합니다.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_dict = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 예외 정보 추가
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        # extra 필드 추가
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            ):
                log_dict[key] = value

        return json.dumps(log_dict, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    log_dir: Optional[Path] = None,
) -> None:
    """애플리케이션 로깅을 설정합니다.

    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일명 (None이면 파일 로깅 비활성화)
        json_format: JSON 형식 사용 여부 (프로덕션용)
        log_dir: 로그 파일 저장 디렉토리

    Example:
        >>> # 개발 환경
        >>> setup_logging(level="DEBUG")

        >>> # 프로덕션 환경
        >>> setup_logging(
        ...     level="INFO",
        ...     log_file="app.log",
        ...     json_format=True,
        ... )
    """
    # 로그 레벨 변환
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
            datefmt="%H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 (선택적)
    if log_file:
        if log_dir is None:
            log_dir = Path.cwd() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)

        # 파일은 항상 JSON 형식 (검색/분석 용이)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    # 설정 완료 로그
    root_logger.info(
        f"로깅 설정 완료 (level={level}, json={json_format}, file={log_file})"
    )


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환합니다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 설정된 로거 인스턴스

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("처리 시작", extra={"run_id": "abc123"})
    """
    return logging.getLogger(name)


class LogContext:
    """로그 컨텍스트 관리자

    특정 코드 블록 내에서 추가 컨텍스트를 로그에 포함시킵니다.

    Example:
        >>> with LogContext(run_id="abc123", user="홍길동"):
        ...     logger.info("작업 시작")  # run_id, user 자동 포함
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self._old_factory = None

    def __enter__(self):
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **factory_kwargs):
            record = old_factory(*args, **factory_kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        self._old_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
        return False


# 편의 함수: 빠른 로깅 설정
def configure_for_development() -> None:
    """개발 환경용 로깅 설정 (컬러 출력, DEBUG 레벨)"""
    setup_logging(level="DEBUG", json_format=False)


def configure_for_production(log_file: str = "app.log") -> None:
    """프로덕션 환경용 로깅 설정 (JSON 형식, INFO 레벨)"""
    setup_logging(level="INFO", log_file=log_file, json_format=True)


if __name__ == "__main__":
    # 테스트
    print("=== 개발 환경 로깅 테스트 ===")
    configure_for_development()

    logger = get_logger("test")
    logger.debug("디버그 메시지")
    logger.info("정보 메시지")
    logger.warning("경고 메시지")
    logger.error("에러 메시지")
    logger.critical("치명적 오류 메시지")

    # 컨텍스트 테스트
    print("\n=== 컨텍스트 로깅 테스트 ===")
    with LogContext(run_id="test-123", user="홍길동"):
        logger.info("컨텍스트가 포함된 로그")

    # 예외 로깅 테스트
    print("\n=== 예외 로깅 테스트 ===")
    try:
        raise ValueError("테스트 예외")
    except ValueError:
        logger.exception("예외가 발생했습니다")

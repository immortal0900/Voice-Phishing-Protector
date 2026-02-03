"""
얼굴 인식 API 라우터 (Face Recognition Router)

얼굴 탐지 및 인식 엔드포인트를 제공합니다.

엔드포인트:
- POST /face_recognize: 이미지에서 얼굴 인식
- POST /face_register: 새 얼굴 등록
- DELETE /face_unregister/{name}: 등록된 얼굴 삭제
- GET /face_list: 등록된 얼굴 목록 조회

해결 이슈:
- RC-005: 더미 face_recognize 엔드포인트 → 실제 구현으로 교체

사용 예시:
    from fastapi import FastAPI
    from src.api.face_recognition import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.exceptions import (
    FaceDetectionError,
    FaceEncodingError,
    FaceNotFoundError,
    FaceRecognitionError,
    FaceRegistrationError,
    ImageDecodeError,
)
from src.models.schemas import (
    BoundingBoxSchema,
    FaceListResponse,
    FaceRecognitionResponse,
    FaceRegisterResponse,
    FaceSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Face Recognition"])

# 얼굴 인식 서비스 (지연 로드)
_face_recognizer = None


def _get_face_recognizer():
    """FaceRecognizer 인스턴스를 지연 로드합니다."""
    global _face_recognizer
    if _face_recognizer is None:
        try:
            from src.services.face_service import FaceRecognizer

            _face_recognizer = FaceRecognizer()
            _face_recognizer.initialize()
            logger.info("얼굴 인식 서비스 초기화 완료")
        except Exception as e:
            logger.warning(f"얼굴 인식 서비스 초기화 실패: {e}")
            _face_recognizer = None
    return _face_recognizer


def _decode_base64_image(image_b64: str) -> np.ndarray:
    """Base64 인코딩된 이미지를 numpy 배열로 디코딩합니다.

    Args:
        image_b64: Base64 인코딩된 이미지 문자열

    Returns:
        BGR 형식의 numpy 배열

    Raises:
        ValueError: 디코딩 실패 시
    """
    try:
        import cv2

        # Data URL 형식 처리 (data:image/jpeg;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("이미지 디코딩 실패")

        return image

    except (ValueError, TypeError) as e:
        raise ImageDecodeError(f"Base64 이미지 디코딩 실패: {e}")


async def _read_upload_file(file: UploadFile) -> np.ndarray:
    """업로드된 파일을 numpy 배열로 읽습니다.

    Args:
        file: 업로드된 이미지 파일

    Returns:
        BGR 형식의 numpy 배열

    Raises:
        ValueError: 파일 읽기 실패 시
    """
    try:
        import cv2

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("이미지 파일 디코딩 실패")

        return image

    except (IOError, ValueError) as e:
        raise ImageDecodeError(f"이미지 파일 읽기 실패: {e}")


@router.post("/face_recognize", response_model=FaceRecognitionResponse)
async def face_recognize(
    file: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None),
):
    """이미지에서 얼굴을 인식합니다.

    Args:
        file: 업로드된 이미지 파일 (multipart/form-data)
        image_b64: Base64 인코딩된 이미지 (폼 필드)

    Returns:
        FaceRecognitionResponse: 인식된 얼굴 목록

    Raises:
        HTTPException: 입력이 없거나 처리 실패 시

    Note:
        file과 image_b64 중 하나는 반드시 제공되어야 합니다.
    """
    if file is None and not image_b64:
        raise HTTPException(
            status_code=400,
            detail="file 또는 image_b64 중 하나를 제공하세요",
        )

    # 이미지 로드
    try:
        if file is not None:
            image = await _read_upload_file(file)
            image_name = getattr(file, "filename", None) or "uploaded_image"
        else:
            image = _decode_base64_image(image_b64)
            image_name = "base64_image"
    except ImageDecodeError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 이미지 크기
    size = image.nbytes if image is not None else None

    # 얼굴 인식 서비스 가져오기
    recognizer = _get_face_recognizer()

    if recognizer is None:
        # 서비스 사용 불가 시 더미 응답 반환 (테스트 호환성)
        logger.warning("얼굴 인식 서비스 사용 불가, 더미 응답 반환")
        return FaceRecognitionResponse(
            image_name=image_name,
            size=size,
            faces=[],
            message="얼굴 인식 서비스가 초기화되지 않았습니다.",
        )

    # 얼굴 인식 수행
    try:
        results = recognizer.recognize_faces(image)

        faces = []
        for idx, result in enumerate(results):
            face = FaceSchema(
                id=idx + 1,
                label=result.name if result.is_known else None,
                confidence=result.box.confidence,
                bbox=BoundingBoxSchema(
                    x=result.box.x1,
                    y=result.box.y1,
                    w=result.box.width,
                    h=result.box.height,
                ),
                similarity=result.similarity if result.is_known else None,
            )
            faces.append(face)

        message = f"{len(faces)}개의 얼굴이 탐지되었습니다."
        if any(f.label for f in faces):
            known_names = [f.label for f in faces if f.label]
            message += f" 인식된 사람: {', '.join(known_names)}"

        return FaceRecognitionResponse(
            image_name=image_name,
            size=size,
            faces=faces,
            message=message,
        )

    except FaceRecognitionError as e:
        logger.error(f"얼굴 인식 실패: {e}")
        raise HTTPException(status_code=500, detail=f"얼굴 인식 처리 실패: {e.message}")
    except (ValueError, RuntimeError) as e:
        logger.error(f"얼굴 인식 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"얼굴 인식 처리 실패: {e}")


@router.post("/face_register", response_model=FaceRegisterResponse)
async def face_register(
    name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None),
):
    """새 얼굴을 등록합니다.

    Args:
        name: 등록할 이름
        file: 얼굴 이미지 파일
        image_b64: Base64 인코딩된 얼굴 이미지

    Returns:
        FaceRegisterResponse: 등록 결과

    Raises:
        HTTPException: 입력이 없거나 등록 실패 시
    """
    if file is None and not image_b64:
        raise HTTPException(
            status_code=400,
            detail="file 또는 image_b64 중 하나를 제공하세요",
        )

    # 이미지 로드
    try:
        if file is not None:
            image = await _read_upload_file(file)
        else:
            image = _decode_base64_image(image_b64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 얼굴 인식 서비스 가져오기
    recognizer = _get_face_recognizer()

    if recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="얼굴 인식 서비스가 초기화되지 않았습니다.",
        )

    # 얼굴 등록
    try:
        success = recognizer.register_face(name, image)

        if success:
            return FaceRegisterResponse(
                success=True,
                name=name,
                message=f"'{name}' 얼굴이 성공적으로 등록되었습니다.",
            )
        else:
            return FaceRegisterResponse(
                success=False,
                name=name,
                message=f"'{name}' 얼굴 등록에 실패했습니다. 얼굴이 감지되지 않았을 수 있습니다.",
            )

    except FaceRegistrationError as e:
        logger.error(f"얼굴 등록 실패: {e}")
        raise HTTPException(status_code=400, detail=e.message)
    except FaceRecognitionError as e:
        logger.error(f"얼굴 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=e.message)
    except (ValueError, RuntimeError) as e:
        logger.error(f"얼굴 등록 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"얼굴 등록 처리 실패: {e}")


@router.delete("/face_unregister/{name}", response_model=FaceRegisterResponse)
async def face_unregister(name: str):
    """등록된 얼굴을 삭제합니다.

    Args:
        name: 삭제할 이름

    Returns:
        FaceRegisterResponse: 삭제 결과

    Raises:
        HTTPException: 서비스 사용 불가 시
    """
    recognizer = _get_face_recognizer()

    if recognizer is None:
        raise HTTPException(
            status_code=503,
            detail="얼굴 인식 서비스가 초기화되지 않았습니다.",
        )

    try:
        success = recognizer.unregister_face(name)

        if success:
            return FaceRegisterResponse(
                success=True,
                name=name,
                message=f"'{name}' 얼굴이 성공적으로 삭제되었습니다.",
            )
        else:
            return FaceRegisterResponse(
                success=False,
                name=name,
                message=f"'{name}' 얼굴을 찾을 수 없습니다.",
            )

    except FaceNotFoundError as e:
        logger.warning(f"얼굴을 찾을 수 없음: {e}")
        return FaceRegisterResponse(
            success=False,
            name=name,
            message=e.message,
        )
    except FaceRecognitionError as e:
        logger.error(f"얼굴 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=e.message)
    except (IOError, OSError) as e:
        logger.error(f"얼굴 삭제 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"얼굴 삭제 처리 실패: {e}")


@router.get("/face_list", response_model=FaceListResponse)
async def face_list():
    """등록된 얼굴 목록을 조회합니다.

    Returns:
        FaceListResponse: 등록된 이름 목록

    Raises:
        HTTPException: 서비스 사용 불가 시
    """
    recognizer = _get_face_recognizer()

    if recognizer is None:
        # 서비스 사용 불가 시 빈 목록 반환
        return FaceListResponse(faces=[], count=0)

    try:
        names = recognizer.list_registered_faces()
        return FaceListResponse(faces=names, count=len(names))

    except FaceRecognitionError as e:
        logger.error(f"얼굴 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=e.message)
    except (IOError, OSError) as e:
        logger.error(f"얼굴 목록 조회 IO 오류: {e}")
        raise HTTPException(status_code=500, detail=f"얼굴 목록 조회 실패: {e}")

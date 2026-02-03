"""
얼굴 인식 서비스 모듈 (Face Recognition Service)

YOLO 기반 얼굴 탐지와 ArcFace 기반 얼굴 인식을 통합하여 제공합니다.

해결 이슈:
- SC-002: face_rec_yolo_arcface_cash.py SRP 위반 → 책임별 클래스 분리
- RC-003: 경로 하드코딩 → config에서 경로 로드
- RD-010: 긴 함수 분리 → synchronize_known_faces 분리

구성:
- FaceDetector: YOLO 기반 얼굴 탐지
- FaceEncoder: ArcFace 기반 임베딩 추출
- EmbeddingCache: 임베딩 캐시 관리 (pickle)
- FaceRecognizer: 통합 인식 서비스

사용 예시:
    from src.services.face_service import FaceRecognizer

    recognizer = FaceRecognizer()
    recognizer.initialize()

    # 단일 이미지에서 얼굴 인식
    results = recognizer.recognize_faces(image)
    for result in results:
        print(f"{result.name}: {result.similarity:.2f}")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """얼굴 바운딩 박스

    Attributes:
        x1, y1: 좌상단 좌표
        x2, y2: 우하단 좌표
        confidence: 탐지 신뢰도
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class FaceRecognitionResult:
    """얼굴 인식 결과

    Attributes:
        name: 인식된 이름 (Unknown이면 미등록자)
        similarity: 유사도 점수 (0.0 ~ 1.0)
        box: 바운딩 박스
        embedding: 얼굴 임베딩 벡터 (선택적)
        is_known: 등록된 얼굴인지 여부
    """

    name: str
    similarity: float
    box: BoundingBox
    embedding: Optional[np.ndarray] = None
    is_known: bool = False


@dataclass
class FaceServiceConfig:
    """얼굴 인식 서비스 설정

    Attributes:
        similarity_threshold: 동일 인물 판정 임계값
        yolo_confidence: YOLO 탐지 신뢰도 임계값
        yolo_model_path: YOLO 모델 파일 경로
        face_images_dir: 등록 얼굴 이미지 디렉토리
        embeddings_cache_path: 임베딩 캐시 파일 경로
        recognition_model: DeepFace 인식 모델명
    """

    similarity_threshold: float = 0.70
    yolo_confidence: float = 0.6
    yolo_model_path: Optional[Path] = None
    face_images_dir: Optional[Path] = None
    embeddings_cache_path: Optional[Path] = None
    recognition_model: str = "ArcFace"

    @classmethod
    def from_settings(cls) -> "FaceServiceConfig":
        """settings에서 설정을 로드합니다."""
        return cls(
            similarity_threshold=settings.similarity_threshold,
            yolo_confidence=settings.yolo_confidence,
            yolo_model_path=settings.yolo_model_path,
            face_images_dir=settings.face_images_dir,
            embeddings_cache_path=settings.face_embeddings_cache,
        )


class FaceDetector:
    """YOLO 기반 얼굴 탐지기

    YOLOv8 모델을 사용하여 이미지에서 얼굴을 탐지합니다.

    Attributes:
        model: YOLO 모델 인스턴스
        confidence: 탐지 신뢰도 임계값
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence: float = 0.6,
    ):
        self.model_path = model_path or settings.yolo_model_path
        self.confidence = confidence
        self.model = None
        self._initialized = False

    def initialize(self) -> None:
        """YOLO 모델을 로드합니다."""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"YOLO 모델 파일을 찾을 수 없습니다: {self.model_path}"
                )

            logger.info(f"YOLO 모델 로드 중: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            self._initialized = True
            logger.info("YOLO 모델 로드 완료")

        except ImportError:
            raise ImportError(
                "ultralytics 패키지가 설치되지 않았습니다: pip install ultralytics"
            )

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
    ) -> List[BoundingBox]:
        """이미지에서 얼굴을 탐지합니다.

        Args:
            image: BGR 형식의 이미지 (OpenCV 형식)
            confidence: 탐지 신뢰도 임계값 (None이면 기본값 사용)

        Returns:
            탐지된 얼굴들의 바운딩 박스 리스트
        """
        if not self._initialized:
            self.initialize()

        conf = confidence if confidence is not None else self.confidence
        results = self.model(image, conf=conf, verbose=False)

        boxes: List[BoundingBox] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0]) if box.conf is not None else conf
                boxes.append(
                    BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf_score,
                    )
                )

        return boxes


class FaceEncoder:
    """ArcFace 기반 얼굴 임베딩 추출기

    DeepFace 라이브러리를 통해 ArcFace 모델로 임베딩을 추출합니다.

    Attributes:
        model_name: DeepFace 모델명 (기본: ArcFace)
    """

    def __init__(self, model_name: str = "ArcFace"):
        self.model_name = model_name
        self._initialized = False

    def initialize(self) -> None:
        """DeepFace 모델을 초기화합니다."""
        if self._initialized:
            return

        try:
            from deepface import DeepFace

            # 더미 이미지로 모델 미리 로드 (첫 호출 시 지연 방지)
            # 실제로는 첫 represent 호출 시 모델이 로드됨
            logger.info(f"DeepFace {self.model_name} 모델 초기화 준비")
            self._initialized = True

        except ImportError:
            raise ImportError(
                "deepface 패키지가 설치되지 않았습니다: pip install deepface"
            )

    def encode(
        self,
        image: Union[np.ndarray, str, Path],
        enforce_detection: bool = True,
    ) -> Optional[np.ndarray]:
        """이미지에서 얼굴 임베딩을 추출합니다.

        Args:
            image: 이미지 (numpy 배열 또는 파일 경로)
            enforce_detection: 얼굴 탐지 강제 여부

        Returns:
            512차원 임베딩 벡터 또는 None (실패 시)
        """
        if not self._initialized:
            self.initialize()

        try:
            from deepface import DeepFace

            img_path = str(image) if isinstance(image, (str, Path)) else image

            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=enforce_detection,
            )

            if embedding_objs and len(embedding_objs) > 0:
                return np.array(embedding_objs[0]["embedding"])

            return None

        except Exception as e:
            logger.warning(f"임베딩 추출 실패: {e}")
            return None

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """두 임베딩 간의 코사인 유사도를 계산합니다.

        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩

        Returns:
            유사도 (0.0 ~ 1.0, 높을수록 유사)
        """
        from scipy.spatial.distance import cosine

        distance = cosine(embedding1, embedding2)
        similarity = 1.0 - distance
        return max(0.0, min(1.0, similarity))


class EmbeddingCache:
    """얼굴 임베딩 캐시 관리자

    등록된 얼굴의 임베딩을 pickle 파일로 저장/로드합니다.
    프로그램 시작 시 캐시를 사용하여 빠르게 초기화할 수 있습니다.

    Attributes:
        cache_path: 캐시 파일 경로
        names: 등록된 이름 리스트
        embeddings: 등록된 임베딩 리스트
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or settings.face_embeddings_cache
        self.names: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def load(self) -> bool:
        """캐시 파일을 로드합니다.

        Returns:
            로드 성공 여부
        """
        if not self.cache_path or not self.cache_path.exists():
            logger.info("캐시 파일이 존재하지 않습니다.")
            return False

        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.names = data.get("names", [])
                self.embeddings = data.get("encodings", [])

            logger.info(f"캐시 로드 완료: {len(self.names)}명")
            return True

        except Exception as e:
            logger.error(f"캐시 로드 실패: {e}")
            return False

    def save(self) -> bool:
        """캐시를 파일에 저장합니다.

        Returns:
            저장 성공 여부
        """
        if not self.cache_path:
            return False

        try:
            # 디렉토리가 없으면 생성
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_path, "wb") as f:
                data = {
                    "names": self.names,
                    "encodings": self.embeddings,
                }
                pickle.dump(data, f)

            logger.info(f"캐시 저장 완료: {len(self.names)}명")
            return True

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False

    def add(self, name: str, embedding: np.ndarray) -> None:
        """새 얼굴을 캐시에 추가합니다."""
        self.names.append(name)
        self.embeddings.append(embedding)

    def remove(self, name: str) -> bool:
        """얼굴을 캐시에서 제거합니다.

        Returns:
            제거 성공 여부
        """
        try:
            idx = self.names.index(name)
            self.names.pop(idx)
            self.embeddings.pop(idx)
            return True
        except ValueError:
            return False

    def get_embedding(self, name: str) -> Optional[np.ndarray]:
        """이름으로 임베딩을 조회합니다."""
        try:
            idx = self.names.index(name)
            return self.embeddings[idx]
        except ValueError:
            return None

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.names


class FaceRecognizer:
    """통합 얼굴 인식 서비스

    FaceDetector, FaceEncoder, EmbeddingCache를 조합하여
    완전한 얼굴 인식 파이프라인을 제공합니다.

    사용 예시:
        recognizer = FaceRecognizer()
        recognizer.initialize()

        # 웹캠에서 얼굴 인식
        results = recognizer.recognize_faces(frame)

        # 새 얼굴 등록
        recognizer.register_face("홍길동", face_image)
    """

    def __init__(self, config: Optional[FaceServiceConfig] = None):
        self.config = config or FaceServiceConfig.from_settings()
        self.detector = FaceDetector(
            model_path=self.config.yolo_model_path,
            confidence=self.config.yolo_confidence,
        )
        self.encoder = FaceEncoder(model_name=self.config.recognition_model)
        self.cache = EmbeddingCache(cache_path=self.config.embeddings_cache_path)
        self._initialized = False

    def initialize(self) -> None:
        """서비스를 초기화하고 캐시를 동기화합니다."""
        if self._initialized:
            return

        logger.info("얼굴 인식 서비스 초기화 중...")

        # 캐시 로드
        self.cache.load()

        # 얼굴 이미지 폴더와 동기화
        self._synchronize_with_folder()

        self._initialized = True
        logger.info(f"초기화 완료: {len(self.cache)}명 등록됨")

    def _synchronize_with_folder(self) -> None:
        """얼굴 이미지 폴더와 캐시를 동기화합니다.

        새로 추가된 이미지는 등록하고, 삭제된 이미지는 캐시에서 제거합니다.
        """
        if not self.config.face_images_dir or not self.config.face_images_dir.exists():
            logger.warning(f"얼굴 이미지 폴더가 없습니다: {self.config.face_images_dir}")
            return

        # 현재 폴더의 이름 목록
        current_names = {
            f.stem for f in self.config.face_images_dir.iterdir() if f.is_file()
        }
        cached_names = set(self.cache.names)

        new_names = current_names - cached_names
        deleted_names = cached_names - current_names

        data_changed = False

        # 삭제된 얼굴 제거
        for name in deleted_names:
            logger.info(f"삭제된 얼굴 제거: {name}")
            self.cache.remove(name)
            data_changed = True

        # 새 얼굴 등록
        for name in new_names:
            # 이미지 파일 찾기
            image_path = None
            for f in self.config.face_images_dir.iterdir():
                if f.stem == name:
                    image_path = f
                    break

            if image_path:
                embedding = self.encoder.encode(image_path, enforce_detection=True)
                if embedding is not None:
                    self.cache.add(name, embedding)
                    logger.info(f"새 얼굴 등록: {name}")
                    data_changed = True
                else:
                    logger.warning(f"얼굴 등록 실패 (임베딩 추출 불가): {name}")

        # 변경 사항 저장
        if data_changed:
            self.cache.save()

    def recognize_faces(
        self,
        image: np.ndarray,
        return_embeddings: bool = False,
    ) -> List[FaceRecognitionResult]:
        """이미지에서 얼굴을 인식합니다.

        Args:
            image: BGR 형식의 이미지
            return_embeddings: 결과에 임베딩 포함 여부

        Returns:
            인식 결과 리스트
        """
        if not self._initialized:
            self.initialize()

        results: List[FaceRecognitionResult] = []

        # 얼굴 탐지
        boxes = self.detector.detect(image)

        for box in boxes:
            try:
                # 얼굴 영역 추출
                face_crop = image[box.y1 : box.y2, box.x1 : box.x2]

                # 임베딩 추출
                embedding = self.encoder.encode(face_crop, enforce_detection=False)

                if embedding is None:
                    results.append(
                        FaceRecognitionResult(
                            name="Unknown",
                            similarity=0.0,
                            box=box,
                            is_known=False,
                        )
                    )
                    continue

                # 등록된 얼굴과 비교
                best_name = "Unknown"
                best_similarity = 0.0

                for i, known_embedding in enumerate(self.cache.embeddings):
                    similarity = self.encoder.compute_similarity(
                        embedding, known_embedding
                    )

                    if (
                        similarity > best_similarity
                        and similarity >= self.config.similarity_threshold
                    ):
                        best_similarity = similarity
                        best_name = self.cache.names[i]

                results.append(
                    FaceRecognitionResult(
                        name=best_name,
                        similarity=best_similarity,
                        box=box,
                        embedding=embedding if return_embeddings else None,
                        is_known=best_name != "Unknown",
                    )
                )

            except Exception as e:
                logger.warning(f"얼굴 인식 중 오류: {e}")
                results.append(
                    FaceRecognitionResult(
                        name="Error",
                        similarity=0.0,
                        box=box,
                        is_known=False,
                    )
                )

        return results

    def register_face(
        self,
        name: str,
        image: Union[np.ndarray, str, Path],
    ) -> bool:
        """새 얼굴을 등록합니다.

        Args:
            name: 등록할 이름
            image: 얼굴 이미지 (numpy 배열 또는 파일 경로)

        Returns:
            등록 성공 여부
        """
        if not self._initialized:
            self.initialize()

        try:
            embedding = self.encoder.encode(image, enforce_detection=True)

            if embedding is None:
                logger.warning(f"얼굴 등록 실패: 임베딩 추출 불가 ({name})")
                return False

            # 이미 등록된 이름이면 업데이트
            if name in self.cache:
                self.cache.remove(name)

            self.cache.add(name, embedding)
            self.cache.save()

            logger.info(f"얼굴 등록 완료: {name}")
            return True

        except Exception as e:
            logger.error(f"얼굴 등록 실패 ({name}): {e}")
            return False

    def unregister_face(self, name: str) -> bool:
        """등록된 얼굴을 제거합니다.

        Args:
            name: 제거할 이름

        Returns:
            제거 성공 여부
        """
        if not self._initialized:
            self.initialize()

        if self.cache.remove(name):
            self.cache.save()
            logger.info(f"얼굴 제거 완료: {name}")
            return True

        logger.warning(f"얼굴 제거 실패: 등록되지 않은 이름 ({name})")
        return False

    def list_registered_faces(self) -> List[str]:
        """등록된 얼굴 이름 목록을 반환합니다."""
        if not self._initialized:
            self.initialize()

        return list(self.cache.names)

    def is_face_registered(self, name: str) -> bool:
        """이름이 등록되어 있는지 확인합니다."""
        if not self._initialized:
            self.initialize()

        return name in self.cache


# 싱글톤 인스턴스 (선택적 사용)
_default_recognizer: Optional[FaceRecognizer] = None


def get_face_recognizer() -> FaceRecognizer:
    """기본 얼굴 인식기를 반환합니다.

    싱글톤 패턴으로 한 번만 생성됩니다.
    """
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = FaceRecognizer()
        _default_recognizer.initialize()
    return _default_recognizer


if __name__ == "__main__":
    print("=== 얼굴 인식 서비스 테스트 ===")

    # 설정 출력
    config = FaceServiceConfig.from_settings()
    print(f"기본 설정:")
    print(f"  유사도 임계값: {config.similarity_threshold}")
    print(f"  YOLO 신뢰도: {config.yolo_confidence}")
    print(f"  모델 경로: {config.yolo_model_path}")
    print(f"  이미지 폴더: {config.face_images_dir}")
    print(f"  캐시 경로: {config.embeddings_cache_path}")

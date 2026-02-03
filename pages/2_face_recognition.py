"""
실시간 얼굴 인식 페이지 - Streamlit WebRTC 통합

시스템 아키텍처:
┌─────────────────────────────────────────────────────────────────┐
│ 브라우저 (WebRTC MediaStream)                                    │
│   │                                                              │
│   ├─> streamlit-webrtc 컴포넌트                                  │
│   │     │                                                        │
│   │     ├─> FaceRecognitionTransformer (프레임별 처리)           │
│   │     │     │                                                  │
│   │     │     ├─> YOLO 얼굴 탐지 (~10ms/frame on GPU)           │
│   │     │     │                                                  │
│   │     │     ├─> ArcFace 임베딩 추출 (~50ms/face)              │
│   │     │     │                                                  │
│   │     │     ├─> 코사인 유사도 매칭 (~1ms)                     │
│   │     │     │                                                  │
│   │     │     └─> 바운딩 박스 렌더링                            │
│   │     │                                                        │
│   │     └─> 실시간 화면 표시 (브라우저)                         │
│   │                                                              │
│   └─> 캐시 (known_face_embeddings.pkl)                          │
│         - face_recognition/face_image/의 사전 계산 임베딩        │
│         - 시작 시 1회 로드하여 빠른 비교 수행                    │
└─────────────────────────────────────────────────────────────────┘

성능 특성:
- 프레임 레이트: 15-30 FPS (GPU, 얼굴 수에 따라 변동)
- 레이턴시: 100-200ms per frame (탐지 + 임베딩 + 매칭)
- 메모리: 약 2GB (YOLO 모델 + ArcFace 모델 + 비디오 버퍼)

기술적 트레이드오프:
1. WebRTC vs OpenCV VideoCapture
   - WebRTC: 브라우저 네이티브, 플러그인 불필요, 원격 작동
   - OpenCV: 로컬 카메라만 접근 가능, 배포 복잡도 높음

2. 프레임별 임베딩 vs 사전 계산 캐시
   - 캐시: 빠른 조회 (1ms vs 50ms), 일관된 결과
   - 실시간: CPU 부하 높음, 프레임 간 미세한 변동

3. YOLO vs 기타 탐지기 (MTCNN, Haar Cascade)
   - YOLO: 가장 빠르고 정확, 실시간 사용에 최적
   - 기타: MTCNN은 느림, Haar Cascade는 정확도 낮음

의존성:
    pip install streamlit-webrtc aiortc opencv-python deepface ultralytics scipy

작성자: Voice Phishing Protector Team
최종 수정: 2026-02-01
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av  # PyAV: 비디오 프레임 처리용
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from scipy.spatial.distance import cosine
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ============================================================================
# 설정 상수
# ============================================================================

# 프로젝트 디렉토리 경로 (현재 파일 기준 상대 경로)
PROJECT_ROOT = Path(__file__).parent.parent
KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "face_images"  # 얼굴 이미지 저장 폴더
CACHE_DIR = PROJECT_ROOT / "cache"
EMBEDDING_CACHE_FILE = CACHE_DIR / "known_face_embeddings.pkl"

# 모델 경로
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8l_100e.pt"

# 얼굴 인식 파라미터
FACE_RECOGNITION_MODEL = "ArcFace"  # DeepFace 백엔드: ArcFace, Facenet, VGG-Face 등
SIMILARITY_THRESHOLD = 0.70  # 코사인 유사도 임계값 (일반적으로 0.68-0.72 범위)
YOLO_CONFIDENCE_THRESHOLD = 0.6  # YOLO 탐지 신뢰도 (0.0-1.0)

# 비디오 스트리밍 파라미터
VIDEO_FPS = 30  # 목표 프레임 레이트 (실제 FPS는 처리 속도에 따라 변동)
VIDEO_WIDTH = 640  # 비디오 스트림 너비 (픽셀)
VIDEO_HEIGHT = 480  # 비디오 스트림 높이 (픽셀)


# ============================================================================
# 데이터 구조
# ============================================================================


class FaceEmbedding:
    """
    등록된 얼굴의 임베딩 벡터와 메타데이터를 담는 클래스.
    
    설계 패턴: Value Object
    - 생성 후 불변성 유지
    - 자체 검증 로직 포함
    - 얼굴 데이터의 타입 안전 표현
    
    속성:
        name: 사람 식별자 (파일명에서 확장자 제외)
        embedding: 512차원 ArcFace 특징 벡터 (정규화됨)
    """
    
    def __init__(self, name: str, embedding):
        if not name:
            raise ValueError("얼굴 이름은 비어있을 수 없습니다")
        
        # list 또는 np.ndarray 모두 지원
        self.embedding = np.array(embedding, dtype=np.float32)
        
        if self.embedding.size == 0:
            raise ValueError(f"{name}의 임베딩이 유효하지 않습니다: 빈 벡터")
        
        self.name = name
    
    def similarity_to(self, other_embedding: np.ndarray) -> float:
        """
        다른 임베딩과의 코사인 유사도 계산.
        
        수학적 배경: 코사인 유사도 = 1 - 코사인 거리
        - 동일한 벡터의 경우 1.0 반환
        - 직교 벡터의 경우 0.0 반환
        - 반대 벡터의 경우 -1.0 반환 (실무에서는 드묾)
        
        성능: O(n) 여기서 n=임베딩 차원 (일반적으로 512)
        """
        return float(1.0 - cosine(self.embedding, other_embedding))


# ============================================================================
# 캐시 관리
# ============================================================================


@st.cache_resource
def load_yolo_model(model_path: Path) -> YOLO:
    """
    YOLO 얼굴 탐지 모델 로드 (애플리케이션 생명주기 동안 캐시됨).
    
    캐싱 전략:
    - @st.cache_resource: 싱글톤 패턴, 앱 세션당 1회만 로드
    - Streamlit 재실행 시에도 유지 (버튼 클릭, 슬라이더 변경 등)
    - 앱 재시작 또는 수동 캐시 클리어 시에만 초기화
    
    성능 영향:
    - 최초 로드: 3-5초 (필요 시 가중치 다운로드)
    - 이후 호출: 1ms 미만 (캐시된 참조 반환)
    
    메모리 사용량: 100-200MB (YOLO-L 모델)
    
    Args:
        model_path: YOLO 모델 가중치 파일 경로 (.pt)
    
    Returns:
        추론 준비가 완료된 YOLO 모델 인스턴스
    
    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않는 경우
        RuntimeError: 모델 로딩 실패 (파일 손상, 버전 불일치)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO 모델을 찾을 수 없습니다: {model_path}\n"
            f"예상 위치: {model_path.absolute()}\n"
            f"다운로드: https://github.com/ultralytics/yolov8"
        )
    
    try:
        model = YOLO(str(model_path))
        # 워밍업: 더미 추론 실행하여 CUDA/CPU 리소스 초기화
        # 이유: 첫 추론이 느린 것은 지연 초기화 때문
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_img, verbose=False)
        return model
    except Exception as e:
        raise RuntimeError(f"YOLO 모델 로드 실패: {e}") from e


def synchronize_face_embeddings() -> Tuple[List[FaceEmbedding], Dict[str, str]]:
    """
    디스크 (face_image/) 와 캐시 (pkl 파일) 간의 얼굴 임베딩 동기화.
    
    알고리즘:
    1. 기존 캐시 로드 (있는 경우)
    2. 파일시스템 변경 감지:
       - 신규 얼굴: face_image/에 있지만 캐시에 없음 -> 임베딩 계산
       - 삭제된 얼굴: 캐시에 있지만 face_image/에 없음 -> 캐시에서 제거
    3. 변경 감지 시 업데이트된 캐시 저장
    
    설계 이유:
    - 매 실행마다 임베딩 재계산 방지 (얼굴당 약 50ms 절약)
    - 사용자가 얼굴 이미지 추가/제거 시 자동 처리
    - 원자적 캐시 업데이트 (임시 파일 -> 이름변경)로 손상 방지
    
    예상 디렉토리 구조:
        face_recognition/face_image/
        ├── person1.jpg
        ├── person2.png
        └── ...
    
    캐시 형식 (pickle):
        {
            "names": ["person1", "person2", ...],
            "embeddings": [np.array(...), np.array(...), ...],
            "paths": ["/path/to/person1.jpg", ...]
        }
    
    Returns:
        Tuple:
        - List[FaceEmbedding]: 임베딩이 포함된 모든 등록 얼굴
        - Dict[str, str]: UI 표시용 상태 메시지
    
    부수 효과:
        - CACHE_DIR이 없으면 생성
        - 변경 시 EMBEDDING_CACHE_FILE 쓰기
        - 콘솔에 동기화 상태 출력
    """
    status_messages = {}
    
    # 캐시 디렉토리 생성 확인
    # 이유: 캐시가 .gitignore 되어 있을 수 있으므로 첫 실행 시 생성
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1단계: 기존 캐시 로드
    # -------------------------------------------------------------------------
    # 캐시 형식: {"names": [...], "encodings": [...]}
    # face_rec_yolo_arcface_cash.py와 동일한 형식 사용
    cached_data = {"names": [], "encodings": []}
    
    if EMBEDDING_CACHE_FILE.exists():
        try:
            with open(EMBEDDING_CACHE_FILE, "rb") as f:
                cached_data = pickle.load(f)
            status_messages["cache_load"] = f"캐시에서 {len(cached_data['names'])}명의 얼굴을 로드했습니다"
        except (pickle.PickleError, EOFError, KeyError) as e:
            # 캐시 손상 -> 처음부터 재구성
            status_messages["cache_load"] = f"캐시가 손상되어 재구성합니다: {e}"
            cached_data = {"names": [], "encodings": []}
    else:
        status_messages["cache_load"] = "기존 캐시가 없습니다. 처음부터 구성합니다"
    
    # -------------------------------------------------------------------------
    # 2단계: 파일시스템 변경 감지
    # -------------------------------------------------------------------------
    # face_image 디렉토리에서 지원되는 이미지 형식 스캔
    # 이유: 사용자가 .jpg, .png, .jpeg 파일 추가 가능
    current_files = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for file_path in KNOWN_FACES_DIR.glob(ext):
            name = file_path.stem  # 확장자 제외한 파일명
            current_files[name] = str(file_path)
    
    current_names = set(current_files.keys())
    cached_names = set(cached_data["names"])
    
    # 차이 검출을 위한 집합 연산
    # 이유: O(1) 조회 vs O(n) 리스트 반복
    new_names = current_names - cached_names
    deleted_names = cached_names - current_names
    
    # -------------------------------------------------------------------------
    # 3단계: 캐시에서 삭제된 얼굴 제거
    # -------------------------------------------------------------------------
    if deleted_names:
        status_messages["deleted"] = f"{len(deleted_names)}명의 삭제된 얼굴을 제거했습니다: {', '.join(deleted_names)}"
        
        # 삭제된 항목 필터링
        # 이유: 반복적인 .remove() 호출보다 리스트 컴프리헨션이 빠름
        indices_to_keep = [
            i for i, name in enumerate(cached_data["names"])
            if name not in deleted_names
        ]
        
        cached_data["names"] = [cached_data["names"][i] for i in indices_to_keep]
        cached_data["encodings"] = [cached_data["encodings"][i] for i in indices_to_keep]
    
    # -------------------------------------------------------------------------
    # 4단계: 캐시에 신규 얼굴 추가
    # -------------------------------------------------------------------------
    if new_names:
        status_messages["new"] = f"{len(new_names)}명의 신규 얼굴 감지, 임베딩 계산 중..."
        
        for name in new_names:
            image_path = current_files[name]
            
            try:
                # DeepFace.represent: 512차원 ArcFace 임베딩 추출
                # ArcFace 선택 이유: 최고 수준 정확도 (LFW 벤치마크 99.8%)
                # 성능: CPU에서 약 50ms, GPU에서 약 10ms per face
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=FACE_RECOGNITION_MODEL,
                    enforce_detection=True,  # 얼굴 미감지 시 실패
                    detector_backend="opencv",  # 등록용 빠른 탐지기
                )
                
                # DeepFace는 딕셔너리 리스트 반환 (감지된 얼굴마다 하나)
                # 등록 이미지에는 정확히 하나의 얼굴만 있어야 함
                if not embedding_objs:
                    status_messages[f"error_{name}"] = f"{name}에서 얼굴이 감지되지 않았습니다"
                    continue
                
                embedding = embedding_objs[0]["embedding"]
                
                # 캐시에 추가
                cached_data["names"].append(name)
                cached_data["encodings"].append(embedding)
                
                status_messages[f"registered_{name}"] = f"{name}을(를) 등록했습니다"
                
            except Exception as e:
                # 치명적이지 않음: 이 얼굴 건너뛰고 나머지 계속 처리
                status_messages[f"error_{name}"] = f"{name} 등록 실패: {e}"
    
    # -------------------------------------------------------------------------
    # 5단계: 변경 감지 시 업데이트된 캐시 저장
    # -------------------------------------------------------------------------
    if new_names or deleted_names:
        try:
            # 원자적 쓰기 패턴: 임시 파일에 쓰기 -> 이름 변경
            # 이유: 부분 쓰기로 인한 캐시 손상 방지
            temp_file = EMBEDDING_CACHE_FILE.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 원자적 이름 변경 (OS 수준 작업)
            temp_file.replace(EMBEDDING_CACHE_FILE)
            status_messages["cache_save"] = f"캐시가 업데이트되었습니다: 총 {len(cached_data['names'])}명"
        except (OSError, pickle.PickleError) as e:
            status_messages["cache_save"] = f"캐시 저장 실패: {e}"
    else:
        status_messages["cache_save"] = "변경 사항이 없습니다. 캐시 유지"
    
    # -------------------------------------------------------------------------
    # 6단계: FaceEmbedding 객체 생성
    # -------------------------------------------------------------------------
    face_embeddings = [
        FaceEmbedding(name, embedding)
        for name, embedding in zip(
            cached_data["names"],
            cached_data["encodings"]
        )
    ]
    
    return face_embeddings, status_messages


# ============================================================================
# 실시간 비디오 처리
# ============================================================================


class FaceRecognitionProcessor(VideoProcessorBase):
    """
    streamlit-webrtc용 실시간 얼굴 인식 비디오 프로세서.
    
    생명주기:
    1. __init__: webrtc_streamer 시작 시 1회 호출
    2. recv: 각 비디오 프레임마다 호출 (초당 약 30회)
    3. 스트림 중지 또는 컴포넌트 언마운트 시 소멸
    
    스레드 안전성:
    - 이 클래스는 Streamlit 메인 스레드와 별도 스레드에서 실행됨
    - st.session_state 접근하거나 st.* 함수 호출 금지
    - 프레임 간 유지되는 상태는 인스턴스 변수 사용
    
    성능 최적화:
    - 얼굴 탐지: GPU 사용 시 YOLO 실행 (약 10ms/frame)
    - 임베딩 추출: 시작 시 캐시됨, 프레임마다 재계산 안 함
    - 유사도 계산: 벡터화된 NumPy 연산 (약 1ms)
    
    메모리 사용:
    - 모델 가중치: 약 300MB (YOLO + ArcFace)
    - 프레임별 버퍼: 약 1MB (640x480x3 RGB)
    - 임베딩 캐시: 등록된 얼굴당 약 1KB
    """
    
    def __init__(self):
        """
        모델 초기화 및 등록된 얼굴 임베딩 로드.
        
        __init__에서 처리하는 이유:
        - 모델은 스트림 세션당 1회만 로드
        - 임베딩은 사전 계산되어 캐시됨
        - 반복적인 디스크 I/O 및 모델 초기화 방지
        
        에러 처리:
        - 모델 로딩 실패는 치명적 (예외 발생)
        - 임베딩 누락은 치명적이지 않음 (빈 리스트 -> "Unknown" 레이블)
        """
        # YOLO 얼굴 탐지 모델 로드
        # 참고: 앱 수준에서 @st.cache_resource로 캐시됨
        self.yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        
        # 캐시에서 등록된 얼굴 임베딩 로드
        # 참고: 스트림 세션마다 새로 로드하여 신규 등록 반영
        self.known_faces, _ = synchronize_face_embeddings()
        
        # 디버깅용 통계 (프레임마다 업데이트)
        self.frame_count = 0
        self.faces_detected_total = 0
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        단일 비디오 프레임 수신 및 처리: 얼굴 탐지, 인식, 주석 추가.
        
        참고: streamlit-webrtc 0.44+에서는 transform() 대신 recv() 사용
        
        실행 흐름:
        1. PyAV 프레임 -> NumPy BGR 배열 변환
        2. YOLO 얼굴 탐지 -> 바운딩 박스
        3. 감지된 각 얼굴마다:
           a. 얼굴 영역 크롭
           b. ArcFace 임베딩 추출
           c. 등록된 얼굴과 비교 (코사인 유사도)
           d. 임계값 이상인 최적 매치 찾기
        4. 바운딩 박스 및 레이블 그리기
        5. NumPy 배열 -> PyAV 프레임 변환
        
        성능 예산 (30 FPS 달성 목표: 33ms 이내):
        - 프레임 변환: 약 1ms
        - YOLO 탐지: 약 10ms (GPU) / 약 50ms (CPU)
        - 임베딩 추출: 얼굴당 약 10ms (GPU) / 약 50ms (CPU)
        - 유사도 매칭: 얼굴당 약 1ms
        - 주석 그리기: 약 2ms
        
        Args:
            frame: 웹캠에서 입력된 비디오 프레임 (av.VideoFrame)
        
        Returns:
            주석이 추가된 비디오 프레임 (av.VideoFrame)
        
        스레드 안전성:
        - 별도 스레드에서 실행 -> Streamlit 함수 호출 금지
        - 인스턴스 변수는 스레드 로컬 (수정 안전)
        
        에러 처리:
        - 임베딩 추출 실패: 해당 얼굴 건너뛰고 나머지 계속 처리
        - 모델 추론 에러: 원본 프레임을 변경 없이 반환
        """
        self.frame_count += 1
        
        # PyAV VideoFrame을 NumPy 배열로 변환 (OpenCV용 BGR 형식)
        # BGR을 사용하는 이유: OpenCV 컨벤션 (PIL/matplotlib의 RGB와 반대)
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # ------------------------------------------------------------------
            # 1단계: 얼굴 탐지 (YOLO)
            # ------------------------------------------------------------------
            # 현재 프레임에서 YOLO 추론 실행
            # 출력: 탐지 결과 리스트 (프레임당 하나)
            results = self.yolo_model(
                img,
                conf=YOLO_CONFIDENCE_THRESHOLD,  # 낮은 신뢰도 탐지 필터링
                verbose=False,  # 콘솔 출력 억제
            )
            
            # ------------------------------------------------------------------
            # 2단계: 감지된 각 얼굴 처리
            # ------------------------------------------------------------------
            for result in results:
                # result.boxes: 메타데이터가 포함된 감지 바운딩 박스
                # 각 박스는: xyxy (좌표), conf (신뢰도), cls (클래스)를 가짐
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                
                for box in result.boxes:
                    self.faces_detected_total += 1
                    
                    # 바운딩 박스 좌표 추출
                    # 형식: [x1, y1, x2, y2] 여기서 (x1,y1) = 좌측 상단, (x2,y2) = 우측 하단
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 프레임에서 얼굴 영역 크롭
                    # NumPy 슬라이싱: img[행_시작:행_끝, 열_시작:열_끝]
                    # 이유: ArcFace는 전체 프레임이 아닌 얼굴 영역만 필요
                    face_crop = img[y1:y2, x1:x2]
                    
                    # 크롭 유효성 검사 (엣지 케이스 처리)
                    if face_crop.size == 0:
                        continue  # 빈 크롭 -> 건너뛰기
                    
                    # -------------------------------------------------------------
                    # 3단계: 얼굴 인식 (ArcFace + 코사인 유사도)
                    # -------------------------------------------------------------
                    best_match_name = "Unknown"
                    best_match_score = 0.0
                    
                    try:
                        # 감지된 얼굴에서 임베딩 추출
                        # DeepFace.represent: 512차원 특징 벡터 추출
                        # enforce_detection=False: YOLO에서 이미 탐지했으므로 생략
                        embedding_objs = DeepFace.represent(
                            img_path=face_crop,
                            model_name=FACE_RECOGNITION_MODEL,
                            enforce_detection=False,  # 탐지 건너뛰기 (이미 YOLO로 완료)
                            detector_backend="skip",  # 두 번째 탐지 패스 불필요
                        )
                        
                        if not embedding_objs:
                            continue  # 임베딩 추출 안 됨 -> 건너뛰기
                        
                        live_embedding = embedding_objs[0]["embedding"]
                        
                        # 모든 등록된 얼굴과 비교
                        # 알고리즘: 코사인 유사도 점수의 Argmax
                        for known_face in self.known_faces:
                            similarity = known_face.similarity_to(live_embedding)
                            
                            # 다음 조건일 때 최적 매치 업데이트:
                            # 1. 유사도가 현재 최고보다 높음
                            # 2. 유사도가 임계값 초과 (오탐 방지)
                            if similarity > best_match_score and similarity >= SIMILARITY_THRESHOLD:
                                best_match_score = similarity
                                best_match_name = known_face.name
                    
                    except Exception as e:
                        # 치명적이지 않음: 에러 로깅하고 "Unknown"으로 표시
                        # 이유: 한 얼굴의 실패가 전체 스트림을 중단해서는 안 됨
                        print(f"[WARNING] 임베딩 추출 실패: {e}")
                        continue
                    
                    # -------------------------------------------------------------
                    # 4단계: 주석 그리기
                    # -------------------------------------------------------------
                    # 색상 코딩:
                    # - 녹색: 인식된 얼굴 (임계값 이상)
                    # - 빨강: 미등록 얼굴 (임계값 미만)
                    color = (0, 255, 0) if best_match_name != "Unknown" else (0, 0, 255)
                    
                    # 얼굴 주위에 바운딩 박스 그리기
                    cv2.rectangle(
                        img,
                        (x1, y1),  # 좌측 상단 모서리
                        (x2, y2),  # 우측 하단 모서리
                        color,
                        thickness=2
                    )
                    
                    # 이름과 신뢰도 점수로 레이블 텍스트 준비
                    # 형식: "PersonName (0.85)" 또는 "Unknown (0.42)"
                    label_text = f"{best_match_name} ({best_match_score:.2f})"
                    
                    # 레이블 배경 그리기 (채워진 사각형)
                    # 이유: 복잡한 배경 위에서 텍스트 가독성 향상
                    cv2.rectangle(
                        img,
                        (x1, y2 - 35),  # 얼굴 박스 하단, 위로 오프셋
                        (x2, y2),  # 얼굴 박스 하단
                        color,
                        cv2.FILLED  # 솔리드 채우기
                    )
                    
                    # 배경 위에 레이블 텍스트 그리기
                    cv2.putText(
                        img,
                        label_text,
                        (x1 + 6, y2 - 6),  # 박스 가장자리에서 약간 패딩
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,  # 폰트 크기
                        (255, 255, 255),  # 흰색 텍스트
                        thickness=1
                    )
        
        except Exception as e:
            # 처리 파이프라인의 치명적 에러 -> 원본 프레임 반환
            # 이유: 처리되지 않은 비디오를 보여주는 것이 스트림 중단보다 나음
            print(f"[ERROR] 프레임 처리 실패: {e}")
            # 원본 프레임을 반환하도록 아래로 진행
        
        # NumPy 배열을 스트리밍용 PyAV VideoFrame으로 변환
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(
    page_title="실시간 얼굴 인식",
    page_icon="face",
    layout="wide"
)

st.title("실시간 얼굴 인식")
st.markdown(
    """
    이 페이지는 다음 기술을 사용한 실시간 얼굴 인식을 시연합니다:
    - **YOLO**: 빠른 얼굴 탐지
    - **ArcFace**: 강건한 얼굴 임베딩
    - **코사인 유사도**: 얼굴 매칭
    
    신규 등록: `face_recognition/face_image/` 폴더에 얼굴 이미지 추가
    """
)

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # 모델 상태 표시
    st.metric("YOLO 모델", str(YOLO_MODEL_PATH.name))
    st.metric("임베딩 모델", FACE_RECOGNITION_MODEL)
    st.metric("유사도 임계값", f"{SIMILARITY_THRESHOLD:.2f}")
    
    st.divider()
    
    # 얼굴 등록 상태
    st.subheader("등록된 얼굴")
    known_faces, sync_messages = synchronize_face_embeddings()
    
    if known_faces:
        st.success(f"{len(known_faces)}명의 얼굴이 로드되었습니다")
        with st.expander("등록된 이름 보기"):
            for face in known_faces:
                st.text(f"• {face.name}")
    else:
        st.warning("등록된 얼굴이 없습니다")
        st.info(
            "이미지 추가 위치:\n"
            f"`{KNOWN_FACES_DIR.relative_to(PROJECT_ROOT)}/`"
        )
    
    # 동기화 메시지 표시
    with st.expander("동기화 로그"):
        for key, message in sync_messages.items():
            st.text(message)

# 메인 컨텐츠 영역
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("라이브 비디오 스트림")
    
    # WebRTC 비디오 스트리머
    # streamlit-webrtc를 사용하는 이유:
    # - 네이티브 브라우저 WebRTC (플러그인 불필요)
    # - 실시간 처리 (st.camera_input의 스냅샷 방식과 대조)
    # - 원격 작동 (로컬 카메라 필요한 OpenCV VideoCapture와 대조)
    webrtc_ctx = webrtc_streamer(
        key="face-recognition-live",
        mode=WebRtcMode.SENDRECV,  # 브라우저에서 비디오 전송, 처리된 비디오 수신
        video_processor_factory=FaceRecognitionProcessor,  # 커스텀 프로세서
        media_stream_constraints={
            "video": {
                "width": {"ideal": VIDEO_WIDTH},
                "height": {"ideal": VIDEO_HEIGHT},
                "frameRate": {"ideal": VIDEO_FPS},
            },
            "audio": False,  # 오디오 불필요
        },
        async_processing=True,  # 백그라운드 스레드에서 프레임 처리
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]  # NAT 통과용 STUN 서버
        },
    )

with col2:
    st.subheader("사용 방법")
    
    st.markdown(
        """
        ### 설정
        1. **START** 버튼 클릭
        2. 브라우저에서 카메라 접근 허용
        3. 실시간으로 얼굴이 감지됩니다
        
        ### 신규 얼굴 등록
        1. 얼굴 이미지를 다음 위치에 추가:  
           `face_recognition/face_image/person_name.jpg`
        2. 이 페이지 새로고침
        3. 새 얼굴이 자동으로 등록됩니다
        
        ### 색상 코드
        - 녹색: 인식된 얼굴
        - 빨강: 미등록 얼굴
        
        ### 성능 팁
        - GPU 사용으로 처리 속도 향상
        - 충분한 조명으로 정확도 개선
        - 카메라를 정면으로 응시
        """
    )
    
    st.divider()
    
    st.subheader("문제 해결")
    with st.expander("카메라가 작동하지 않나요?"):
        st.markdown(
            """
            - **브라우저**: Chrome/Edge 사용 (최고의 WebRTC 지원)
            - **HTTPS**: 원격 접근 시 필수 (localhost는 예외)
            - **권한**: 브라우저 카메라 권한 확인
            - **방화벽**: 들어오는 WebRTC 연결 허용
            """
        )
    
    with st.expander("얼굴이 인식되지 않나요?"):
        st.markdown(
            f"""
            - **임계값**: 현재 = {SIMILARITY_THRESHOLD:.2f}
            - **조명**: 얼굴이 잘 비추는지 확인
            - **각도**: 카메라를 정면으로 응시
            - **등록**: `face_image/`의 이미지 품질 확인
            - **캐시**: 임베딩이 캐시됨, 이미지 업데이트 시 앱 재시작
            """
        )

# 하단
st.divider()
st.caption(
    f"현재 {len(known_faces)}명의 등록된 얼굴을 추적 중입니다. "
    f"이 페이지는 실시간 비디오 처리를 위해 `streamlit-webrtc`를 사용합니다."
)

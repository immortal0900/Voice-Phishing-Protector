import cv2
import os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pickle
# 1. 설정 , 모델 로딩
# ----------------------------------------------------------

# YOlO 얼굴 탐지 모델 로드
YOLO_MODEL_PATH = "models\yolov8l_100e.pt"
# YOLO_MODEL_PATH = "models\yolov8x-face-lindevs.pt"


yolo_model = YOLO(YOLO_MODEL_PATH)

# 인식할 얼굴 이미지들이 저장된 폴더
KNOWN_FACES_DIR = "face_image"

# 얼굴 임베딩 모델(ArcFace)
FACE_RECOGNITION_MODEL = "ArcFace"

# 얼굴 유사도 임계값(이 값 이상이면 같은 사람)
# 코사인 유사도는 1에 가까울수록 비슷함
# 보통 0.68~0.72 사이값
SIMILARITY_THRESHOLD = 0.70

# 등록된 얼굴의 임베딩과 이름 저장
known_face_encodings = []
known_face_names = []


# 2. 등록된 얼굴 로딩 및  임베딩 추출
# ----------------------------------------------------------
def load_known_faces():
    # KNOWN_FACES_DIR 폴더에 있는 모든 이미지 읽어서 각 얼굴 임베딩 계산 후 전역리스트에 저장
    print("등록된 얼굴 로딩 및 임베딩 추출 시작...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        # 이미지 파일 경로
        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        # 파일 이름에서 확장자를 제거하여 사람 이름으로 사용
        person_name = os.path.splitext(filename)[0]
        # [0] 두개로 쪼갠것 중 앞에꺼

        try:
            # DeepFace를 사용하여 얼굴 임베딩 추출
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=FACE_RECOGNITION_MODEL,
                enforce_detection=True,  # 이미지에서 얼굴 반드시 찾도록
            )

            embedding = embedding[0]["embedding"]
            # represent 함수는 리스트 형태로 반환하므로, 첫번째 요소의 임베딩 값 사용

            known_face_encodings.append(embedding)
            known_face_names.append(person_name)

            print(f"{person_name}님의 얼굴을 성공적으로 등록했습니다.")

        except Exception as e:
            # 이미지에서 얼굴을 찾지 못하거나 에러발생
            print(f"{person_name}님의 얼굴을 등록하는데 실패했습니다. 에러: {e}")


# 3. 실시간 영상 처리
# ----------------------------------------------------------
# 등록 함수 호출
load_known_faces()

# 웹캠 열기
video_capture = cv2.VideoCapture(0)
# (0)은 기본 웹캠 번호
print("웹캠을 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        # ret은 불리언 값, 프레임을 읽으면 True, 읽지 못하면 False
        break
    yolo_results = yolo_model(frame, conf=0.6)
    # conf=0.6: 60% 이상 확신하는 얼굴만 결과에 포함

    # 탐지된 각 얼굴에 대해 반복
    for result in yolo_results:
        for box in result.boxes:
            # result.boxes는 탐지된 얼굴의 바운딩 박스 정보
            # box.xyxy는 객체에게 상자좌표를 xyxy 형식으로 요청하는 것, xywh일 경우 중심점 x, y와 너비, 높이 형식으로 반환
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            try:
                face_crop = frame[y1:y2, x1:x2]
                # 탐지된 얼굴 영역을 잘라낸다. (OpenCV는 높이, 너비 순서)
                # frame[세로 범위, 가로 범위]

                # 잘라낸 얼굴 영역을 임베딩 추출
                embedding_objs = DeepFace.represent(
                    img_path=face_crop,
                    model_name=FACE_RECOGNITION_MODEL,
                    enforce_detection=True,
                    # 이미 얼굴을 잘라냈기에 추가 탐지 방지지
                )
                live_embedding = embedding_objs[0]["embedding"]

                best_match_name = "Unknown"
                # 초기값 설정
                best_match_score = 0

                for i, known_embedding in enumerate(known_face_encodings):
                    distance = cosine(live_embedding, known_embedding)
                    # 코사인 거리 계산 (0에 가까울수록 유사)
                    similarity = 1 - distance
                    # 1에 가까울수록 유사(코사인 유사도)

                    if (
                        similarity > best_match_score
                        and similarity >= SIMILARITY_THRESHOLD
                    ):
                        # 현재까지의 최고 점수보다 높고, 임계값 이상이면
                        best_match_score = similarity
                        # 더 높은 유사도 갱신
                        best_match_name = known_face_names[i]

                # 결과 표시
                name_to_display = f"{best_match_name}({best_match_score:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 얼굴 주위에 사각형

                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                # 이름표 배경 그리기

                cv2.putText(
                    frame,
                    name_to_display,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                )
                # 이름 쓰기

            except Exception as e:
                # 실시간 임베딩 계산 중 오류 발생시
                print(f"기존얼굴과의 유사도 분석중 오류: {e}")
                pass

    cv2.imshow("Video", frame)
    # 최종화면 출력

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        # 사용자가 키보드 'q'를 누를 때까지 1ms씩 계속 기다리다가, 'q'가 눌리면 while 루프를 탈출해라
        # (1): 괄호 안의 숫자는 **기다리는 시간(milliseconds)**을 의미. 1은 1ms (0.001초)를 기다리라는 뜻
        # ord()= 문자를 아스키 코드로 변환, 쓰는 이유 cv2.waitKey()가 반환하는 값이 아스키 코드로 반환되기 때문
        # cv2.waitKey()가 반환하는 키보드 값은 8비트(1바이트)보다 큰 값을 가질 수 있음
        # 0xFF는 마지막 8비트 값만 남기고 나머지는 0으로 만듬
        # 따라서 0xFF는 키보드 입력 값을 동일하게 8비트 아스키코드로 만들어주기 위해 필요함

# 종료
video_capture.release() # video_capture을 종료
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")

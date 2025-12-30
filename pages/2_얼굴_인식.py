import streamlit as st
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="얼굴 인식", layout="wide")
st.title("얼굴 인식")

DEFAULT_MODEL_PATH = os.path.join(os.getcwd(), "models", "yolov8l_100e.pt")


@st.cache_resource
def load_yolo_model(path: str):
	try:
		from ultralytics import YOLO
	except Exception as e:
		raise RuntimeError(
			"패키지 'ultralytics'가 설치되어 있지 않습니다. 설치: uv pip install ultralytics opencv-python"
		) from e
	if not os.path.exists(path):
		raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
	model = YOLO(path)
	return model


def parse_results(result):
	# Ultralitycs v8: result.boxes.xyxy, result.boxes.conf, result.boxes.cls
	boxes = np.empty((0, 4))
	confs = np.array([])
	classes = np.array([])
	if hasattr(result, 'boxes') and result.boxes is not None:
		try:
			boxes = result.boxes.xyxy.cpu().numpy()
			confs = result.boxes.conf.cpu().numpy()
			classes = result.boxes.cls.cpu().numpy()
		except Exception:
			# Fallback if tensors are not on CPU
			boxes = np.array(result.boxes.xyxy.tolist())
			confs = np.array(result.boxes.conf.tolist())
			classes = np.array(result.boxes.cls.tolist())
	return boxes, confs, classes


def draw_boxes(img_np: np.ndarray, boxes: np.ndarray, confs: np.ndarray):
	import cv2
	img_draw = img_np.copy()
	for i, box in enumerate(boxes):
		x1, y1, x2, y2 = map(int, box)
		cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
		label = f"{confs[i]:.2f}"
		cv2.putText(img_draw, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	return img_draw


with st.sidebar:
	st.header("설정")
	model_path = st.text_input("모델 파일 경로", value=DEFAULT_MODEL_PATH)
	use_camera = st.checkbox("카메라 입력 사용 (사진 캡쳐)", value=False)
	conf_threshold = st.slider("신뢰도 임계값", min_value=0.0, max_value=1.0, value=0.3)

col1, col2 = st.columns([1, 1])

with col1:
	st.subheader("입력")
	img_file = None
	cam_image = None
	if use_camera:
		cam_image = st.camera_input("카메라로 사진 찍기")
	else:
		img_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
	run_button = st.button("검출 실행")

with col2:
	st.subheader("결과")
	result_area = st.empty()

model = None
load_error = None
if run_button:
	# Load model lazily and cache
	try:
		model = load_yolo_model(model_path)
	except Exception as e:
		load_error = e

	if load_error is not None:
		st.error(str(load_error))
	else:
		# Get image from either camera or upload
		pil_img = None
		if use_camera and cam_image is not None:
			pil_img = Image.open(cam_image).convert("RGB")
		elif not use_camera and img_file is not None:
			pil_img = Image.open(img_file).convert("RGB")

		if pil_img is None:
			st.warning("이미지를 제공해주세요 (업로드 또는 카메라).")
		else:
			img_np = np.array(pil_img)
			try:
				results = model(img_np)
			except Exception as e:
				st.error(f"모델 추론 실패: {e}")
				results = None

			if results is None or len(results) == 0:
				result_area.info("검출 결과가 없습니다.")
			else:
				r = results[0]
				boxes, confs, classes = parse_results(r)
				# Apply confidence threshold
				keep = confs >= conf_threshold if confs.size else np.array([])
				if keep.size:
					boxes = boxes[keep]
					confs = confs[keep]
					classes = classes[keep]

				if boxes.shape[0] == 0:
					result_area.info("신뢰도 임계값을 넘는 검출이 없습니다.")
				else:
					img_boxes = draw_boxes(img_np, boxes, confs)
					result_area.image(img_boxes, caption="검출 결과", use_column_width=True)

					st.subheader("검출된 얼굴 크롭")
					crops_cols = st.columns(min(4, boxes.shape[0]))
					for i, box in enumerate(boxes):
						x1, y1, x2, y2 = map(int, box)
						crop = img_np[y1:y2, x1:x2]
						if crop.size == 0:
							crops_cols[i % 4].text("빈 크롭")
						else:
							crops_cols[i % 4].image(Image.fromarray(crop), use_column_width=True)

st.markdown("---")
st.caption(f"모델 경로: {model_path}")

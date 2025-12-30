import streamlit as st

# streamlit run app.py --server.address 0.0.0.0 --server.port 8501

st.set_page_config(
    page_title="AI 기반 분석 도구",
    page_icon="🤖",
    layout="wide",
)

st.title("AI 기반 분석 도구")
st.sidebar.success("위에서 분석할 페이지를 선택하세요.")

st.write(
    """
    좌측 사이드바에서 원하는 분석 도구를 선택하세요.
    - **보이스 피싱 검사**: 오디오 파일을 분석하여 보이스피싱 위험도를 탐지합니다.
    - **얼굴 인식**: (준비 중)
    """
)

# Try to preload PLM/LLM from phishing_project so Streamlit run starts with models ready when possible
model_status = "not attempted"
try:
    from phishing_project import inference as inference
    # load PLM; do not force LLM load because it may be large — only attempt PLM by default
    try:
        inference.load_plm()
        model_status = "PLM loaded"
    except Exception as e:
        model_status = f"PLM load failed: {e}"
    # Expose a button to load the LLM adapter on demand
    if st.sidebar.button('LLM 로드 (수동)'):
        try:
            inference.load_llm()
            model_status = "LLM loaded"
        except Exception as e:
            model_status = f"LLM load failed: {e}"
except Exception as e:
    model_status = f"inference module not available: {e}"

st.sidebar.markdown(f"**모델 상태:** {model_status}")

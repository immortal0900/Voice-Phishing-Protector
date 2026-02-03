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

# ============================================================================
# 모델 사전 로딩 전략
# ============================================================================
# 첫 요청 지연시간을 줄이기 위해 시작 시 PLM(패턴 기반 언어 모델)을 사전 로드합니다.
# LLM(대규모 언어 모델)은 메모리 사용량이 높아 요청 시 로드합니다.
#
# 아키텍처 참고사항:
# - 일관성을 위해 src/api/voice_analysis.py와 동일한 import 패턴 사용
# - scripts/finetuning/inference.py 접근을 위해 sys.path 주입 사용
# - 레거시 finturing_plm_and_slm/inference.py 모듈 import 회피
# ============================================================================

model_status = "not attempted"

try:
    import sys
    from pathlib import Path
    
    # Python 모듈 검색 경로에 scripts/finetuning 추가
    # 이유: inference 모듈이 패키지가 아니므로 명시적 경로 주입 필요
    # 트레이드오프: 상대 import 사용 가능하지만, 명시적 경로가 디버깅에 더 명확함
    finetuning_path = Path(__file__).parent / "scripts" / "finetuning"
    if str(finetuning_path) not in sys.path:
        sys.path.insert(0, str(finetuning_path))
    
    # 활성 inference 모듈 import (scripts/finetuning/inference.py)
    # 참고: FastAPI 백엔드에서 사용하는 것과 동일한 모듈
    import inference
    
    # PLM (패턴 기반 언어 모델) 사전 로드
    # 이유: PLM은 경량(약 500MB)이며 즉각적인 위험 평가 제공
    # 트레이드오프: 시작 시간 +2-3초, 하지만 첫 요청 콜드 스타트 제거
    try:
        inference.load_plm()
        model_status = "PLM이 성공적으로 로드되었습니다"
    except FileNotFoundError as e:
        model_status = f"PLM 모델 파일을 찾을 수 없습니다: {e}"
    except Exception as e:
        model_status = f"PLM 로드 실패: {e}"
    
    # 수동 LLM 로딩 버튼 노출
    # 이유: LLM(Gemma 2B)은 약 4GB VRAM/RAM 필요, 따라서 요청 시 로드
    # 사용자 경험: 사용자가 메모리 비용을 지불할 시점을 선택 가능
    if st.sidebar.button('LLM 로드 (수동)', help="대용량 모델 로드 (4GB+ 메모리 필요)"):
        try:
            inference.load_llm()
            model_status = "LLM이 성공적으로 로드되었습니다"
        except Exception as e:
            model_status = f"LLM 로드 실패: {e}"

except ImportError as e:
    # 모듈을 찾을 수 없음 - 의존성 누락 또는 잘못된 경로 가능성
    model_status = f"inference 모듈 import 실패: {e}"
except Exception as e:
    # 모듈 초기화 중 예상치 못한 에러 포괄 처리
    model_status = f"모델 설정 중 예상치 못한 에러: {e}"

st.sidebar.markdown(f"**모델 상태:** {model_status}")

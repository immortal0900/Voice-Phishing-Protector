import os
import time
import tempfile
import json
import re
import shutil
from typing import Optional

import requests
import streamlit as st


def extract_clean_json(text: str) -> Optional[str]:
    """LLM에서 반환한 문자열에서 JSON만 추출해 pretty JSON 문자열로 반환.
    실패 시 None 반환."""
    if text is None:
        return None
    # 1) ```json ... ``` 코드블록 우선
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = m.group(1) if m else text
    # 2) 그대로 시도
    for s in [candidate, candidate.strip(), candidate.strip().strip('"\'')]:
        try:
            obj = json.loads(s)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            pass
    # 3) 첫 '{'부터 마지막 '}' 사이를 잘라서 재시도
    if '{' in candidate and '}' in candidate and candidate.find('{') < candidate.rfind('}'):
        sliced = candidate[candidate.find('{'):candidate.rfind('}') + 1]
        try:
            obj = json.loads(sliced)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return None

def write_llm_input(out_dir: str, text: str):
    """LLM에 전달된 입력을 파일에 기록합니다."""
    try:
        with open(os.path.join(out_dir, "last_llm_input.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

def backend_base_url() -> str:
    return st.session_state.get("backend_url", "http://127.0.0.1:8000")

def backend_start(file_path: str, *, fw_model: str, device: str, compute_type: str, chunk_sec: int, simulate_realtime: bool, ollama_url: str, gemma_model: str, sys_prompt: Optional[str]) -> str:
    url = f"{backend_base_url()}/pipeline/start"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        data = {
            "fw_model": fw_model,
            "device": device,
            "compute_type": compute_type,
            "chunk_sec": str(chunk_sec),
            "simulate_realtime": str(simulate_realtime).lower(),
            "ollama_url": ollama_url,
            "gemma_model": gemma_model,
            "sys_prompt": sys_prompt or "",
        }
        resp = requests.post(url, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["run_id"]


def backend_stop(run_id: str) -> None:
    url = f"{backend_base_url()}/pipeline/stop"
    resp = requests.post(url, data={"run_id": run_id}, timeout=30)
    resp.raise_for_status()


def backend_status(run_id: str) -> str:
    url = f"{backend_base_url()}/pipeline/status"
    resp = requests.get(url, params={"run_id": run_id}, timeout=10)
    resp.raise_for_status()
    return resp.text


def backend_log(run_id: str, tail: int = 200) -> str:
    url = f"{backend_base_url()}/pipeline/log"
    resp = requests.get(url, params={"run_id": run_id, "tail": tail}, timeout=10)
    resp.raise_for_status()
    return resp.text


def backend_result(run_id: str) -> str:
    url = f"{backend_base_url()}/pipeline/result"
    resp = requests.get(url, params={"run_id": run_id}, timeout=10)
    if resp.headers.get("content-type", "").startswith("application/json"):
        return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    return resp.text


def backend_llm_input(run_id: str) -> str:
    url = f"{backend_base_url()}/pipeline/llm_input"
    resp = requests.get(url, params={"run_id": run_id}, timeout=10)
    resp.raise_for_status()
    return resp.text


def start_backend_run(file_path: str, *, fw_model: str, device: str, compute_type: str, chunk_sec: int, simulate_realtime: bool, ollama_url: str, gemma_model: str, sys_prompt: Optional[str]) -> str:
    return backend_start(
        file_path,
        fw_model=fw_model,
        device=device,
        compute_type=compute_type,
        chunk_sec=chunk_sec,
        simulate_realtime=simulate_realtime,
        ollama_url=ollama_url,
        gemma_model=gemma_model,
        sys_prompt=sys_prompt,
    )

def setup_session_state():
    """세션 상태를 초기화합니다."""
    if 'running' not in st.session_state:
        st.session_state['running'] = False
    if 'stopped' not in st.session_state:
        st.session_state['stopped'] = False
    if 'audio_path' not in st.session_state:
        st.session_state['audio_path'] = None
    if 'out_dir' not in st.session_state:
        st.session_state['out_dir'] = None
    if 'pipeline_thread' not in st.session_state:
        st.session_state['pipeline_thread'] = None
    if 'run_id' not in st.session_state:
        st.session_state['run_id'] = None
    if 'backend_url' not in st.session_state:
        st.session_state['backend_url'] = "http://127.0.0.1:8000"

def display_ui_sidebar():
    """사이드바 UI 요소를 표시하고 설정을 반환합니다."""
    with st.sidebar:
        st.header("설정")
        # fw_model = st.selectbox("faster-whisper 모델", ["tiny", "base", "small", "medium", "large-v3"], index=2)
        fw_model = "small"  # 기본값으로 고정
        device = st.selectbox("디바이스", ["cuda", "cpu"], index=0)
        compute_type = st.selectbox("Compute Type", ["float16", "int8_float16", "int8"], index=0)
        chunk_sec = st.number_input("청크 길이(초)", min_value=1, max_value=15, value=5, step=1)
        simulate_realtime = st.checkbox("리얼타임 시뮬레이션(청크 간 대기)", value=True)
        auto_refresh = st.checkbox("자동 갱신", value=True, help="로그/결과를 0.5초 간격으로 자동 갱신합니다.")

    st.subheader("백엔드/LLM")
    # backend_url = st.text_input("Backend URL (FastAPI)", value=st.session_state.get("backend_url", "http://127.0.0.1:8000"))
    backend_url = "http://127.0.0.1:8000"  # 기본값으로 고정
    st.session_state["backend_url"] = backend_url
    # ollama_url = st.text_input("Ollama URL", value="http://127.0.0.1:11434")
    ollama_url = "http://127.0.0.1:11434"  # 기본값으로 고정
    # gemma_model = st.text_input("Ollama 모델명", value="gemma2:9b")
    gemma_model = "gemma2:9b"  # 기본값으로 고정
    
    sys_prompt_path = os.path.join(os.getcwd(), "system_ko.txt")
    sys_prompt = None
    if os.path.exists(sys_prompt_path):
        try:
            with open(sys_prompt_path, "r", encoding="utf-8") as f:
                sys_prompt = f.read()
        except Exception as e:
            st.warning(f"system_ko.txt 읽기 실패: {e}")
    else:
        st.warning("system_ko.txt 파일을 찾지 못했습니다. 프로젝트 루트에 배치해 주세요.")

    with st.expander("시스템 프롬프트 미리보기", expanded=False):
        st.code((sys_prompt or "(비어 있음)"), language="markdown")
            
    return fw_model, device, compute_type, chunk_sec, simulate_realtime, auto_refresh, backend_url, ollama_url, gemma_model, sys_prompt

def display_results(run_id: str):
    """결과(상태, 로그, LLM 출력)를 표시합니다."""
    st.subheader("실시간 처리 현황")

    # Status
    st.caption("Status")
    try:
        status_text = backend_status(run_id)
        st.code(status_text)
    except Exception as e:
        st.error(f"Status 요청 실패: {e}")

    # Log
    st.caption("실시간 로그")
    try:
        log_text = backend_log(run_id, tail=500)
        st.text_area("Log", value=log_text, height=200, disabled=True)
    except Exception as e:
        st.text_area("Log", value=f"로그 요청 실패: {e}", height=200, disabled=True)

    # LLM Input
    st.caption("STT 변환 결과 (LLM 입력)")
    try:
        llm_in = backend_llm_input(run_id)
        st.text_area("LLM Input", llm_in or "(대기중)", height=200, disabled=True)
    except Exception as e:
        st.text_area("LLM Input", f"요청 실패: {e}", height=200, disabled=True)

    # LLM Result
    st.caption("LLM 분석 결과")
    try:
        result_text = backend_result(run_id)
        st.text_area("LLM Result", value=result_text, height=200, disabled=True, key="llm_result_text")
    except Exception as e:
        st.text_area("LLM Result", value=f"결과 요청 실패: {e}", height=200, disabled=True)

def main():
    st.set_page_config(page_title = "STT+LLM 실시간 파이프라인", layout = "wide")
    st.title("STT+LLM 실시간 파이프라인(데모)")

    setup_session_state()

    fw_model, device, compute_type, chunk_sec, simulate_realtime, auto_refresh, backend_url, ollama_url, gemma_model, sys_prompt = display_ui_sidebar()

    uploaded = st.file_uploader("오디오 파일 업로드(MP3/WAV/FLAC/OGG)", type = ["mp3", "wav", "flac", "ogg"]) 

    run_btn = st.button("실행", disabled=st.session_state.get('running', False))
    stop_btn = st.button("중지", disabled=not st.session_state.get('running', False))

    if run_btn and uploaded and not st.session_state.get('running', False):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded.getvalue())
            st.session_state['audio_path'] = tmp_audio.name

        st.session_state['running'] = True
        st.session_state['stopped'] = False

        st.info(f"업로드 저장: {st.session_state['audio_path']}")

        # 백엔드 시작 호출
        try:
            run_id = start_backend_run(
                file_path=st.session_state['audio_path'],
                fw_model=fw_model,
                device=device,
                compute_type=compute_type,
                chunk_sec=chunk_sec,
                simulate_realtime=simulate_realtime,
                ollama_url=ollama_url,
                gemma_model=gemma_model,
                sys_prompt=sys_prompt,
            )
            st.session_state['run_id'] = run_id
            st.success(f"처리를 시작했습니다. run_id={run_id}")
        except Exception as e:
            st.error(f"백엔드 시작 실패: {e}")
        st.rerun()

    if stop_btn and st.session_state.get('running', False):
        st.session_state['running'] = False
        st.session_state['stopped'] = True
        try:
            if st.session_state.get('run_id'):
                backend_stop(st.session_state['run_id'])
        except Exception as e:
            st.warning(f"백엔드 중지 요청 실패: {e}")
        st.warning("중지 요청: 백엔드가 현재 작업을 완료하면 멈춥니다. 새 파일을 처리하려면 페이지를 새로고침하세요.")
        st.rerun()

    # 로그/결과 표시 섹션
    if st.session_state.get('run_id'):
        display_results(st.session_state['run_id'])
        
        # 자동 갱신
        if st.session_state.get('running', False) and auto_refresh:
            time.sleep(0.5)
            st.rerun()
    
    # 처리 완료 후 running 상태 false로 변경
    if st.session_state.get('run_id'):
        try:
            status = backend_status(st.session_state['run_id'])
            if any(k in status for k in ["처리 완료", "중지", "에러"]):
                st.session_state['running'] = False
        except Exception:
            pass

    if stop_btn:
        st.session_state['running'] = False
        st.warning("중지 요청: 새로고침(F5)으로 세션을 재시작하세요.")


if __name__ == "__main__":
    main()

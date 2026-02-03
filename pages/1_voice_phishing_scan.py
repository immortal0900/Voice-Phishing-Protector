import streamlit as st
import streamlit.components.v1 as components
import requests
import time
import json


st.set_page_config(page_title="보이스 피싱 검사", layout="wide")

st.title("보이스 피싱 검사")

# --- 세션 상태 초기화 ---
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = True
if 'status_data' not in st.session_state:
    st.session_state.status_data = {}
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
# compatibility keys referenced by older UI code
if 'stt_result' not in st.session_state:
    st.session_state.stt_result = ''
if 'llm_result' not in st.session_state:
    st.session_state.llm_result = ''
if 'use_blocking_poller' not in st.session_state:
    st.session_state.use_blocking_poller = False
if 'local_run_id' not in st.session_state:
    st.session_state.local_run_id = None


# 소형 유틸: 스크롤 가능한 텍스트 박스 렌더링
# Helper: escape for HTML
def _html_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>'))

def render_scrollable_box_html(key: str, text: str, height: int = 200):
    # Use localStorage to preserve scrollTop per run_id and key
    run_id = st.session_state.get('run_id') or 'no-run'
    safe = _html_escape(text)
    # Debug marker: data attribute added so we can detect whether the rendered DOM originates from this app
    html = f"""
        <!-- APP_DEBUG_MARKER: VOICE_PF_MARKER_20250924 -->
        <div id='box' data-app-debug='voice_pf_marker' style='height:{height}px; overflow:auto; white-space:pre-wrap; font-family: monospace; border:1px solid #eee; padding:8px; color:#fff; background:inherit; text-align:left;'>
        {safe}
        </div>
    <script>
    const key = 'scroll_' + {repr(run_id)} + '_' + {repr(key)};
    const box = document.getElementById('box');
    try {{
      const saved = localStorage.getItem(key);
      if (saved) {{ box.scrollTop = parseInt(saved); }}
      box.addEventListener('scroll', () => {{ localStorage.setItem(key, box.scrollTop); }});
    }} catch(e) {{ console.log(e); }}
    </script>
    """
    components.html(html, height=height + 20)
def poll_backend_status(run_id: str, backend: str = "http://127.0.0.1:8000", poll_interval: float = 1.0, max_tries: int = 600):
    """블로킹 폴링: 백엔드 `/status/{run_id}` 를 주기적으로 호출하여 얻은 결과를
    `st.session_state.status_data` 에 저장합니다. 이 함수는 호출자(페이지)의 실행을
    블록하며, UI에서 진행 상황을 실시간으로 보여주기 위해 사용합니다.
    """
    tries = 0
    while tries < max_tries:
        tries += 1
        try:
            r = requests.get(f"{backend}/status/{run_id}", params={"_ts": time.time()}, timeout=6)
            r.raise_for_status()
            data = r.json()
            st.session_state.status_data = data
            # 상태가 최종 완료(`all_complete`)이거나 에러면 종료
            if data.get('status') in ('all_complete', 'error'):
                st.session_state.analysis_complete = True
                st.session_state.analysis_running = False
                return
        except requests.exceptions.RequestException as e:
            # 네트워크/타임아웃 등은 재시도 — UI에는 간단히 로그 표시
            st.session_state.status_data = {'status': 'poll_error', 'error': str(e)}
        time.sleep(poll_interval)
    # 최대 시도 횟수 초과
    st.session_state.analysis_complete = True
    st.session_state.analysis_running = False


# --- 메인 UI ---
uploaded_file = st.file_uploader(
    "분석할 오디오 파일을 업로드하세요 (MP3, WAV, FLAC, OGG)",
    type=["mp3", "wav", "flac", "ogg"],
)

audio_placeholder = st.empty()
action_placeholder = st.empty()
results_placeholder = st.empty()


if uploaded_file is not None:
    audio_placeholder.audio(uploaded_file, format=uploaded_file.type)

    with action_placeholder:
        if st.button("분석 시작", disabled=st.session_state.get("analysis_running", False)):
            st.session_state.analysis_complete = False
            st.session_state.analysis_running = True
            results_placeholder.empty()
            with st.spinner("백엔드에 파일을 업로드하고 STT/분석을 시작합니다..."):
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    resp = requests.post("http://127.0.0.1:8000/analyze", files=files, timeout=30)
                    resp.raise_for_status()
                    run_id = resp.json().get('run_id')
                    st.session_state.run_id = run_id
                except requests.exceptions.RequestException as e:
                    st.error(f"백엔드 요청 실패: {e}")
                    st.session_state.analysis_running = False
                    st.session_state.analysis_complete = True
                    run_id = None

            # 클라이언트(JS) 폴러가 `/status/{run_id}`를 계속 조회하도록
            # run_id 및 상태 플래그만 설정합니다. 실제 폴링은 브라우저에서
            # 삽입된 JS 컴포넌트가 수행합니다.
            if run_id:
                st.session_state.run_id = run_id
                st.session_state.analysis_complete = False
                st.session_state.analysis_running = True

# 결과 렌더링 영역: 실시간 분석(폴링) 중에는 중복 표시를 피하기 위해 숨깁니다
if not (st.session_state.run_id and not st.session_state.analysis_complete):
    with results_placeholder.container():
        st.markdown("**음성 텍스트 변환 (STT)**")
        stt_text = st.session_state.status_data.get('stt_result', '') if st.session_state.status_data else ''
        render_scrollable_box_html('stt', stt_text, height=200)

        st.markdown("**서버 분석 결과 (PLM / LLM / 앙상블)**")
        # Show both LLM-reported score and heuristic LLM score (PLM-on-LLM) if available
        pretty = json.dumps({
            'status': st.session_state.status_data.get('status'),
            'PLM_risk_score': st.session_state.status_data.get('PLM_risk_score'),
            'LLM_reported_score': st.session_state.status_data.get('LLM_reported_score') if st.session_state.status_data.get('LLM_reported_score') is not None else st.session_state.status_data.get('LLM_risk_score'),
            'LLM_risk_score': st.session_state.status_data.get('LLM_risk_score'),
            'comprehensive_risk_score': st.session_state.status_data.get('comprehensive_risk_score'),
            'reasoning': st.session_state.status_data.get('reasoning'),
        }, ensure_ascii=False, indent=2)
        render_scrollable_box_html('analysis', pretty, height=200)

        # 상태 표시
        # st.info(f"Run ID: {st.session_state.run_id}") if st.session_state.run_id else None
        # st.write(f"현재 상태: {st.session_state.status_data.get('status')}")

        # 수동 초기화 버튼: 사용자 요청 시만 결과를 지움
        if st.button("초기화 (결과 지우기)"):
            st.session_state.run_id = None
            st.session_state.analysis_complete = True
            st.session_state.stt_result = ""
            st.session_state.llm_result = ""
            st.session_state.use_blocking_poller = False
            st.session_state.analysis_running = False
            st.session_state.local_run_id = None

 



# If a run is in progress, first try a quick server-side status check; if not complete, render client-side poller
if st.session_state.run_id and not st.session_state.analysis_complete:
    run_id = st.session_state.run_id
    backend = "http://127.0.0.1:8000"

    # Quick server-side check: if already complete, persist to session_state automatically
    try:
        r = requests.get(
            f"{backend}/status/{run_id}",
            params={"_ts": time.time()},
            headers={"Cache-Control": "no-cache"},
            timeout=3,
        )
        r.raise_for_status()
        data = r.json()
        if data.get('status') in ('all_complete', 'error'):
            st.session_state.stt_result = data.get('stt_result', '')
            st.session_state.llm_result = data.get('llm_result', '')
            st.session_state.analysis_complete = True
            # Do NOT call st.experimental_rerun() or otherwise reload the page.
            # Results should remain visible until the user manually clicks the 초기화 button.
    except Exception:
        # network/timeouts ignored here; fall back to client poller
        pass

    # If still not complete, render a client-side JS poller (embedded HTML) so the browser issues repeated GETs
    if not st.session_state.analysis_complete:
        html_template = """
        <!-- APP_DEBUG_MARKER: VOICE_PF_POLLER_20250924 -->
        <div id='voice_pf_debug_banner' style='font-family: monospace; color:#fff; padding:6px; background:#222; border-radius:6px; margin-bottom:8px;'>APP DEBUG BANNER: voice_pf_debug_banner</div>
        <div style='font-family: monospace; color:#fff;'>
            <h3 style='color:#fff;'>음성 텍스트 변환</h3>
            <div id='stt_box' style='white-space:pre-wrap; background:#000; color:#fff; padding:8px; border-radius:6px; min-height:160px;'>대기 중...</div>
            <h3 style='color:#fff; margin-top:12px;'>보이스피싱 분석</h3>
            <div id='llm_box' style='white-space:pre-wrap; background:#000; color:#fff; padding:8px; border-radius:6px; min-height:160px;'>대기 중...</div>
            <div id='poll_status' style='color: #bbb; font-size:12px; margin-top:6px;'>폴링 대기...</div>
        </div>
        <script>
            (function(){
            const runId = '__RUN_ID__';
            const backend = '__BACKEND__';
            const POLL_INTERVAL_MS = 1000;
            const MAX_BACKOFF_MS = 30_000;
            function cacheBustUrl(url){ return url + '?_ts=' + Date.now(); }

            // Prevent duplicate pollers if Streamlit re-inserts the component by
            // using localStorage as a cross-iframe persistent lock. This survives
            // re-insertion/reload of the iframe where window globals would be lost.
            const storageKey = 'voice_pf_poller_' + runId;
            try {
                const raw = localStorage.getItem(storageKey);
                let isRunning = false;
                if (raw) {
                    try {
                        const parsed = JSON.parse(raw);
                        if (parsed && parsed.state === 'running' && parsed.ts) {
                            const age = Date.now() - parsed.ts;
                            // consider the lock stale after 30 seconds
                            if (age < 30000) {
                                isRunning = true;
                            } else {
                                // stale lock -> clear it
                                try { localStorage.removeItem(storageKey); } catch(e){}
                            }
                        }
                    } catch(e) {
                        // non-JSON value: if it's 'running' treat as old and clear
                        if (raw === 'running') {
                            try { localStorage.removeItem(storageKey); } catch(e){}
                        }
                    }
                }
                if (isRunning) {
                    console.debug('poller already running (localStorage) for', runId);
                    return;
                }
                // set running state with timestamp and start heartbeat
                const heartbeat = { state: 'running', ts: Date.now() };
                localStorage.setItem(storageKey, JSON.stringify(heartbeat));
                let hbInterval = setInterval(()=>{
                    try { const cur = JSON.parse(localStorage.getItem(storageKey) || '{}'); cur.ts = Date.now(); localStorage.setItem(storageKey, JSON.stringify(cur)); } catch(e){}
                }, 5000);
                // ensure cleanup on page unload
                window.addEventListener('beforeunload', function(){ try{ clearInterval(hbInterval); localStorage.removeItem(storageKey); } catch(e){} });
            } catch(e) {
                console.warn('localStorage unavailable, falling back to window token', e);
                const token = '__VOICE_POLL_' + runId + '__';
                if(window[token]){
                    console.debug('poller already running for', runId);
                    return;
                }
                window[token] = true;
            }

            const sttBox = document.getElementById('stt_box');
            const llmBox = document.getElementById('llm_box');
            const pollStatus = document.getElementById('poll_status');

            let backoff = 0;
            let stopped = false;

            async function fetchOnce(){
                try{
                    // Short polling: do not request server-side blocking; this makes
                    // the request return immediately so the loop issues frequent GETs.
                    const url = backend + '/status/' + runId + '?_ts=' + Date.now();
                    const r = await fetch(url, {cache: 'no-store'});
                    if(!r.ok){
                        console.error('status fetch failed', r.status);
                        return { error: true, statusCode: r.status };
                    }
                    const j = await r.json();
                    return { error: false, data: j };
                } catch(e){
                    console.error('poll fetch exception', e);
                    return { error: true, exception: String(e) };
                }
            }

            async function onTick(){
                if(stopped) return;
                const res = await fetchOnce();
                if(res.error){
                    // Keep trying repeatedly; show a simple error and retry after a short delay
                    pollStatus.textContent = `네트워크/서버 오류. 재시도 중... (${res.exception || res.statusCode})`;
                    await new Promise(r=>setTimeout(r, 2000));
                    return; // next interval tick will retry
                }

                backoff = 0;
                const j = res.data;
                pollStatus.textContent = `상태: ${j.status || 'unknown'} | 업데이트: ${new Date().toLocaleTimeString()}`;
                if(sttBox) sttBox.textContent = j.stt_result || '';
                if(llmBox){
                    // Prefer the structured status response when available
                    const hasStructured = (j.PLM_risk_score !== undefined) || (j.comprehensive_risk_score !== undefined) || (j.LLM_reported_score !== undefined) || (j.LLM_risk_score !== undefined) || (j.LLM_risk_score !== undefined);
                    if(hasStructured){
                        llmBox.textContent = JSON.stringify({
                            'PLM_risk_score': j.PLM_risk_score,
                            'LLM_reported_score': (j.LLM_reported_score !== undefined ? j.LLM_reported_score : j.LLM_risk_score),
                            'LLM_risk_score': j.LLM_risk_score,
                            'comprehensive_risk_score': j.comprehensive_risk_score,
                            'reasoning': j.reasoning,
                        }, null, 2);
                    } else {
                        llmBox.textContent = j.llm_result || '';
                    }
                }

                if(j.status === 'all_complete' || j.status === 'error'){
                    pollStatus.textContent = `완료: ${j.status}`;
                    stopped = true;
                    try{ localStorage.removeItem(storageKey); } catch(e){}
                    return;
                }
            }

            // Start a sequential long-poll loop that waits for each request to finish
            (async function(){
                while(!stopped){
                    await onTick();
                    if(stopped) break;
                    // Short pause before next long-poll to avoid immediate tight reloop
                    await new Promise(r => setTimeout(r, 200));
                }
            })();

        })();
        </script>
        """
        html_template = html_template.replace('__RUN_ID__', run_id).replace('__BACKEND__', backend)
        components.html(html_template, height=480)

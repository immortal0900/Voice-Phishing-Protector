import shutil
import tempfile
import threading
import time
import uuid
import os
import glob
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import random
import json
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Models and external inference orchestrator are intentionally disabled in this deployment.
# Avoid loading large models or importing external inference modules when the API is
# intended to be used from `./pages` only.
inference_module = None

# Prefer server-side models module under `models/`. Do not import model code from `pages/`.
try:
    from models import local_models as local_models
except Exception:
    local_models = None

# uvicorn main:app --host 0.0.0.0 --port 8000

# Lazy STT model (to avoid loading per-run)
STT_MODEL = None
def get_stt_model():
    """Return a cached FasterWhisperSTT instance or None if unavailable."""
    global STT_MODEL
    if STT_MODEL is None:
        try:
            from rt_pipeline import FasterWhisperSTT
            STT_MODEL = FasterWhisperSTT(model="small")
        except Exception as e:
            print(f"[WARN] could not initialize STT model: {e}")
            STT_MODEL = None
    return STT_MODEL

app = FastAPI(title="AI 분석 백엔드")

# Allow Streamlit (local) to fetch status via browser JS
# In development it's often useful to allow requests from other hosts (e.g. when
# Streamlit is accessed via a LAN IP). For simplicity we allow all origins here.
# NOTE: For production, restrict allow_origins to the specific trusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are disabled for this simplified deployment.
MODELS_LOADED = False
MODEL_LOAD_ERROR = "models disabled in this configuration"

# No system prompt file is read in this deployment to avoid accessing files
# outside of the `./pages` directory.
DEFAULT_SYS_PROMPT = None

# --- 데이터 관리 ---
RUNS = {}
MAX_RUNS = 100  # 최대 저장 run 개수

class Status(BaseModel):
    status: str
    stt_result: str = ""
    llm_result: str = ""
    # Expose scoring fields so clients (Streamlit) can render PLM/LLM scores
    PLM_risk_score: float | None = None
    LLM_risk_score: float | None = None
    LLM_reported_score: float | None = None
    comprehensive_risk_score: float | None = None
    reasoning: str | None = None
    key_evidence: List[str] | None = None
    llm_text: str | None = None


class Face(BaseModel):
    id: int
    label: str | None = None
    confidence: float
    bbox: dict


class FaceRecognitionResponse(BaseModel):
    image_name: str | None = None
    size: int | None = None
    faces: List[Face] = []
    message: str = ""


@app.post('/face_recognize', response_model=FaceRecognitionResponse)
async def face_recognize(file: UploadFile | None = File(None), image_b64: str | None = Form(None)):
    """
    더미 얼굴 인식 엔드포인트 (테스트용)
    - multipart/form-data로 `file` 업로드 또는 `image_b64` 폼 필드로 base64 이미지를 보내면 동작합니다.
    - 실제 얼굴 인식은 수행하지 않으며, 테스트용 더미 JSON을 반환합니다.
    """
    if file is None and not image_b64:
        raise HTTPException(status_code=400, detail="file 또는 image_b64 중 하나를 제공하세요")

    image_name = getattr(file, 'filename', None) if file is not None else 'image_b64'
    size = None
    if file is not None:
        try:
            # 파일 크기 확인 (스트림 위치를 건드리지 않음)
            cur = None
            try:
                cur = file.file.tell()
            except Exception:
                cur = None
            try:
                file.file.seek(0, os.SEEK_END)
                size = file.file.tell()
            except Exception:
                size = None
            try:
                if cur is not None:
                    file.file.seek(cur)
            except Exception:
                pass
        except Exception:
            size = None

    # 더미 얼굴 데이터 생성: 0~3개 얼굴
    faces = []
    num_faces = random.randint(0, 3)
    for i in range(num_faces):
        f = {
            'id': i + 1,
            'label': random.choice([None, 'unknown', 'person_A', 'person_B']),
            'confidence': round(random.uniform(0.6, 0.99), 3),
            'bbox': {
                'x': random.randint(10, 300),
                'y': random.randint(10, 300),
                'w': random.randint(40, 150),
                'h': random.randint(40, 150),
            }
        }
        faces.append(f)

    return {
        'image_name': image_name,
        'size': size,
        'faces': faces,
        'message': 'This is a dummy face recognition result for testing.'
    }

def analysis_pipeline(audio_path: str, run_id: str, system_prompt: str | None = None):
    """백그라운드에서 실행될 분석 파이프라인

    동작:
    - 오디오 파일을 5초 단위로 청크로 분리
    - 각 청크를 즉시 STT(transcribe)하고 `RUNS[run_id]['stt_result']`를 누적 업데이트
    - 6개 청크(=30초)가 축적될 때마다 `pages.local_models`의 비동기 분석(start_local_async_analysis)을 호출
    - 청크 기반 처리 실패 시 전체 파일을 폴백으로 전사
    """
    try:
        RUNS[run_id]["status"] = "processing_stt"

        # attempt to import STT implementation
        try:
            from rt_pipeline import FasterWhisperSTT
        except Exception:
            FasterWhisperSTT = None

        # prepare STT model lazily
        stt_model = None
        if FasterWhisperSTT is not None:
            try:
                stt_model = FasterWhisperSTT(model="small")
            except Exception as e:
                print(f"[WARN] failed to init STT model: {e}")
                stt_model = None

        chunk_sec = 5
        incremental_text = ""
        chunk_counter = 0
        ran_local_analysis = False

        # Try chunked processing using soundfile. If soundfile/underlying mpg123
        # fails to decode an MP3, fall back to converting the file to WAV via
        # ffmpeg and retry. This avoids sporadic 'dequantization failed' errors
        # coming from libmpg123.
        def convert_to_wav(src_path: str) -> str:
            """Convert input audio to a WAV file using ffmpeg. Returns path to wav.

            If ffmpeg is not available or conversion fails, raises Exception.
            """
            import subprocess, tempfile
            base = os.path.basename(src_path)
            name, _ = os.path.splitext(base)
            out_path = os.path.join(tempfile.gettempdir(), f"{name}_conv.wav")
            cmd = [
                'ffmpeg', '-y', '-i', src_path,
                '-ar', '16000', '-ac', '1', out_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return out_path
            except Exception as e:
                raise

        try:
            import soundfile as sf
            try:
                sf_obj = sf.SoundFile(audio_path)
            except Exception:
                # conversion fallback
                try:
                    conv = convert_to_wav(audio_path)
                    sf_obj = sf.SoundFile(conv)
                    # replace audio_path so cleanup below can remove conv if needed
                    audio_path = conv
                except Exception as e:
                    raise
            sr = sf_obj.samplerate
            frames_per_chunk = int(sr * chunk_sec)
            n = 1
            tmp_dir = tempfile.mkdtemp(prefix = f"stt_chunks_{run_id}_")
            chunk_files = []

            try:
                while True:
                    frames = sf_obj.read(frames_per_chunk, always_2d=True)
                    if frames is None or getattr(frames, "size", 0) == 0:
                        break

                    out_path = os.path.join(tmp_dir, f"chunk_{n:04d}.wav")
                    try:
                        sf.write(out_path, frames, sr)
                    except Exception as e:
                        print(f"[WARN] failed to write chunk {out_path}: {e}")
                        n += 1
                        continue

                    # Transcribe this chunk if STT available
                    seg_text = ""
                    if stt_model is not None:
                        try:
                            seg_text = stt_model.transcribe(out_path)
                        except Exception as e:
                            print(f"[WARN] chunk transcribe failed for {out_path}: {e}")
                            seg_text = ""

                    if seg_text:
                        incremental_text = (incremental_text + " " + seg_text).strip()
                        RUNS[run_id]["stt_result"] = incremental_text

                    chunk_files.append(out_path)
                    chunk_counter += 1
                    n += 1

                    # Simulate real-time chunk arrival: wait for the chunk duration
                    try:
                        time.sleep(chunk_sec)
                    except Exception:
                        pass

                    # Every 6 chunks (30s), trigger PLM/LLM analysis via pages.local_models
                    if (chunk_counter % 6) == 0:
                        # start local async analysis if available
                        if local_models is not None:
                            try:
                                RUNS[run_id]["status"] = "processing_llm"
                                local_models.start_local_async_analysis(incremental_text, run_id)
                                ran_local_analysis = True
                            except Exception as e:
                                print(f"[WARN] failed to start local analysis at chunk {chunk_counter}: {e}")
                            finally:
                                # if more chunks remain we'll go back to STT processing
                                RUNS[run_id]["status"] = "processing_stt"

                # after all chunks processed
                full_text = incremental_text

                # if we finished and the last batch wasn't handled (chunk_counter % 6 != 0), run final analysis
                if (chunk_counter % 6) != 0 and local_models is not None:
                    try:
                        RUNS[run_id]["status"] = "processing_llm"
                        local_models.start_local_async_analysis(full_text, run_id)
                        ran_local_analysis = True
                    except Exception as e:
                        print(f"[WARN] final local analysis failed: {e}")
                    finally:
                        RUNS[run_id]["status"] = "processing_llm"

                # Mark that STT finished processing all chunks
                try:
                    RUNS[run_id]["stt_done"] = True
                except Exception:
                    pass

            finally:
                try:
                    sf_obj.close()
                except Exception:
                    pass

                # cleanup tmp_dir chunk files
                try:
                    for p in chunk_files:
                        try:
                            if os.path.exists(p):
                                os.unlink(p)
                        except Exception:
                            pass
                    try:
                        if os.path.isdir(tmp_dir):
                            os.rmdir(tmp_dir)
                    except Exception:
                        pass
                except Exception:
                    pass

        except Exception as chunk_e:
            # Fallback: transcribe the full file if chunked processing fails
            print(f"[WARN] chunked STT failed, falling back to full-file transcribe: {chunk_e}")
            try:
                if stt_model is not None:
                    # If STT fails on the original mp3, attempt conversion to wav first
                    try:
                        full_text = stt_model.transcribe(audio_path)
                    except Exception:
                        try:
                            conv = convert_to_wav(audio_path)
                            full_text = stt_model.transcribe(conv)
                            # ensure we remove conv later
                            audio_path = conv
                        except Exception as e:
                            print(f"[WARN] full-file transcribe after conversion also failed: {e}")
                            full_text = ""
                else:
                    full_text = ""
            except Exception as e:
                print(f"[ERROR] full-file transcribe failed: {e}")
                full_text = ""

            if full_text is None:
                full_text = ""
            RUNS[run_id]["stt_result"] = full_text
            chunk_counter = 0
            # Trigger analysis once with full_text
            if local_models is not None:
                try:
                    RUNS[run_id]["status"] = "processing_llm"
                    local_models.start_local_async_analysis(full_text, run_id)
                    ran_local_analysis = True
                except Exception as e:
                    print(f"[WARN] local analysis failed on full-file fallback: {e}")
            # In all cases mark STT as done for this run (we attempted full-file transcribe)
            try:
                RUNS[run_id]["stt_done"] = True
            except Exception:
                pass

        # Decide final RUNS status: if we started local analysis, leave status so that /status can merge LOCAL_ASYNC
        if ran_local_analysis:
            # actual completion will be reflected by merging LOCAL_ASYNC in get_status
            RUNS[run_id]["status"] = "processing_llm"
        else:
            RUNS[run_id]["status"] = "all_complete"
            try:
                RUNS[run_id]["stt_done"] = True
            except Exception:
                pass

    except Exception as e:
        RUNS[run_id]["status"] = "error"
        RUNS[run_id]["llm_result"] = f"오류 발생: {e}"
        print(f"[ERROR] run {run_id} pipeline failed: {e}")

@app.get("/")
def read_root():
    return {"message": "AI 분석 백엔드가 실행 중합니다. (모델 로드는 비활성화됨)"}


@app.post("/analyze")
async def start_analysis(file: UploadFile = File(...), sys_prompt: str | None = Form(None)):
    """
    분석을 시작하고 즉시 run_id를 반환합니다.
    """
    # 모델 로드는 비활성화된 구성입니다. 분석은 모델 없이 파일 저장 및 run_id 발급만 수행합니다.

    # 오래된 run 정리
    if len(RUNS) >= MAX_RUNS:
        oldest_run_id = next(iter(RUNS))
        del RUNS[oldest_run_id]

    # 저장 위치를 프로젝트의 `./pages` 디렉터리로 제한합니다.
    pages_dir = os.path.join(os.getcwd(), "pages")
    os.makedirs(pages_dir, exist_ok=True)
    # Clean up previous temporary audio files in `pages` to avoid accumulation.
    # Only remove common audio extensions to avoid touching page source files.
    try:
        audio_patterns = ("*.wav", "*.mp3", "*.flac", "*.ogg")
        for pat in audio_patterns:
            for p in glob.glob(os.path.join(pages_dir, pat)):
                try:
                    os.remove(p)
                    print(f"[INFO] removed old temp audio file: {p}")
                except Exception as e:
                    print(f"[WARN] failed to remove temp audio file {p}: {e}")
    except Exception:
        pass

    # Use run_id-based filename to save uploaded file in streamlit_app
    orig_name = getattr(file, "filename", None) or "uploaded"
    _, ext = os.path.splitext(orig_name)
    if not ext:
        ext = ".wav"
    # generate run_id and path for this request
    run_id = uuid.uuid4().hex
    tmp_audio_path = os.path.join(pages_dir, f"{run_id}{ext}")
    try:
        try:
            file.file.seek(0)
        except Exception:
            pass
        with open(tmp_audio_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
    except Exception as e:
        print(f"[ERROR] failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    try:
        size = os.path.getsize(tmp_audio_path)
    except Exception:
        size = None
    print(f"[INFO] received file saved to {tmp_audio_path} (orig={orig_name}, size={size})")

    # 상태 초기화
    RUNS[run_id] = {
        "status": "pending",
        "stt_result": "",
        "llm_result": "",
        "stt_done": False,
    }

    # Determine effective system prompt: request-provided > default file
    effective_sys_prompt = sys_prompt or DEFAULT_SYS_PROMPT

    # 백그라운드에서 파이프라인 실행 (모델 비활성화 상태이므로 간단히 상태를 갱신)
    thread = threading.Thread(target=analysis_pipeline, args=(tmp_audio_path, run_id, effective_sys_prompt))
    thread.start()

    return {"run_id": run_id}


if __name__ == '__main__':
    # Run the app with an external bind so other machines on the LAN can reach it.
    # Use environment variables API_HOST/API_PORT to override if desired.
    try:
        import uvicorn
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', '8000'))
        print(f"[INFO] starting uvicorn on {host}:{port}")
        uvicorn.run("main:app", host=host, port=port)
    except Exception as e:
        print(f"[WARN] failed to start uvicorn from __main__: {e}")


@app.get("/status/{run_id}", response_model=Status)
async def get_status(run_id: str, wait: int | None = None, timeout: int = 25):
    """
    지정된 run_id의 현재 상태와 결과를 반환합니다.
    """
    run_data = RUNS.get(run_id)
    # debug logging for diagnosis
    if run_data:
        stt_len = len(run_data.get('stt_result', '') or '')
        llm_len = len(run_data.get('llm_result', '') or '')
        print(f"[STATUS] GET {run_id} -> status={run_data.get('status')} stt_len={stt_len} llm_len={llm_len}")
    else:
        print(f"[STATUS] GET {run_id} -> not found")

    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    # Optionally support long-polling: if 'wait' is provided (non-null), block up to
    # `timeout` seconds until the run status becomes 'complete' or 'error'. This is
    # used by clients that prefer a single blocking request instead of frequent short polls.
    import time as _time
    start_ts = _time.time()
    def _should_wait():
        if wait is None:
            return False
        # wait > 0 indicates client wants to block until completion or timeout
        return True

    # If requested, loop until status becomes terminal or timeout reached
    while _should_wait():
        merged = dict(run_data)
        try:
            if local_models is not None:
                async_runs = getattr(local_models, 'LOCAL_ASYNC', {}) or {}
                if run_id in async_runs:
                    aentry = async_runs.get(run_id, {})
                    merged.update(aentry)
        except Exception:
            pass

        cur_status = merged.get('status')
        if cur_status in ('all_complete', 'error'):
            break
        if (_time.time() - start_ts) >= float(timeout):
            # timeout elapsed, break and return current merged status
            break
        # sleep briefly before re-checking
        _time.sleep(0.5)

    # Merge in local async inference status if present (by comparing text or run ids)
    merged = dict(run_data)
    try:
        # Prefer merging any async results from the server-side local_models module if available
        if local_models is not None:
            async_runs = getattr(local_models, 'LOCAL_ASYNC', {}) or {}
            if run_id in async_runs:
                aentry = async_runs.get(run_id, {})
                merged.update(aentry)
                # Only expose the human-readable LLM result when the async entry
                # is in terminal state 'complete' to avoid leaking partial JSON.
                if aentry.get('status') == 'all_complete':
                    llm_text = aentry.get('llm_text') or ''
                    reasoning = aentry.get('reasoning') or ''
                    pretty = ''
                    try:
                        if llm_text:
                            try:
                                parsed = json.loads(llm_text)
                            except Exception:
                                parsed = None
                            if isinstance(parsed, dict):
                                parts = []
                                comp = aentry.get('comprehensive_risk_score')
                                if comp is not None:
                                    parts.append(f"종합 점수: {float(comp):.3f}")
                                plm = aentry.get('PLM_risk_score')
                                if plm is not None:
                                    parts.append(f"PLM 점수: {float(plm):.3f}")
                                llm_sc = aentry.get('LLM_risk_score')
                                if llm_sc is not None:
                                    parts.append(f"LLM 점수: {float(llm_sc):.3f}")
                                parsed_reason = parsed.get('reasoning') if isinstance(parsed, dict) else None
                                if parsed_reason:
                                    parts.append(f"설명: {parsed_reason}")
                                elif reasoning:
                                    parts.append(f"설명: {reasoning}")
                                ev = parsed.get('key_evidence') if isinstance(parsed, dict) else None
                                if ev:
                                    parts.append("증거: " + ", ".join(map(str, ev)))
                                if not parts:
                                    pretty = json.dumps(parsed, ensure_ascii=False)
                                else:
                                    pretty = "\n".join(parts)
                            else:
                                pretty = llm_text
                        else:
                            pretty = reasoning or json.dumps(aentry, ensure_ascii=False)
                    except Exception:
                        pretty = llm_text or reasoning or json.dumps(aentry, ensure_ascii=False)

                    merged['llm_result'] = pretty
            else:
                stext = run_data.get('stt_result', '') or ''
                for aid, aentry in async_runs.items():
                    if aentry.get('PLM_risk_score') is not None and (sentry := aentry.get('conversation_text')):
                        if sentry.strip() and stext.strip() and (sentry.strip()[:200] == stext.strip()[:200]):
                            merged.update(aentry)
                            # Map async analysis output into the llm_result field for clients (formatted)
                            llm_text = aentry.get('llm_text') or ''
                            reasoning = aentry.get('reasoning') or ''
                            try:
                                parsed = json.loads(llm_text) if llm_text else None
                            except Exception:
                                parsed = None
                            if isinstance(parsed, dict):
                                parts = []
                                comp = aentry.get('comprehensive_risk_score')
                                if comp is not None:
                                    parts.append(f"종합 점수: {float(comp):.3f}")
                                if aentry.get('reasoning'):
                                    parts.append(f"설명: {aentry.get('reasoning')}")
                                if parsed.get('key_evidence'):
                                    parts.append("증거: " + ", ".join(map(str, parsed.get('key_evidence'))))
                                merged['llm_result'] = "\n".join(parts) if parts else json.dumps(parsed, ensure_ascii=False)
                            else:
                                merged['llm_result'] = llm_text or reasoning or json.dumps(aentry, ensure_ascii=False)
                            break
    except Exception:
        pass

    # If we obtained an llm_result from async runs, copy it back into RUNS so
    # subsequent debug logs reflect the LLM output as well.
    try:
        if merged.get('llm_result'):
            RUNS[run_id]['llm_result'] = merged.get('llm_result')
        # Also propagate status updates (e.g., processing_plm/complete)
        if merged.get('status'):
            try:
                new_status = merged.get('status')
                # Only allow final terminal token 'all_complete' to overwrite RUNS status if STT has finished
                if new_status == 'all_complete':
                    if RUNS.get(run_id, {}).get('stt_done'):
                        RUNS[run_id]['status'] = new_status
                    else:
                        # keep existing RUNS status (likely still processing_stt)
                        pass
                else:
                    RUNS[run_id]['status'] = new_status
            except Exception:
                pass
    except Exception:
        pass

    return merged


@app.post('/predict_text')
async def predict_text(run_id: str = Form(...), text: str = Form(...)):
    """텍스트를 받아 pages.local_models를 이용해 PLM/LLM 분석을 시작하고 즉시 run_id를 반환합니다.
    클라이언트(예: Streamlit)는 /status/{run_id}로 결과를 조회할 수 있습니다.
    """
    if local_models is None:
        raise HTTPException(status_code=503, detail="Local models module not available")

    # Ensure PLM is loaded (best-effort)
    try:
        try:
            local_models.load_local_plm()
        except Exception:
            pass
        local_models.start_local_async_analysis(text, run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start local analysis: {e}")

    return {"run_id": run_id}


"""간단 실시간 파이프라인 (요구 조건 간소화 버전)

기능:
1) 오디오 입력을 5초 청크로 순차 저장(in_dir)
2) 7-청크(=35초) 슬라이딩 윈도우로 faster-whisper STT
3) 매 7번째 청크마다 누적 텍스트 중복 제거 후 Gemma3(ollama)로 분류 요청
4) LLM 판단을 콘솔에 즉시 출력

가정/제약:
- 입력 오디오는 16kHz mono PCM .wav 파일들이 순차적으로 in_dir에 떨어진다고 가정(예: 녹음기에서 5초 단위 저장)
- 복잡한 예외처리/재시도/로깅 생략
"""

# 예시 실행(단일 실행으로 피더+파이프라인 병렬 수행)
# python .\rt_pipeline.py --input-audio "fss_voice_files\NR5923_마스킹 완료(대환대출 사기)_.mp3" --in-dir ".\live_chunks" --chunk-sec 5 --simulate-realtime --start-clean --exit-when-done --fw-model small --device cuda --compute-type float16 --ollama-url http://127.0.0.1:11434 --gemma-model gemma2:9b --sys-prompt .\system_ko.txt

from __future__ import annotations

import os
import time
import json
import glob
import subprocess
import threading
import tempfile
import traceback
from typing import List, Deque, Optional
from collections import deque


def list_chunks(in_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    return files


class FasterWhisperSTT:
    def __init__(self, model: str = "small", device: str = "cuda", compute_type: str = "float16"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device = device, compute_type = compute_type)

    def transcribe(self, path: str, lang: str = "ko") -> str:
        segs, _ = self.model.transcribe(path, language = lang, beam_size = 5, vad_filter = True)
        return " ".join((s.text or "").strip() for s in segs if (s.text or "").strip())


class Gemma3Classifier:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "gemma2:9b"):
        # Gemma3 가용 모델명은 환경에 따라 다름. LLM 서버 쪽에서 지원 모델명을 확인.
        import requests
        self.s = requests.Session()
        self.base = base_url
        self.model = model

    def classify(self, text: str, system_prompt: str | None = None) -> str:
        # '/api/generate' 엔드포인트는 단일 프롬프트 문자열을 사용합니다.
        prompt = []
        if system_prompt:
            prompt.append(system_prompt)
        
        prompt.append(f"아래 한국어 통화 전사를 보이스피싱 여부로 JSON만 출력: {text[:4000]}")
        
        full_prompt = "\n\n".join(prompt)

        r = self.s.post(f"{self.base}/api/generate", json = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False
        }, timeout = 120)
        r.raise_for_status()
        data = r.json()
        # '/api/generate'의 응답 형식은 'response' 필드에 결과가 들어 있습니다.
        msg = data.get("response", "").strip()
        return msg


def feeder_from_audio(
    input_audio: str,
    out_dir: str,
    chunk_sec: int = 5,
    simulate_realtime: bool = True,
    prefix: str = "chunk_",
    feeder_done: Optional[threading.Event] = None,
) -> None:
    """입력 오디오를 N초 단위 WAV 파일로 out_dir에 순차 저장.

    지원 형식:
    - WAV/FLAC/OGG: soundfile로 직접 로드(FFmpeg 불필요)
    - MP3 등: pydub(FFmpeg 필요)로 로드
    """
    os.makedirs(out_dir, exist_ok = True)
    ext = os.path.splitext(input_audio)[1].lower()
    use_soundfile = ext in {".wav", ".flac", ".ogg"}
    print(f"[FEEDER] input_audio={input_audio} ext={ext} use_soundfile={use_soundfile} out_dir={os.path.abspath(out_dir)} chunk_sec={chunk_sec}")

    temp_converted: str | None = None
    try:
        # If input isn't a directly supported soundfile format, try converting via ffmpeg
        if not use_soundfile:
            try:
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                # convert to mono 16k WAV which is commonly expected by STT
                cmd = [
                    "ffmpeg", "-y", "-i", input_audio, "-ar", str(16000), "-ac", str(1), temp_path
                ]
                subprocess.run(cmd, check = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                temp_converted = temp_path
                input_to_read = temp_converted
                print(f"[FEEDER] converted input to WAV via ffmpeg: {temp_converted}")
                # after conversion, treat it like a soundfile input
                import soundfile as sf
                data, sr = sf.read(input_to_read, always_2d = True)
                total_frames = data.shape[0]
                frames_per_chunk = int(sr * chunk_sec)
                idx = 0
                n = 1
                while idx < total_frames:
                    chunk = data[idx: idx + frames_per_chunk]
                    idx += frames_per_chunk
                    out_path = os.path.join(out_dir, f"{prefix}{n:04d}.wav")
                    sf.write(out_path, chunk, sr)
                    exists = os.path.exists(out_path)
                    print(f"[FEEDER] wrote chunk: {out_path} (sr={sr}, frames={chunk.shape[0]}) exists={exists}")
                    n += 1
                    if simulate_realtime:
                        time.sleep(chunk_sec)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] ffmpeg conversion failed: {e}")
                print(e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e))
                raise
            except Exception:
                # fallback: try pydub if ffmpeg or soundfile path fails
                try:
                    from pydub import AudioSegment
                    seg = AudioSegment.from_file(input_audio)
                    ms = len(seg)
                    step = chunk_sec * 1000
                    n = 1
                    for start in range(0, ms, step):
                        piece = seg[start: start + step]
                        out_path = os.path.join(out_dir, f"{prefix}{n:04d}.wav")
                        piece.export(out_path, format = "wav")
                        exists = os.path.exists(out_path)
                        print(f"[FEEDER] wrote chunk: {out_path} (ms_start={start}) exists={exists}")
                        n += 1
                        if simulate_realtime:
                            time.sleep(chunk_sec)
                except Exception as inner_e:
                    print(f"[ERROR] feeder failed during pydub fallback: {inner_e}")
                    raise
        else:
            import soundfile as sf
            data, sr = sf.read(input_audio, always_2d = True)
            total_frames = data.shape[0]
            frames_per_chunk = int(sr * chunk_sec)
            idx = 0
            n = 1
            while idx < total_frames:
                chunk = data[idx: idx + frames_per_chunk]
                idx += frames_per_chunk
                out_path = os.path.join(out_dir, f"{prefix}{n:04d}.wav")
                sf.write(out_path, chunk, sr)
                exists = os.path.exists(out_path)
                print(f"[FEEDER] wrote chunk: {out_path} (sr={sr}, frames={chunk.shape[0]}) exists={exists}")
                n += 1
                if simulate_realtime:
                    time.sleep(chunk_sec)
    except Exception as e:
        print(f"[ERROR] feeder failed: {e}")
        print(traceback.format_exc())
        print("[HINT] MP3 입력일 경우 FFmpeg 설치가 필요합니다. (pydub이 FFmpeg를 사용)")
    finally:
        if feeder_done is not None:
            feeder_done.set()
        if temp_converted is not None and os.path.exists(temp_converted):
            try:
                os.unlink(temp_converted)
            except Exception:
                pass


def simple_dedup(old: str, new: str) -> str:
    # 매우 단순한 중복 제거: 마지막 200자 기준으로 겹치는 부분 제거
    pivot = old[-200:] if len(old) > 200 else old
    if pivot and new.startswith(pivot):
        return old + new[len(pivot):]
    return (old + " " + new).strip()


def main():
    import argparse
    p = argparse.ArgumentParser(description = "5초 청크 → 7-청크 STT → Gemma3 분류")
    p.add_argument("--in-dir", default = "live_chunks", help = "5초 .wav 청크 폴더")
    p.add_argument("--input-audio", help = "한 번에 실행 시, 이 오디오를 5초 WAV 청크로 분할 투입(MP3는 ffmpeg 필요)")
    p.add_argument("--chunk-sec", type = int, default = 5, help = "청크 길이(초)")
    p.add_argument("--simulate-realtime", action = "store_true", help = "실제 통화처럼 청크 간 sleep 수행")
    p.add_argument("--start-clean", action = "store_true", help = "시작 시 in-dir 내 기존 .wav 청크 삭제")
    p.add_argument("--exit-when-done", action = "store_true", help = "피더 완료 및 모든 청크 처리 후 자동 종료")
    p.add_argument("--fw-model", default = "small", help = "faster-whisper 모델")
    p.add_argument("--device", default = "cuda", help = "cuda|cpu")
    p.add_argument("--compute-type", default = "float16", help = "float16|int8_float16 등")
    p.add_argument("--ollama-url", default = "http://127.0.0.1:11434", help = "Ollama 주소")
    p.add_argument("--gemma-model", default = "gemma2:9b", help = "Gemma3 호환 모델명")
    p.add_argument("--sys-prompt", help = "시스템 프롬프트 파일")
    args = p.parse_args()

    os.makedirs(args.in_dir, exist_ok = True)
    if args.start_clean:
        for pth in glob.glob(os.path.join(args.in_dir, "*.wav")):
            try:
                os.remove(pth)
            except Exception:
                pass

    stt = FasterWhisperSTT(model = args.fw_model, device = args.device, compute_type = args.compute_type)
    clf = Gemma3Classifier(base_url = args.ollama_url, model = args.gemma_model)
    sys_prompt = None
    if args.sys_prompt and os.path.isfile(args.sys_prompt):
        with open(args.sys_prompt, "r", encoding = "utf-8") as f:
            sys_prompt = f.read()

    feeder_thread: Optional[threading.Thread] = None
    feeder_done = threading.Event()
    if args.input_audio and os.path.isfile(args.input_audio):
        print(f"[INFO] feeder starting: '{args.input_audio}' -> '{args.in_dir}', chunk={args.chunk_sec}s, simulate={args.simulate_realtime}")
        feeder_thread = threading.Thread(
            target = feeder_from_audio,
            kwargs = dict(
                input_audio = args.input_audio,
                out_dir = args.in_dir,
                chunk_sec = args.chunk_sec,
                simulate_realtime = args.simulate_realtime,
                feeder_done = feeder_done,
            ),
            daemon = True,
        )
        feeder_thread.start()
    elif args.input_audio:
        print(f"[WARN] input-audio not found: '{args.input_audio}' — 피더를 시작하지 않습니다.")

    seen: set[str] = set()
    pending_texts: List[str] = []  # 들어온 청크별 전사 텍스트를 순서대로 보관
    next_start_index = 0            # 다음 7-청크 윈도우의 시작 인덱스(0-based). 0→6→12→...
    merged_text = ""

    print(f"[INFO] watching {args.in_dir} ({args.chunk_sec}초 .wav 청크)")
    try:
        while True:
            files = list_chunks(args.in_dir)
            for path in files:
                if path in seen:
                    continue
                # 간단 안정화 대기
                t0 = os.path.getmtime(path)
                time.sleep(0.2)
                if os.path.getmtime(path) != t0:
                    continue

                seen.add(path)
                print(f"[INFO] new chunk detected, processing: {path}")
                try:
                    text = stt.transcribe(path)
                    print(f"[INFO] transcribed {path}: {len(text)} chars")
                except Exception as e:
                    print(f"[ERROR] transcribe failed for {path}: {e}")
                    text = ""
                pending_texts.append(text)

                # 충분한 청크가 모였으면(현재까지 개수 >= next_start_index + 7),
                # 7-청크 윈도우를 6청크 간격으로 반복 처리
                while len(pending_texts) >= (next_start_index + 7):
                    start_idx = next_start_index
                    end_idx = next_start_index + 7
                    block = " ".join(pending_texts[start_idx:end_idx]).strip()
                    merged_text = simple_dedup(merged_text, block)

                    # 시간 범위 안내(초)
                    start_sec = start_idx * 5
                    end_sec = end_idx * 5
                    print(f"\n[INFO] LLM 호출 윈도우: {start_sec}s ~ {end_sec}s (청크 {start_idx+1}~{end_idx})")
                    resp = clf.classify(merged_text, system_prompt = sys_prompt)
                    print("[LLM 판단]\n", resp, "\n", sep = "")

                    # 다음 윈도우는 6청크 뒤에서 시작(5초 오버랩)
                    next_start_index += 6

            # 피더 완료 및 모든 청크를 처리했으면 종료(남은 부분이 7청크 미만이면 추가 호출 없음)
            if args.exit_when_done and (feeder_thread is not None) and feeder_done.is_set():
                all_seen = len(seen) == len(files)
                if all_seen and (len(pending_texts) < (next_start_index + 7)):
                    break

            time.sleep(0.3)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""다중 STT 배치 파이프라인

엔진 3종을 순차 실행하여 전사 결과를 저장하고 비교 CSV를 생성합니다.

- Faster-Whisper → 출력 폴더: fw_stt_voice_files
- whisper.cpp     → 출력 폴더: wcpp_stt_voice_files
- Wav2Vec2 (ko)   → 출력 폴더: w2v_stt_voice_files

공통 출력:
- 각 파일에 대해 .txt + .meta.json 저장
- compare_transcripts.csv (선택 엔진만 컬럼 생성)
- 실패 시 failure_*.log에 기록

기본 설정:
- 순차 실행, tqdm 한 줄
- 입력 폴더 비재귀 처리
- VAD/전처리는 엔진별 내부 처리 사용(필요 최소한)
- 장치: GPU 사용(가능 시), 없으면 CPU fallback

주의: whisper.cpp 실행 파일/모델 파일 경로가 없으면 해당 엔진은 건너뜁니다.
"""

from __future__ import annotations

import os
import sys
import csv
import io
import json
import time
import glob
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

from tqdm import tqdm
import torch  # Wav2Vec2에서 사용

# 출력 폴더(프로젝트 루트 기준)
FW_OUT = os.path.abspath("fw_stt_voice_files")
WCPP_OUT = os.path.abspath("wcpp_stt_voice_files")
W2V_OUT = os.path.abspath("w2v_stt_voice_files")

COMPARE_CSV = os.path.abspath("compare_transcripts.csv")
FAIL_FW = os.path.abspath("failure_fw.log")
FAIL_WCPP = os.path.abspath("failure_wcpp.log")
FAIL_W2V = os.path.abspath("failure_w2v.log")

# whisper.cpp 기본 경로(프로젝트 내부)
WHISPER_CPP_EXE = os.path.abspath(os.path.join("tools", "whisper_cpp", "bin", "whisper_cpp.exe"))
WHISPER_CPP_MODEL_DIR = os.path.abspath(os.path.join("models", "whisper"))


@dataclass
class SegmentResult:
    index: int
    start: float
    end: float
    text: str
    speaker: str = "speaker_1"
    confidence: Optional[float] = None


@dataclass
class FileTranscription:
    file: str
    rel_path: str
    language: str
    text: str
    duration_sec: float
    segments: List[SegmentResult]
    engine: str
    processed_at: float = time.time()
    error: Optional[str] = None

    def to_meta_dict(self):
        d = asdict(self)
        d["segments"] = [asdict(s) for s in self.segments]
        return d


def list_audio_files(input_dir: str) -> List[str]:
    supported = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
    files: List[str] = []
    for name in os.listdir(input_dir):
        p = os.path.join(input_dir, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in supported:
            files.append(p)
    files.sort()
    return files

def write_outputs(out_dir: str, ft: FileTranscription):
    os.makedirs(out_dir, exist_ok = True)
    base_no_ext = os.path.splitext(os.path.basename(ft.file))[0]
    txt_path = os.path.join(out_dir, f"{base_no_ext}.txt")
    meta_path = os.path.join(out_dir, f"{base_no_ext}.meta.json")
    with open(txt_path, "w", encoding = "utf-8") as f:
        f.write(ft.text)
    with open(meta_path, "w", encoding = "utf-8") as f:
        json.dump(ft.to_meta_dict(), f, ensure_ascii = False, indent = 2)


def append_failure(log_path: str, rel_path: str, err: Exception):
    with open(log_path, "a", encoding = "utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{rel_path}\t{err}\n")


# ----------------------- Faster-Whisper -----------------------
class FasterWhisperEngine:
    def __init__(self, model_name: str = "medium", device: str = "cuda", compute_type: str = "float16", language: str = "ko"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"faster-whisper 불러오기 실패: {e}")
        self._WhisperModel = WhisperModel
        self._model = None

    def ensure_model(self):
        if self._model is None:
            self._model = self._WhisperModel(self.model_name, device = self.device, compute_type = self.compute_type)
        return self._model

    def transcribe(self, file_path: str) -> FileTranscription:
        model = self.ensure_model()
        segments, info = model.transcribe(file_path, language = self.language, beam_size = 5)
        seg_list: List[SegmentResult] = []
        texts: List[str] = []
        for idx, s in enumerate(segments):
            t = (s.text or "").strip()
            texts.append(t)
            seg_list.append(
                SegmentResult(index = idx, start = float(getattr(s, "start", 0.0) or 0.0), end = float(getattr(s, "end", 0.0) or 0.0), text = t, confidence = getattr(s, "avg_logprob", None))
            )
        full_text = " ".join([t for t in texts if t])
        dur = float(getattr(info, "duration", 0.0) or 0.0)
        rel = os.path.basename(file_path)
        return FileTranscription(
            file = os.path.basename(file_path),
            rel_path = rel,
            language = self.language,
            text = full_text,
            duration_sec = dur,
            segments = seg_list,
            engine = f"faster-whisper-{self.model_name}-{self.compute_type}-{self.device}"
        )


# ----------------------- whisper.cpp -----------------------
class WhisperCppEngine:
    def __init__(self, exe_path: str = WHISPER_CPP_EXE, model_dir: str = WHISPER_CPP_MODEL_DIR, language: str = "ko"):
        self.exe_path = exe_path
        self.model_dir = model_dir
        self.language = language

    def _pick_model(self) -> Optional[str]:
        if not os.path.isdir(self.model_dir):
            return None
        # 우선순위: *.gguf > *.bin (선호하는 파일명을 먼저 매칭)
        patterns = ["*.gguf", "*.bin"]
        for pat in patterns:
            matches = glob.glob(os.path.join(self.model_dir, pat))
            if matches:
                # turbo가 있다면 우선 사용
                matches.sort(key = lambda p: ("turbo" not in os.path.basename(p).lower(), os.path.basename(p)))
                return matches[0]
        return None

    def is_available(self) -> bool:
        return os.path.isfile(self.exe_path) and self._pick_model() is not None

    def transcribe(self, file_path: str) -> FileTranscription:
        if not os.path.isfile(self.exe_path):
            raise FileNotFoundError(f"whisper.cpp 실행 파일을 찾을 수 없습니다: {self.exe_path}")
        model_path = self._pick_model()
        if not model_path:
            raise FileNotFoundError(f"whisper.cpp 모델 파일이 없습니다: {self.model_dir}")

        with tempfile.TemporaryDirectory() as tmpd:
            cmd = [
                self.exe_path,
                "-m", model_path,
                "-f", file_path,
                "-l", self.language,
                "-oj",
                "-otxt",
                "-ofolder", tmpd,
            ]
            # Windows에서 경로에 공백이 있을 수 있으니 shell=False 유지
            proc = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
            if proc.returncode != 0:
                raise RuntimeError(f"whisper.cpp 실패: rc={proc.returncode}, stderr={proc.stderr[:500]}")

            base = os.path.splitext(os.path.basename(file_path))[0]
            json_path = os.path.join(tmpd, f"{base}.json")
            txt_path = os.path.join(tmpd, f"{base}.txt")
            text = ""
            segs: List[SegmentResult] = []
            dur = 0.0
            if os.path.isfile(json_path):
                with open(json_path, "r", encoding = "utf-8") as f:
                    data = json.load(f)
                # whisper.cpp JSON 형식에 따라 파싱
                # 예상: {"segments": [{"start":0.0,"end":1.1,"text":"..."}, ...], "duration": ...}
                for idx, s in enumerate(data.get("segments", [])):
                    t = (s.get("text") or "").strip()
                    segs.append(SegmentResult(index = idx, start = float(s.get("start", 0.0)), end = float(s.get("end", 0.0)), text = t))
                text = " ".join([s.text for s in segs if s.text])
                dur = float(data.get("duration", 0.0))
            elif os.path.isfile(txt_path):
                with open(txt_path, "r", encoding = "utf-8") as f:
                    text = f.read().strip()
            else:
                # 산출물이 없으면 실패 처리
                raise RuntimeError("whisper.cpp 출력 파일을 찾을 수 없습니다")

        rel = os.path.basename(file_path)
        return FileTranscription(
            file = os.path.basename(file_path),
            rel_path = rel,
            language = self.language,
            text = text,
            duration_sec = dur,
            segments = segs,
            engine = "whisper.cpp"
        )


# ----------------------- Wav2Vec2 (Korean) -----------------------
class Wav2Vec2Engine:
    def __init__(self, model_id: str = "kresnik/wav2vec2-large-xlsr-korean", device: str = "cuda", chunk_sec: int = 30, overlap_sec: float = 1.0):
        self.model_id = model_id
        self.device = device
        self.chunk_sec = int(max(10, chunk_sec))
        self.overlap_sec = float(max(0.0, overlap_sec))
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
        except Exception as e:
            raise RuntimeError(f"transformers 불러오기 실패: {e}")
        self._Wav2Vec2ForCTC = Wav2Vec2ForCTC
        self._Wav2Vec2Processor = Wav2Vec2Processor
        self._processor = None
        self._model = None

        from pydub import AudioSegment  # 이미 프로젝트에 존재
        self._AudioSegment = AudioSegment

    def ensure_model(self):
        if self._processor is None:
            self._processor = self._Wav2Vec2Processor.from_pretrained(self.model_id)
        if self._model is None:
            self._model = self._Wav2Vec2ForCTC.from_pretrained(self.model_id)
            dev = self.device if self.device in {"cuda", "cpu"} else "cpu"
            self._model = self._model.to(dev)
        return self._processor, self._model

    @staticmethod
    def _to_float_list(seg) -> List[float]:
        # pydub AudioSegment → 16k mono float list [-1, 1]
        sample_width = seg.sample_width
        max_val = float(2 ** (8 * sample_width - 1))
        raw = seg.get_array_of_samples()
        return [s / max_val for s in raw]

    def transcribe(self, file_path: str) -> FileTranscription:
        proc, model = self.ensure_model()
        # 로드 및 16k mono
        audio = self._AudioSegment.from_file(file_path)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.channels != 1:
            audio = audio.set_channels(1)

        total_ms = len(audio)
        step_ms = int(self.chunk_sec * 1000)
        overlap_ms = int(self.overlap_sec * 1000)

        all_text: List[str] = []
        segs: List[SegmentResult] = []
        idx = 0
        t = 0
        while t < total_ms:
            start_ms = t
            end_ms = min(t + step_ms, total_ms)
            chunk = audio[start_ms:end_ms]
            floats = self._to_float_list(chunk)
            inputs = proc(floats, sampling_rate = 16000, return_tensors = "pt", padding = True)
            with torch.no_grad():  # type: ignore
                logits = model(inputs.input_values.to(model.device)).logits
            ids = logits.argmax(dim = -1)
            text = proc.batch_decode(ids)[0].strip()
            all_text.append(text)
            segs.append(SegmentResult(index = idx, start = start_ms / 1000.0, end = end_ms / 1000.0, text = text))
            idx += 1
            if end_ms >= total_ms:
                break
            t = end_ms - overlap_ms if overlap_ms > 0 else end_ms

        full_text = " ".join([t for t in all_text if t])
        rel = os.path.basename(file_path)
        return FileTranscription(
            file = os.path.basename(file_path),
            rel_path = rel,
            language = "ko",
            text = full_text,
            duration_sec = total_ms / 1000.0,
            segments = segs,
            engine = f"wav2vec2-{self.model_id}"
        )


def run_engine_over_files(engine_name: str, engine, input_dir: str, out_dir: str, failure_log: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok = True)
    files = list_audio_files(input_dir)
    results: Dict[str, str] = {}
    for f in tqdm(files, desc = f"{engine_name}", unit = "file"):
        rel = os.path.basename(f)
        try:
            ft = engine.transcribe(f)
            write_outputs(out_dir, ft)
            results[rel] = ft.text
        except Exception as e:  # pylint: disable=broad-except
            append_failure(failure_log, rel, e)
            results[rel] = ""
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description = "Multi-engine STT batch runner")
    parser.add_argument("input_dir", help = "입력 음성 폴더 경로(비재귀)")
    parser.add_argument("--engines", default = "fw,wcpp,w2v", help = "실행할 엔진(comma): fw,wcpp,w2v")
    parser.add_argument("--fw-model", default = "medium", help = "faster-whisper 모델 이름")
    parser.add_argument("--fw-device", default = "cuda", help = "faster-whisper 장치(cuda/cpu)")
    parser.add_argument("--fw-compute-type", default = "float16", help = "faster-whisper compute_type")
    parser.add_argument("--w2v-model", default = "kresnik/wav2vec2-large-xlsr-korean", help = "Wav2Vec2 모델 id")
    parser.add_argument("--w2v-device", default = "cuda", help = "Wav2Vec2 장치(cuda/cpu)")
    parser.add_argument("--w2v-chunk", type = int, default = 30, help = "Wav2Vec2 청크 초")
    parser.add_argument("--w2v-overlap", type = float, default = 1.0, help = "Wav2Vec2 오버랩 초")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)

    # 엔진 선택 파싱
    selected = {e.strip() for e in args.engines.split(",") if e.strip()}
    valid = {"fw", "wcpp", "w2v"}
    unknown = selected - valid
    if unknown:
        print(f"[경고] 알 수 없는 엔진 키워드 무시: {sorted(list(unknown))}")
    selected = selected & valid
    if not selected:
        print("[오류] 실행할 엔진이 없습니다. --engines fw,w2v 같은 형식으로 지정하세요.")
        sys.exit(2)

    os.makedirs(FW_OUT, exist_ok = True)
    os.makedirs(WCPP_OUT, exist_ok = True)
    os.makedirs(W2V_OUT, exist_ok = True)

    # Faster-Whisper
    fw_texts: Dict[str, str] = {}
    if "fw" in selected:
        try:
            fw_engine = FasterWhisperEngine(model_name = args.fw_model, device = args.fw_device, compute_type = args.fw_compute_type, language = "ko")
            fw_texts = run_engine_over_files("faster-whisper", fw_engine, input_dir, FW_OUT, FAIL_FW)
        except Exception as e:
            print(f"[경고] faster-whisper 준비 실패: {e}")
            fw_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}
    else:
        fw_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}

    # whisper.cpp
    wcpp_texts: Dict[str, str] = {}
    if "wcpp" in selected:
        try:
            wcpp_engine = WhisperCppEngine(language = "ko")
            if wcpp_engine.is_available():
                wcpp_texts = run_engine_over_files("whisper.cpp", wcpp_engine, input_dir, WCPP_OUT, FAIL_WCPP)
            else:
                print("[경고] whisper.cpp 실행 파일 또는 모델이 없어 건너뜁니다.")
                wcpp_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}
        except Exception as e:
            print(f"[경고] whisper.cpp 준비 실패: {e}")
            wcpp_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}
    else:
        wcpp_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}

    # Wav2Vec2
    w2v_texts: Dict[str, str] = {}
    if "w2v" in selected:
        try:
            w2v_engine = Wav2Vec2Engine(model_id = args.w2v_model, device = args.w2v_device, chunk_sec = args.w2v_chunk, overlap_sec = args.w2v_overlap)
            w2v_texts = run_engine_over_files("wav2vec2", w2v_engine, input_dir, W2V_OUT, FAIL_W2V)
        except Exception as e:
            print(f"[경고] Wav2Vec2 준비 실패: {e}")
            w2v_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}
    else:
        w2v_texts = {os.path.basename(f): "" for f in list_audio_files(input_dir)}

    # 비교 CSV 생성 (선택된 엔진만 컬럼 생성)
    files = list_audio_files(input_dir)
    col_map = [("fw", "fw_text", fw_texts), ("wcpp", "wcpp_text", wcpp_texts), ("w2v", "w2v_text", w2v_texts)]
    cols = ["file"] + [label for key, label, _ in col_map if key in selected]
    with open(COMPARE_CSV, "w", encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for p in files:
            base = os.path.basename(p)
            row = [base]
            for key, label, store in col_map:
                if key in selected:
                    row.append(store.get(base, ""))
            writer.writerow(row)

    print("완료: 결과 폴더 및 compare_transcripts.csv 생성")


if __name__ == "__main__":
    main()

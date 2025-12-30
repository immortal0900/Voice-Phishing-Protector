#!/usr/bin/env python3
# 간단 테스트 유틸리티: rt_pipeline.feeder_from_audio를 호출해서 청크를 생성합니다.
# 사용법:
# python tools\make_test_chunks.py --input "C:\path\to\input.mp3" --outdir "C:\WorkSpace\September_Project\live_chunks" --chunk-sec 5 --simulate-realtime

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트에 있는 rt_pipeline 모듈을 import할 수 있도록 경로 조정
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_pipeline import feeder_from_audio


def main():
    p = argparse.ArgumentParser(description="Make test chunks using rt_pipeline.feeder_from_audio")
    p.add_argument("--input", required=False, help="입력 오디오 파일 경로 (wav 또는 mp3 등). 없으면 --gen-wav로 테스트용 파일을 생성")
    p.add_argument("--gen-wav", action="store_true", help="테스트용 짧은 WAV 파일을 자동 생성하여 사용")
    p.add_argument("--outdir", default=str(ROOT / "live_chunks"), help="청크를 생성할 폴더")
    p.add_argument("--chunk-sec", type=int, default=5, help="청크 길이(초)")
    p.add_argument("--simulate-realtime", action="store_true", help="청크마다 sleep을 수행")
    args = p.parse_args()

    if args.gen_wav:
        # Generate a short 5s mono 16k sine wave wav for testing
        gen_path = ROOT / "_test_input.wav"
        import numpy as np
        import soundfile as sf
        sr = 16000
        t = np.linspace(0, 5, int(sr * 5), False)
        freq = 440.0
        data = 0.1 * np.sin(2 * np.pi * freq * t)
        sf.write(str(gen_path), data, sr)
        in_path = gen_path
        print(f"[TEST] generated test WAV: {in_path}")
    else:
        if not args.input:
            print("[ERROR] --input 또는 --gen-wav 중 하나를 지정하세요")
            return
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"[ERROR] 입력 파일이 존재하지 않습니다: {in_path}")
            return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[TEST] input={in_path} -> outdir={outdir} chunk_sec={args.chunk_sec} simulate_realtime={args.simulate_realtime}")

    try:
        feeder_from_audio(
            input_audio=str(in_path),
            out_dir=str(outdir),
            chunk_sec=args.chunk_sec,
            simulate_realtime=args.simulate_realtime,
            prefix="chunk_test_",
            feeder_done=None,
        )
        print("[TEST] feeder finished")
        # list created files
        import glob
        created = sorted(glob.glob(str(outdir / "*.wav")))
        print(f"[TEST] outdir listing (count={len(created)}): {created}")
    except Exception as e:
        print(f"[TEST] feeder raised exception: {e}")


if __name__ == '__main__':
    main()

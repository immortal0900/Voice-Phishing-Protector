#!/usr/bin/env python3
"""
도구: FFmpeg 변환을 테스트합니다.
사용법:
  python tools\ffmpeg_test.py --input "C:\path\to\file.mp3"

출력: 실행할 ffmpeg 명령, 리턴코드, stdout/stderr, 임시 변환 파일 존재 여부를 보여줍니다.
"""
import argparse
import subprocess
import tempfile
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='원본 오디오(mp3 등)')
    p.add_argument('--sr', type=int, default=16000)
    p.add_argument('--ac', type=int, default=1)
    args = p.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] input not found: {input_path}")
        return

    fd, tmp = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    cmd = [
        'ffmpeg', '-y', '-i', input_path, '-ar', str(args.sr), '-ac', str(args.ac), tmp
    ]
    print('[INFO] running command:')
    print(' '.join(cmd))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print('[ERROR] ffmpeg not found (FileNotFoundError). Is ffmpeg installed and in PATH?')
        print('Try: ffmpeg -version')
        return

    print('\n[INFO] returncode:', proc.returncode)
    print('\n[INFO] stdout:\n', proc.stdout)
    print('\n[INFO] stderr:\n', proc.stderr)

    exists = os.path.exists(tmp)
    print(f"\n[INFO] tmp file: {tmp} exists={exists}")
    if exists:
        try:
            sz = os.stat(tmp).st_size
            print(f"[INFO] tmp size: {sz} bytes")
        except Exception as e:
            print(f"[WARN] could not stat tmp: {e}")

    # cleanup prompt
    if exists:
        try:
            os.unlink(tmp)
            print('[INFO] tmp file removed')
        except Exception as e:
            print('[WARN] failed to remove tmp:', e)

if __name__ == '__main__':
    main()

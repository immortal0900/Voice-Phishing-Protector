#!/usr/bin/env python3
"""
Cleanup script: removes generated test artifacts created during debugging.
Removes:
 - project root: _test_input.wav, converted_test.wav
 - live_chunks/*
 - any chunk_test_*.wav files
"""
import os
from pathlib import Path

def remove_if_exists(p: Path):
    try:
        if p.exists():
            if p.is_file():
                p.unlink()
                print(f"removed file: {p}")
            elif p.is_dir():
                for child in p.glob('*'):
                    try:
                        if child.is_file():
                            child.unlink()
                            print(f"removed file: {child}")
                        else:
                            # don't recurse into directories; skip
                            pass
                    except Exception as e:
                        print(f"failed to remove {child}: {e}")
                print(f"cleared dir: {p}")
    except Exception as e:
        print(f"error handling {p}: {e}")

ROOT = Path(__file__).resolve().parents[1]
items = [
    ROOT / '_test_input.wav',
    ROOT / 'converted_test.wav',
    ROOT / 'live_chunks',
]

print(f"Cleaning test artifacts under {ROOT}")
for it in items:
    remove_if_exists(it)

# additional pattern: chunk_test_*.wav anywhere under project
for p in ROOT.rglob('chunk_test_*.wav'):
    try:
        p.unlink()
        print(f"removed pattern file: {p}")
    except Exception as e:
        print(f"failed to remove pattern file {p}: {e}")

print('Cleanup done.')

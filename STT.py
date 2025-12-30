"""STT 모듈
사용자 요구사항 기반 구현:
1. 엔진: SpeechRecognition + Google Web API (항상 재처리)
2. 언어: 한국어 전용 (language='ko-KR')
3. 출력: TXT (원본 파일명 기준 .txt, 결과 루트: ./stt_voice_files)
4. 세그먼트: 단일 전체 문장만 저장 (세그먼트 불필요)
5. 화자 분리: 필요 -> 1차 버전은 단순 VAD 기반 음성 덩어리 구간별 diarization placeholder ("speaker_1" 고정) 제공
6. 신뢰도: SpeechRecognition Google API는 공식 confidence 제공 X -> 구간별 재인식 시 가짜 점수 대신 None (향후 다른 엔진 교체시 확장)
7. 배치: 지정 폴더 전체 변환 (재귀 X 기본). always reprocess
8. 출력 디렉터리 구조: ./stt_voice_files/<상대파일명>.txt
9. 캐싱 없음
10. 전처리: VAD, 볼륨 정규화, 샘플레이트 16k 통일
11. GPU 사용 지정 불가(해당 엔진), 무시
12. 오류: 각 파일 2회 재시도, 실패 로그 기록
13. 진행: tqdm 표기
14. 확장자: 일반 음성 확장자 모두(.wav .mp3 .m4a .flac .ogg .webm .aac)
15. 긴 파일 분할: 60초 단위 chunk (설정 가능)
16. 모델 크기: (엔진 특성상 비적용)
17. 최대 길이 제한 없음
18. 민감정보 마스킹 없음
19. LLM 후처리용 메타: JSON 동시 저장 (텍스트 외 메타) -> 사용자는 txt 요구, 내부적으로 .meta.json 추가
20. 환경변수 필요 없음 (Google Web API 비공식 클라이언트, quota 주의)
21. 추가 설치 허용: pydub 필요

추후 Whisper 로 교체/병행을 쉽게 하기 위해 Engine 인터페이스 구조화.
"""

from __future__ import annotations

import os
import io
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import speech_recognition as sr
from pydub import AudioSegment, effects

try:
	import webrtcvad  # VAD
	_VAD_AVAILABLE = True
except ImportError:  # fallback 비가용 시 무음 기반 단순 분할
	_VAD_AVAILABLE = False

from tqdm import tqdm

SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MAX_SEC = 60  # 기본 분할 길이(초)
RETRY = 2
OUTPUT_ROOT = os.path.abspath("stt_voice_files")


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
	engine: str = "speechrecognition_google"
	processed_at: float = time.time()
	error: Optional[str] = None

	def to_meta_dict(self):
		d = asdict(self)
		d["segments"] = [asdict(s) for s in self.segments]
		return d


class AudioPreprocessor:
	def __init__(self, target_sample_rate: int = DEFAULT_SAMPLE_RATE):
		self.target_sample_rate = target_sample_rate

	def load(self, path: str) -> AudioSegment:
		audio = AudioSegment.from_file(path)
		# 볼륨 정규화
		audio = effects.normalize(audio)
		# 샘플레이트 변환
		if audio.frame_rate != self.target_sample_rate:
			audio = audio.set_frame_rate(self.target_sample_rate)
		# 모노로 통일
		if audio.channels != 1:
			audio = audio.set_channels(1)
		return audio

	def vad_chunks(self, audio: AudioSegment, frame_ms: int = 30, aggressiveness: int = 2, max_chunk_sec: int = DEFAULT_CHUNK_MAX_SEC) -> List[AudioSegment]:
		"""VAD 기반 유효 음성 덩어리 리스트 반환. webrtcvad 없으면 길이 제한 chunking."""
		if not _VAD_AVAILABLE:
			# webrtcvad 미설치 시 max_chunk_sec 단위로 단순 분할
			chunks = []
			total_ms = len(audio)
			step = max_chunk_sec * 1000
			for start in range(0, total_ms, step):
				end = min(start + step, total_ms)
				chunk = audio[start:end]
				if len(chunk) > 1000:  # 최소 1초
					chunks.append(chunk)
			return chunks

		vad = webrtcvad.Vad(aggressiveness)
		raw = audio.raw_data
		sample_rate = audio.frame_rate
		bytes_per_frame = 2  # 16bit
		frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_frame)
		voiced_segments: List[AudioSegment] = []
		cur_segment = AudioSegment.silent(duration=0, frame_rate=sample_rate)
		silence_accum = 0
		max_silence_ms = 800
		max_chunk_ms = max_chunk_sec * 1000

		for i in range(0, len(raw), frame_bytes):
			frame = raw[i:i + frame_bytes]
			if len(frame) < frame_bytes:
				break
			is_speech = vad.is_speech(frame, sample_rate)
			frame_audio = AudioSegment(
				frame,
				sample_width=2,
				frame_rate=sample_rate,
				channels=1
			)
			if is_speech:
				cur_segment += frame_audio
				silence_accum = 0
			else:
				silence_accum += frame_ms
				if silence_accum >= max_silence_ms or len(cur_segment) >= max_chunk_ms:
					if len(cur_segment) > 500:  # >0.5s
						voiced_segments.append(cur_segment)
					cur_segment = AudioSegment.silent(duration=0, frame_rate=sample_rate)
					silence_accum = 0
		if len(cur_segment) > 500:
			voiced_segments.append(cur_segment)
		return voiced_segments or [audio]


class GoogleRecognizerEngine:
	def __init__(self, language: str = "ko-KR"):
		self.language = language
		self.recognizer = sr.Recognizer()

	def transcribe_chunk(self, chunk: AudioSegment) -> str:
		with io.BytesIO() as buf:
			chunk.export(buf, format="wav")
			buf.seek(0)
			with sr.AudioFile(buf) as source:
				audio_data = self.recognizer.record(source)
		# 네트워크 오류 재시도는 상위에서
		return self.recognizer.recognize_google(audio_data, language=self.language)



class STTProcessor:
	def __init__(self, input_dir: str, output_root: str = OUTPUT_ROOT, language: str = "ko-KR", chunk_sec: int = DEFAULT_CHUNK_MAX_SEC):
		self.input_dir = os.path.abspath(input_dir)
		self.output_root = os.path.abspath(output_root)
		os.makedirs(self.output_root, exist_ok=True)
		self.pre = AudioPreprocessor()
		self.engine = GoogleRecognizerEngine(language=language)
		self.chunk_sec = max(5, chunk_sec)

	def _gather_files(self) -> List[str]:
		files = []
		for name in os.listdir(self.input_dir):
			path = os.path.join(self.input_dir, name)
			if os.path.isfile(path) and os.path.splitext(name)[1].lower() in SUPPORTED_EXT:
				files.append(path)
		files.sort()
		return files

	def process_all(self) -> List[FileTranscription]:
		audio_files = self._gather_files()
		results: List[FileTranscription] = []
		for f in tqdm(audio_files, desc="Transcribing", unit="file"):
			try:
				res = self.process_file(f)
				results.append(res)
			except Exception as e:  # pylint: disable=broad-except
				# 파일 단위 실패 메타 작성
				rel = os.path.relpath(f, self.input_dir)
				ft = FileTranscription(
					file=os.path.basename(f),
					rel_path=rel,
					language="ko-KR",
					text="",
					duration_sec=0.0,
					segments=[],
					error=str(e)
				)
				results.append(ft)
				self._write_outputs(ft)
		return results

	def process_file(self, file_path: str) -> FileTranscription:
		audio = self.pre.load(file_path)
		duration_sec = len(audio) / 1000.0
		chunks = self.pre.vad_chunks(audio, max_chunk_sec=self.chunk_sec)
		segments: List[SegmentResult] = []
		full_text_parts: List[str] = []
		for idx, ch in enumerate(chunks):
			attempt = 0
			last_err = None
			while attempt <= RETRY:
				try:
					text = self.engine.transcribe_chunk(ch)
					start = sum(len(c) for c in chunks[:idx]) / 1000.0
					end = start + len(ch) / 1000.0
					seg = SegmentResult(index=idx, start=start, end=end, text=text)
					segments.append(seg)
					full_text_parts.append(text)
					break
				except Exception as e:  # pylint: disable=broad-except
					last_err = e
					attempt += 1
					if attempt > RETRY:
						# 실패 세그먼트 표기
						start = sum(len(c) for c in chunks[:idx]) / 1000.0
						end = start + len(ch) / 1000.0
						seg = SegmentResult(index=idx, start=start, end=end, text=f"<ERROR:{e}>")
						segments.append(seg)
						full_text_parts.append("")
		transcript_text = " ".join(filter(None, full_text_parts)).strip()
		rel = os.path.relpath(file_path, self.input_dir)
		ft = FileTranscription(
			file=os.path.basename(file_path),
			rel_path=rel,
			language="ko-KR",
			text=transcript_text,
			duration_sec=duration_sec,
			segments=segments
		)
		self._write_outputs(ft)
		return ft

	def _write_outputs(self, ft: FileTranscription):
		# txt 저장
		base_name = os.path.splitext(os.path.basename(ft.file))[0]
		txt_path = os.path.join(self.output_root, f"{base_name}.txt")
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(ft.text or "")
		# meta json (LLM 연계용)
		meta_path = os.path.join(self.output_root, f"{base_name}.meta.json")
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(ft.to_meta_dict(), f, ensure_ascii=False, indent=2)


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Batch STT for voice phishing dataset (Google Web API)")
	parser.add_argument("input_dir", help="입력 음성 폴더 경로")
	parser.add_argument("--out", default=OUTPUT_ROOT, help="출력 루트 (기본: stt_voice_files)")
	parser.add_argument("--lang", default="ko-KR", help="언어 코드")
	parser.add_argument("--chunk-sec", type=int, default=DEFAULT_CHUNK_MAX_SEC, help="최대 청크 길이(초)")
	args = parser.parse_args()

	proc = STTProcessor(args.input_dir, args.out, language=args.lang, chunk_sec=args.chunk_sec)
	proc.process_all()


if __name__ == "__main__":
	main()

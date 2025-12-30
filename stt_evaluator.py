import speech_recognition as sr
import os
import glob
import librosa
import numpy as np
from tqdm import tqdm

# --- 설정 ---
AUDIO_SOURCE_DIR = "fss_paired_audio"
OUTPUT_DIR = "stt_results/Google-Speech-Recognition"
TARGET_SAMPLE_RATE = 16000
NUM_FILES_TO_PROCESS = 30

def process_with_google_sr():
    """
    지정된 폴더의 오디오 파일들을 Google Speech Recognition을 사용하여 처리하고,
    결과를 텍스트 파일로 저장합니다.
    """
    print("--- Google Speech Recognition으로 추론 시작 ---")

    # 1. 오디오 파일 목록 가져오기
    # glob을 사용하여 다양한 오디오 확장자를 한 번에 찾습니다.
    audio_files = []
    for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
        audio_files.extend(glob.glob(os.path.join(AUDIO_SOURCE_DIR, ext)))
    
    if not audio_files:
        print(f"오류: '{AUDIO_SOURCE_DIR}'에서 오디오 파일을 찾을 수 없습니다.")
        return

    # 처리할 파일 수를 30개로 제한
    files_to_process = audio_files[:NUM_FILES_TO_PROCESS]
    print(f"총 {len(files_to_process)}개의 오디오 파일을 처리합니다.")

    # 2. 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과 저장 폴더: '{OUTPUT_DIR}'")

    # 3. Recognizer 초기화
    recognizer = sr.Recognizer()

    # 4. 각 파일에 대해 추론 및 저장
    for audio_path in tqdm(files_to_process, desc="Google SR 처리 중"):
        hypothesis = ""
        try:
            # 오디오 로드 및 전처리
            waveform, sample_rate = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
            
            # speech_recognition이 요구하는 AudioData 형식으로 변환
            audio_bytes = (waveform * 32767).astype(np.int16).tobytes()
            audio_data = sr.AudioData(audio_bytes, sample_rate, 2)  # 16-bit 오디오

            # Google API로 음성 인식 수행
            hypothesis = recognizer.recognize_google(audio_data, language='ko-KR')

        except sr.UnknownValueError:
            hypothesis = "음성을 인식할 수 없습니다."
            print(f"'{os.path.basename(audio_path)}' 처리 중: 음성을 인식할 수 없습니다.")
        except sr.RequestError as e:
            hypothesis = f"API 요청 오류: {e}"
            print(f"'{os.path.basename(audio_path)}' 처리 중 API 요청 오류 발생: {e}")
        except Exception as e:
            hypothesis = f"알 수 없는 오류: {e}"
            print(f"'{os.path.basename(audio_path)}' 처리 중 알 수 없는 오류 발생: {e}")

        # 5. 결과 저장
        # 원본 파일명에서 확장자만 .txt로 변경하여 저장
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(hypothesis)

    print(f"\n--- 모든 작업 완료 ---")
    print(f"{len(files_to_process)}개의 파일에 대한 결과가 '{OUTPUT_DIR}'에 저장되었습니다.")


if __name__ == "__main__":
    process_with_google_sr()

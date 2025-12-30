import os
import glob
import pandas as pd
import evaluate
from tqdm import tqdm

# --- 설정 ---
# 라벨(정답) 텍스트 파일이 있는 디렉토리
LABEL_DIR = "fss_paired_text"

# 모델들의 예측 결과가 저장된 기본 디렉토리
PREDICTION_BASE_DIR = "stt_results"

# 평가할 모델의 이름 (PREDICTION_BASE_DIR 아래에 있는 폴더명과 일치해야 함)
MODELS_TO_EVALUATE = [
    "Google-Speech-Recognition",
    # "Fine-tuned-Whisper",
    "Faster-Whisper",
    "Wav2Vec2"
]

def evaluate_stt_performance():
    """
    라벨과 예측 데이터를 비교하여 STT 모델들의 성능(WER, CER)을 평가합니다.
    """
    print("--- STT 모델 성능 평가 시작 ---")

    # 1. 평가 지표 로드 (WER: 단어 오류율, CER: 글자 오류율)
    try:
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
    except Exception as e:
        print(f"평가 지표 로드 중 오류 발생: {e}")
        print("'evaluate' 및 'jiwer' 라이브러리가 설치되었는지 확인하세요.")
        print("pip install evaluate jiwer")
        return

    # 2. 라벨 파일 목록 가져오기
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    if not label_files:
        print(f"오류: 라벨 디렉토리 '{LABEL_DIR}'에서 텍스트 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(label_files)}개의 라벨 파일을 기준으로 평가를 진행합니다.")

    # 최종 결과를 저장할 리스트
    results_data = []

    # 3. 각 모델에 대해 평가 수행
    for model_name in MODELS_TO_EVALUATE:
        print(f"\n--- '{model_name}' 모델 평가 중 ---")
        
        prediction_dir = os.path.join(PREDICTION_BASE_DIR, model_name)
        if not os.path.isdir(prediction_dir):
            print(f"경고: 예측 폴더 '{prediction_dir}'를 찾을 수 없어 건너뜁니다.")
            continue

        # 해당 모델의 예측(hypothesis)과 라벨(reference)을 담을 리스트
        predictions = []
        references = []
        
        # 4. 라벨-예측 파일 쌍을 찾아 데이터 로드
        found_pairs = 0
        for label_path in tqdm(label_files, desc=f"'{model_name}' 파일 매칭 중"):
            base_filename = os.path.basename(label_path)
            prediction_path = os.path.join(prediction_dir, base_filename)

            if os.path.exists(prediction_path):
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        ref_text = f.read().strip()
                    with open(prediction_path, 'r', encoding='utf-8') as f:
                        pred_text = f.read().strip()
                    
                    # 라벨과 예측이 모두 내용이 있는 경우에만 추가
                    if ref_text and pred_text:
                        references.append(ref_text)
                        predictions.append(pred_text)
                        found_pairs += 1

                except Exception as e:
                    print(f"파일 읽기 오류 ({base_filename}): {e}")
        
        if not references:
            print(f"오류: '{model_name}' 모델에 대해 평가할 데이터 쌍을 찾지 못했습니다.")
            continue
            
        print(f"총 {found_pairs}개의 평가 가능한 파일 쌍을 찾았습니다.")

        # 5. 성능 지표 계산
        try:
            wer = wer_metric.compute(predictions=predictions, references=references)
            cer = cer_metric.compute(predictions=predictions, references=references)
            
            results_data.append({
                "Model": model_name,
                "WER": f"{wer:.4f}",
                "CER": f"{cer:.4f}",
                "Compared_Files": found_pairs
            })
        except Exception as e:
            print(f"'{model_name}' 모델의 성능 계산 중 오류 발생: {e}")

    # 6. 최종 결과 출력
    if not results_data:
        print("\n평가할 모델이 없거나 데이터를 찾을 수 없어 결과를 출력할 수 없습니다.")
        return

    print("\n\n--- 최종 평가 결과 ---")
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    print("\n* WER(단어 오류율)과 CER(글자 오류율)은 낮을수록 좋은 성능을 의미합니다.")


if __name__ == "__main__":
    evaluate_stt_performance()

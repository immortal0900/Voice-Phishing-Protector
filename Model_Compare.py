import ollama
import pandas as pd
import json
from typing import List, Dict, Any
import os
import glob
from multiprocessing import Pool, TimeoutError
import time
from tqdm import tqdm
import random

# --- 설정 ---
# 테스트할 OLLAMA 모델 목록
MODELS_TO_TEST = [
    "gemma2:9b",
    "deepseek-r1:8b",
    "llama3.1:8b",
]

# 작업자 프로세스 수 (CPU 코어 수의 절반 정도를 권장)
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1

# --- 데이터 및 프롬프트 경로 설정 ---
SYSTEM_PROMPT_FILE = "system_ko.txt"
PROBLEM_LOG_FILE = "problematic_samples.log"
TIMEOUT_SECONDS = 60  # 각 샘플 처리 시간 제한 (초)

# '레이블': ['경로1', '경로2', ...] 형식으로 구성
DATA_PATHS = {
    "보이스피싱": [
        "fss_paired_text",
    ],
    "일반대화": [
        "normal_conversations.txt",
    ]
}

def load_system_prompt(file_path: str) -> str:
    """지정된 파일에서 시스템 프롬프트를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"경고: 시스템 프롬프트 파일('{file_path}')을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
        return """당신은 보이스피싱 탐지 전문가입니다. 주어진 텍스트가 보이스피싱인지 일반대화인지 분류하고, 그 결과를 JSON 형식으로만 출력해야 합니다.
출력 형식은 다음과 같아야 합니다:
{
  "classification": "보이스피싱 또는 일반대화",
  "risk_score": 0~100 사이의 정수,
  "is_above_threshold": true 또는 false,
  "reasoning": "판단 이유 요약"
}
다른 설명은 절대 추가하지 마세요."""

def load_test_data(data_paths: Dict[str, List[str]], max_per_label: int = 50) -> List[Dict[str, str]]:
    """지정된 경로에서 텍스트 파일을 읽어 레이블별로 지정된 수만큼 테스트 데이터를 생성합니다."""
    loaded_data = []
    for label, paths in data_paths.items():
        label_specific_data = []
        for path in paths:
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        for content in lines:
                            label_specific_data.append({"text": content, "label": label})
                except Exception as e:
                    print(f"파일을 읽는 중 오류 발생 ({path}): {e}")
            elif os.path.isdir(path):
                files = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if content:
                            label_specific_data.append({"text": content, "label": label})
                    except Exception as e:
                        print(f"파일을 읽는 중 오류 발생 ({file_path}): {e}")
            else:
                print(f"경고: 데이터 경로를 찾을 수 없습니다: {path}")
        
        # 데이터가 충분하면 랜덤으로 샘플링, 아니면 있는 만큼만 사용
        if len(label_specific_data) > max_per_label:
            loaded_data.extend(random.sample(label_specific_data, max_per_label))
        else:
            loaded_data.extend(label_specific_data)

    if not loaded_data:
        raise ValueError("테스트 데이터를 불러오지 못했습니다. DATA_PATHS를 확인하세요.")
    return loaded_data

# ... 이하 코드는 동일 ...

def log_problematic_sample(model_name: str, error_type: str, text: str, label: str, raw_response: str = ""):
    """문제가 발생한 샘플을 파일에 기록합니다."""
    with open(PROBLEM_LOG_FILE, 'a', encoding='utf-8') as f:
        log_entry = {
            "model": model_name,
            "error_type": error_type,
            "true_label": label,
            "text": text,
            "raw_response": raw_response
        }
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# 모델에 역할을 부여하는 시스템 프롬프트 (전역 변수로 선언)
SYSTEM_PROMPT = load_system_prompt(SYSTEM_PROMPT_FILE)

def classification_worker(args):
    """
    프로세스 풀의 작업자 함수. 단일 인수를 받아 처리 결과를 반환합니다.
    """
    model_name, user_text, true_label = args
    raw_response_content = "N/A"
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_text}
            ],
            format='json',
            options={"temperature": 0.0}
        )
        raw_response_content = response['message']['content']
        response_json = json.loads(raw_response_content)
        is_phishing = response_json.get("is_above_threshold")

        if is_phishing is True:
            predicted_label = "보이스피싱"
        elif is_phishing is False:
            predicted_label = "일반대화"
        else:
            predicted_label = "판단불가"
            log_problematic_sample(model_name, "판단불가", user_text, true_label, raw_response_content)

    except json.JSONDecodeError:
        predicted_label = "JSON오류"
        log_problematic_sample(model_name, "JSON오류", user_text, true_label, raw_response_content)
    except Exception as e:
        predicted_label = "오류"
        log_problematic_sample(model_name, "오류", user_text, true_label, str(e))
    
    return model_name, user_text, true_label, predicted_label

def warm_up_model(model_name: str):
    """
    모델을 메모리에 미리 로드하고 안정화(워밍업)합니다.
    """
    tqdm.write(f"'{model_name}' 모델을 메모리에 로드하고 안정화하는 중... 잠시만 기다려주세요.")
    try:
        ollama.generate(model=model_name, prompt="Hello", stream=False)
        tqdm.write(f"'{model_name}' 모델 준비 완료.")
    except Exception as e:
        tqdm.write(f"'{model_name}' 모델 워밍업 중 오류 발생: {e}")

def run_comparison(test_data: List[Dict[str, str]]) -> pd.DataFrame:
    """
    프로세스 풀을 사용하여 모든 모델과 데이터에 대해 성능 비교를 실행합니다.
    """
    all_results: List[Dict[str, Any]] = []

    if os.path.exists(PROBLEM_LOG_FILE):
        os.remove(PROBLEM_LOG_FILE)

    for model in MODELS_TO_TEST:
        warm_up_model(model)
        
        tasks = []
        for item in test_data:
            user_text = item["text"]
            if len(user_text) > 500:
                user_text = user_text[:500]
            tasks.append((model, user_text, item["label"]))

        tqdm.write(f"\n--- '{model}' 모델 테스트 시작 ({len(tasks)}개 작업) ---")
        
        model_results = []
        # with Pool(processes=NUM_WORKERS) as pool:
        #     with tqdm(total=len(tasks), desc=f"Model: {model}") as pbar:
        #         # imap_unordered는 작업이 완료되는 순서대로 결과를 반환
        #         for result in pool.imap_unordered(classification_worker, tasks):
        #             model_results.append(result)
        #             pbar.update(1)
        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(tasks), desc=f"Model: {model}") as pbar:
                async_results = [pool.apply_async(classification_worker, args=(task,)) for task in tasks]
                
                for res in async_results:
                    try:
                        # 타임아웃을 설정하여 결과를 기다림
                        result = res.get(timeout=TIMEOUT_SECONDS)
                        model_results.append(result)
                    except TimeoutError:
                        # 타임아웃 발생 시 로그 기록
                        # task 인수를 찾기 어려우므로, 이 부분은 개선이 필요할 수 있음
                        # 여기서는 간단히 로그만 남깁니다.
                        log_problematic_sample(model, "Timeout", "N/A", "N/A")
                    except Exception as e:
                        log_problematic_sample(model, "PoolError", "N/A", "N/A", str(e))
                    finally:
                        pbar.update(1)


        # 결과 처리
        for model_name, user_text, true_label, predicted_label in model_results:
            valid_labels = {"보이스피싱", "일반대화"}
            hyp_for_cer = predicted_label if predicted_label in valid_labels else ""
            cer = char_error_rate(true_label, hyp_for_cer)
            is_correct = (predicted_label == true_label)

            all_results.append({
                "model": model_name,
                "text": user_text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "cer": cer,
            })

    return pd.DataFrame(all_results)

# --- CER 계산 함수 (변경 없음) ---
def _levenshtein_distance(ref: str, hyp: str) -> int:
    if not ref: return len(hyp)
    if not hyp: return len(ref)
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): dp[i][0] = i
    for j in range(len(hyp) + 1): dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(ref)][len(hyp)]

def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_len = max(1, len(reference))
    return _levenshtein_distance(reference, hypothesis) / ref_len


if __name__ == "__main__":
    # pandas와 tqdm이 설치되어 있지 않다면 터미널에 'uv pip install pandas tqdm'을 입력하여 설치하세요.
    
    # 데이터 로드
    TEST_DATA = load_test_data(DATA_PATHS)
    print(f"총 {len(TEST_DATA)}개의 테스트 데이터를 로드했습니다. (작업자 수: {NUM_WORKERS})")
    
    # 성능 비교 실행
    results_df = run_comparison(TEST_DATA)
    
    # 결과 요약
    print("\n\n--- 최종 결과 요약 ---")
    if not results_df.empty:
        results_df['is_correct'] = pd.to_numeric(results_df['is_correct'], errors='coerce').fillna(0).astype(bool)
        pd.set_option('display.max_colwidth', 50)
        print(results_df)
        
        # 모델별 정확도 및 CER 계산
        print("\n--- 모델별 정확도 및 CER ---")
        summary = results_df.groupby('model').agg(
            accuracy=('is_correct', 'mean'),
            cer=('cer', 'mean')
        ).reset_index()
        summary['accuracy'] = summary['accuracy'].round(4)
        summary['cer'] = summary['cer'].round(4)
        print(summary.to_string(index=False))
    else:
        print("결과 데이터가 없습니다.")

    # 문제 로그 파일 확인
    if os.path.exists(PROBLEM_LOG_FILE):
        print(f"\n--- 문제가 발생한 샘플 로그 ---")
        print(f"'{PROBLEM_LOG_FILE}' 파일에서 자세한 내용을 확인하세요.")
        with open(PROBLEM_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:5]:
                print(line.strip())
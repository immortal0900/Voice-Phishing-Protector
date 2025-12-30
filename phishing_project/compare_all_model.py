# --- 필요한 라이브러리 불러오기 ---
import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

# ==============================================================================
# ✨ 헬퍼 함수 1: 검증 데이터 로드 ✨
# (이전과 동일)
# ==============================================================================
def load_validation_data(normal_path, phishing_path):
    """
    [함수 기능]
    검증 데이터 폴더에서 json 파일들을 읽어 'text'와 'label'을 가진 DataFrame으로 반환합니다.
    """
    data_list = []
    for label, path in [(0, normal_path), (1, phishing_path)]:
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                file_path = os.path.join(path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                conversation = " ".join([d['text'] for d in json_data['dataSet']['dialogs']]) if label == 0 else json_data['text']
                data_list.append({'text': conversation, 'label': label})
    return pd.DataFrame(data_list)

# ==============================================================================
# ✨ 헬퍼 함수 2: ELECTRA 모델 예측 ✨
# (이전과 동일)
# ==============================================================================
def get_electra_prediction(model, tokenizer, sentence):
    """
    [함수 기능]
    ELECTRA와 같은 분류 모델을 위한 예측 함수입니다.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=-1).item()

# ==============================================================================
# ✨ 헬퍼 함수 3: Gemma 모델 예측 ✨
# (이전과 동일)
# ==============================================================================
def get_gemma_prediction(model, tokenizer, sentence):
    """
    [함수 기능]
    Gemma와 같은 생성 모델을 위한 예측 함수입니다.
    """
    prompt_template = "다음 대화 내용이 보이스피싱인지 아닌지 판단 후, '보이스피싱' 또는 '일반 대화'로만 답변해줘.\n\n### 대화 내용:\n{conversation}\n\n### 판단:\n"
    prompt = prompt_template.format(conversation=sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=5)
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result_text.split("### 판단:")[-1].strip()
    return 1 if "보이스피싱" in answer else 0

# ==============================================================================
# ✨ 헬퍼 함수 4: 정확도 계산 ✨
# (이전과 동일)
# ==============================================================================
def calculate_accuracy(true_labels, predicted_labels):
    """
    [함수 기능]
    실제 정답 리스트와 모델 예측 리스트를 비교하여 정확도를 계산합니다.
    """
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return (correct / len(true_labels)) * 100

# ==============================================================================
# ✨ 메인 실행 부분 ✨
# ==============================================================================
if __name__ == "__main__":

    # --- 1. 공통 검증 데이터 준비 ---
    print("--- 1. 성능 비교를 위한 공통 검증 데이터를 불러옵니다. ---")
    val_df = load_validation_data("./val_data/normal", "./val_data/phishing")
    true_labels = val_df['label'].tolist()
    print(f"총 {len(val_df)}개의 검증 데이터 준비 완료.")

    results = {}

   
    # 2. 원본 ELECTRA 모델 성능 평가 (새로 추가된 섹션) 
   
    print("\n--- 2. [ELECTRA-Base] 원본 모델의 성능을 평가합니다. ---")
    electra_base_name = "monologg/koelectra-base-v3-discriminator"
    
    # (출처: Hugging Face 라이브러리에서 사전 학습된 모델을 불러오는 표준 방식)
    electra_base_tokenizer = AutoTokenizer.from_pretrained(electra_base_name)
    # num_labels=2: 우리의 목표(일반/피싱)에 맞게 모델의 출력층을 초기화
    electra_base_model = AutoModelForSequenceClassification.from_pretrained(electra_base_name, num_labels=2)
    electra_base_model.eval()
    
    electra_base_preds = [get_electra_prediction(electra_base_model, electra_base_tokenizer, text) for text in val_df['text']]
    results["ELECTRA-Base"] = calculate_accuracy(true_labels, electra_base_preds)

    # --- 3. 파인튜닝된 ELECTRA 모델 성능 평가 ---
    print("\n--- 3. [ELECTRA-Finetuned] 모델의 성능을 평가합니다. ---")
    electra_tuned_path = "./fine-tuned-phishing-model"
    electra_tuned_tokenizer = AutoTokenizer.from_pretrained(electra_tuned_path)
    electra_tuned_model = AutoModelForSequenceClassification.from_pretrained(electra_tuned_path)
    electra_tuned_model.eval()
    
    electra_tuned_preds = [get_electra_prediction(electra_tuned_model, electra_tuned_tokenizer, text) for text in val_df['text']]
    results["ELECTRA-Finetuned"] = calculate_accuracy(true_labels, electra_tuned_preds)

    # --- 4. 원본 Gemma 모델 성능 평가 ---
    print("\n--- 4. [Gemma-Base] 원본 모델의 성능을 평가합니다. ---")
    gemma_base_name = "google/gemma-2b-it"
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_base_name)
    gemma_base_model = AutoModelForCausalLM.from_pretrained(gemma_base_name, device_map="auto", torch_dtype=torch.float16)
    gemma_base_model.eval()
    
    gemma_base_preds = [get_gemma_prediction(gemma_base_model, gemma_tokenizer, text) for text in val_df['text']]
    results["Gemma-Base"] = calculate_accuracy(true_labels, gemma_base_preds)
    
    del gemma_base_model
    torch.cuda.empty_cache()

    # --- 5. 파인튜닝된 Gemma 모델 성능 평가 ---
    print("\n--- 5. [Gemma-Finetuned] 모델의 성능을 평가합니다. ---")
    gemma_adapter_path = "./gemma-phishing-adapter"
    base_model = AutoModelForCausalLM.from_pretrained(gemma_base_name, device_map="auto", torch_dtype=torch.float16)
    gemma_tuned_model = PeftModel.from_pretrained(base_model, gemma_adapter_path)
    gemma_tuned_model.eval()
    
    gemma_tuned_preds = [get_gemma_prediction(gemma_tuned_model, gemma_tokenizer, text) for text in val_df['text']]
    results["Gemma-Finetuned"] = calculate_accuracy(true_labels, gemma_tuned_preds)

    # --- 6. 최종 결과 종합 출력 ---
    print("\n\n=============================================")
    print("           ✨ 모델 성능 종합 비교 ✨")
    print("=============================================")
    # 결과 딕셔너리를 순회하며 저장된 모든 결과를 출력
    for model_name, accuracy in results.items():
        print(f"{model_name:<20}: {accuracy:>6.2f}%")
    print("=============================================")
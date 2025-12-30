import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torchaudio
import evaluate
from datasets import load_dataset, DatasetDict, Audio, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import librosa
import json
from tqdm import tqdm

# --- 1. 설정 ---
# 모델 및 데이터셋 설정
MODEL_NAME = "openai/whisper-medium"
LANGUAGE = "ko"
TASK = "transcribe"

# 학습 관련 설정
OUTPUT_DIR = "./whisper-medium-finetuned-ko"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8 # VRAM 크기에 따라 조절 (medium 모델은 16GB VRAM에서 8정도가 안정적)
PER_DEVICE_EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
FP16 = True  # GPU가 지원하는 경우 True로 설정
GRADIENT_ACCUMULATION_STEPS = 2 # VRAM 부족 시 늘려서 배치 크기 효과를 얻음

# 저장 및 검증 관련 설정
POST_TRAIN_VALIDATE = True  # 학습 직후 저장된 어댑터를 8bit 로드로 검증
VALIDATION_SAMPLES = 20     # 테스트셋 상위 N개로 간단 검증
SAVE_INFERENCE_CONFIG = True  # 어댑터 경로에 로딩 정보 메타데이터 저장

# torchaudio 백엔드를 soundfile로 설정 (Windows 호환성 문제 해결)
try:
    torchaudio.set_audio_backend("soundfile")
except RuntimeError:
    print("torchaudio 백엔드를 'soundfile'로 설정하는 데 실패했습니다. 이미 다른 백엔드가 설정되었을 수 있습니다.")

def create_dataset_from_local_files():
    """로컬 파일로부터 Hugging Face 데이터셋을 생성합니다. (텍스트 파일 기준)"""
    print("로컬 파일로부터 데이터셋을 생성합니다... (텍스트 파일 기준)")
    
    # 텍스트 파일이 저장된 디렉토리와, 해당 텍스트에 매칭될 오디오 파일이 있는 디렉토리 목록
    # data_scraper.py로 생성된 데이터도 포함
    text_dirs = [
        "fss_text_outputs/B0000206_200690",
        "fss_text_outputs/B0000207_200691",
        "fss_paired_text",
    ]
    audio_roots = [
        "fss_voice_files",
        "fss_voice_files_200691",
        "fss_paired_audio",
    ]
    
    audio_extensions = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"]
    all_audio_files = []
    for audio_root in audio_roots:
        if os.path.isdir(audio_root):
            for ext in audio_extensions:
                all_audio_files.extend(glob.glob(os.path.join(audio_root, ext)))
    
    # 오디오 파일명을 빠르게 찾기 위한 맵 생성 (확장자 제외)
    audio_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in all_audio_files}
    
    data = []
    
    for text_dir in text_dirs:
        if not os.path.isdir(text_dir):
            print(f"경고: 텍스트 디렉토리를 찾을 수 없습니다: {text_dir}")
            continue
        
        text_files = glob.glob(os.path.join(text_dir, "*.txt"))
        
        for text_path in text_files:
            try:
                base_filename = os.path.splitext(os.path.basename(text_path))[0]
                
                # 매칭되는 오디오 파일 찾기
                if base_filename in audio_lookup:
                    audio_path = audio_lookup[base_filename]
                    with open(text_path, 'r', encoding='utf-8') as f:
                        sentence = f.read().strip()
                    
                    if sentence:
                        # 길이 제한 체크를 제거하고, 이후 전처리에서 라벨을 448 토큰으로 안전히 절단합니다.
                        data.append({"audio": audio_path, "sentence": sentence})
            except Exception as e:
                print(f"파일 처리 중 오류 발생 (텍스트: {text_path}): {e}")

    if not data:
        raise ValueError("데이터를 찾을 수 없습니다. 파일 경로와 구조를 확인하세요.")

    # Hugging Face Dataset 객체로 변환
    dataset = Dataset.from_dict({"audio": [item["audio"] for item in data], "sentence": [item["sentence"] for item in data]})
    
    # 90% train, 10% test로 분할
    train_test_split = dataset.train_test_split(test_size=0.1)
    
    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })
    
    print(f"데이터셋 생성 완료: Train {len(dataset_dict['train'])}개, Test {len(dataset_dict['test'])}개")
    return dataset_dict


def prepare_dataset(batch):
    """데이터셋 전처리 함수"""
    # librosa를 사용하여 오디오 파일을 16kHz로 직접 로드
    speech_array, sampling_rate = librosa.load(batch["audio"], sr=16000)

    # 입력 특성 추출
    batch["input_features"] = processor.feature_extractor(speech_array, sampling_rate=sampling_rate).input_features[0]
    # 타겟 텍스트를 라벨 ID로 인코딩
    labels = processor.tokenizer(batch["sentence"], truncation = False).input_ids

    # 오디오 길이에 비례하여 라벨 길이를 축소(Whisper가 약 30초 컨텍스트를 사용하는 점을 고려)
    max_audio_sec = 30.0
    duration_sec = len(speech_array) / float(sampling_rate) if sampling_rate > 0 else max_audio_sec
    if duration_sec > max_audio_sec and len(labels) > 0:
        ratio = max_audio_sec / duration_sec
        keep = max(16, int(len(labels) * ratio))  # 최소 16 토큰 보장
        labels = labels[:keep]

    # 디코더 최대 컨텍스트 448에 맞춰 최종 절단
    if len(labels) > 448:
        labels = labels[:448]

    batch["labels"] = labels
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """음성-텍스트 변환을 위한 데이터 콜레이터"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 입력 특성과 라벨의 길이를 동적으로 패딩
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩된 라벨은 -100으로 마스킹하여 손실 계산에서 제외
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 디코더 출력이 없는 경우, bos_token_id로 시작
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    """평가 지표 계산 함수 (WER)"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 패딩 토큰(-100)을 tokenizer의 pad_token_id로 변경
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 예측 및 라벨 ID를 텍스트로 디코딩
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Word Error Rate 계산
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# --- 전역 변수 정의 ---
# 프로세서와 평가 지표는 map 함수 내에서도 접근 가능해야 하므로 전역으로 선언
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
wer_metric = evaluate.load("wer")


# --- 유틸: 양자화 로드 + 검증 루틴 ---------------------------------------------------------
def save_inference_metadata(adapter_dir: str, base_model_name: str, language: str, task: str) -> None:
    """인퍼런스 시 필요한 메타정보를 어댑터 폴더에 저장합니다."""
    try:
        if SAVE_INFERENCE_CONFIG:
            os.makedirs(adapter_dir, exist_ok=True)
            meta = {
                "base_model": base_model_name,
                "quantization": {"bitsandbytes": True, "load_in_8bit": True},
                "language": language,
                "task": task
            }
            with open(os.path.join(adapter_dir, "inference_config.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"경고: 인퍼런스 메타데이터 저장 실패: {e}")


def load_quantized_finetuned_model(adapter_dir: str):
    """
    저장된 LoRA 어댑터를 8bit 양자화된 베이스 모델에 적용하여 로드합니다.
    반환: (model, processor)
    """
    print("양자화(8-bit)된 베이스 모델과 어댑터를 로드합니다...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config = quantization_config,
        device_map = "auto"
    )
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    proc = WhisperProcessor.from_pretrained(MODEL_NAME, language = LANGUAGE, task = TASK)
    # 생성 시 한국어/전사 프롬프트를 강제하여 안정화
    try:
        peft_model.generation_config.forced_decoder_ids = proc.get_decoder_prompt_ids(language = LANGUAGE, task = TASK)
    except Exception:
        pass
    return peft_model, proc


def validate_saved_model(adapter_dir: str, raw_dataset: DatasetDict, max_items: int = VALIDATION_SAMPLES) -> None:
    """테스트셋 상위 N개로 간단 검증(WER 및 예시 출력)."""
    try:
        model, proc = load_quantized_finetuned_model(adapter_dir)
        device = next(model.parameters()).device
        print(f"검증 장치: {device}")

        test_ds = raw_dataset["test"]
        n = min(max_items, len(test_ds))
        preds, refs = [], []

        print(f"테스트셋 상위 {n}개 샘플로 검증합니다...")
        for i in tqdm(range(n), desc="검증 진행"):
            item = test_ds[i]
            audio_path = item["audio"]
            ref_text = item["sentence"].strip()
            try:
                speech_array, sr = librosa.load(audio_path, sr = 16000)
                inputs = proc(speech_array, sampling_rate = 16000, return_tensors = "pt").input_features
                # 모델 장치/자료형에 맞춰 입력 특징 캐스팅
                target_dtype = torch.float16 if device.type == "cuda" else torch.float32
                inputs = inputs.to(device = device, dtype = target_dtype)
                with torch.no_grad():
                    generated_ids = model.generate(inputs, max_new_tokens = 225)
                pred_text = proc.batch_decode(generated_ids, skip_special_tokens = True)[0]
            except Exception as e:
                pred_text = f"검증 오류: {e}"

            preds.append(pred_text)
            refs.append(ref_text)

            # 샘플 1-2개는 미리보기 출력
            if i < 2:
                print(f"\n[i={i}]\n- REF: {ref_text[:200]}\n- HYP: {pred_text[:200]}")

        # WER 계산
        wer_val = wer_metric.compute(predictions = preds, references = refs)
        print(f"\n[검증] 샘플 {n}개 기준 WER: {wer_val:.4f}")
    except Exception as e:
        print(f"검증 루틴 수행 중 오류: {e}")


if __name__ == "__main__":
    # --- 2. 데이터셋 로드 및 전처리 ---
    # 로컬 파일로부터 데이터셋 생성
    local_dataset = create_dataset_from_local_files()

    # --- 3. 모델, 프로세서, 토크나이저 로드 ---
    print("모델 및 프로세서를 로드합니다...")

    # 8-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" # 자동으로 GPU 할당
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # 모델을 k-bit 학습용으로 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    print("데이터셋 전처리를 시작합니다...")
    # num_proc를 사용하여 멀티프로세싱으로 전처리 속도 향상 (1로 설정하여 안정성 확보)
    tokenized_dataset = local_dataset.map(prepare_dataset, remove_columns=local_dataset.column_names["train"], num_proc=1)

    # --- 4. Trainer 설정 ---
    print("Trainer를 설정합니다...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=FP16,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False, # Hugging Face Hub에 업로드 시 True
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # --- 5. 학습 시작 ---
    print("모델 파인튜닝을 시작합니다...")
    trainer.train()

    # --- 6. 모델 저장 ---
    print("학습된 모델(어댑터)을 저장합니다...")
    model_path = os.path.join(OUTPUT_DIR, "best_lora_adapter")
    trainer.save_model(model_path)  # LoRA 어댑터 가중치 저장
    processor.save_pretrained(model_path)
    save_inference_metadata(model_path, MODEL_NAME, LANGUAGE, TASK)

    print("파인튜닝이 완료되었습니다.")
    print(f"LoRA 어댑터는 '{model_path}' 경로에 저장되었습니다.")

    # --- 7. 저장 모델 로드 후 간단 검증 (양자화 로딩 경로) ---
    if POST_TRAIN_VALIDATE:
        print("\n저장된 어댑터를 8-bit 경로로 로드하여 간단 검증을 수행합니다...")
        validate_saved_model(model_path, local_dataset, max_items = VALIDATION_SAMPLES)

# --- 7. 파인튜닝된 모델 사용 예시 ---
# from transformers import pipeline, WhisperForConditionalGeneration
# from peft import PeftModel

# # 기본 모델 로드 (양자화 적용)
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# base_model = WhisperForConditionalGeneration.from_pretrained(
#     MODEL_NAME,
#     quantization_config=quantization_config,
#     device_map="auto"
# )

# # LoRA 어댑터 적용
# peft_model = PeftModel.from_pretrained(base_model, "./whisper-medium-finetuned-ko/best_lora_adapter")
    
# processor = WhisperProcessor.from_pretrained("./whisper-medium-finetuned-ko/best_lora_adapter")

# pipe = pipeline("automatic-speech-recognition", model=peft_model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)
# result = pipe("path/to/your/audio.wav")
# print(result)

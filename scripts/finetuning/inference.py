"""
phishing_project.inference

보이스피싱 탐지를 위한 하이브리드 앙상블 추론 모듈.

탐지 아키텍처:
    1. PLM (KoELECTRA): 패턴 기반 1차 분류 → PLM_risk_score
    2. LLM (Gemma): 시스템 프롬프트 기반 심층 분석
       - 시나리오 일치 점수: 알려진 보이스피싱 패턴과의 일치도
       - 맥락 위험 점수: 신종 사기 패턴 탐지
       → LLM_risk_score (JSON 응답에서 파싱)
    3. 앙상블: PLM과 LLM 점수를 가중 평균하여 comprehensive_risk_score 산출

주요 함수:
    - load_plm(): PLM(ELECTRA 계열) 모델 로드
    - predict_plm(): 텍스트에 대한 PLM 피싱 확률 반환
    - load_llm(): LLM(Gemma + PEFT adapter) 로드
    - generate_llm(): LLM 텍스트 생성
    - analyze_and_ensemble(): PLM + LLM 앙상블 분석 (동기)
    - start_async_analysis(): PLM 즉시 + LLM 백그라운드 분석 (비동기)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import threading
import uuid
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel

# ========== 로깅 설정 ==========
# 모듈 레벨 로거 설정 (INFO 레벨 이상 출력)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# ==============================

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"


def _get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Models:
    """로딩된 모델을 보관하는 간단한 컨테이너"""

    def __init__(self):
        self.plm_model = None
        self.plm_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.device = _get_device()


MODELS = Models()

# Async run registry for background LLM calls
ASYNC_RUNS: dict = {}


def load_plm(plm_path: str = None) -> None:
    """PLM(분류기)와 토크나이저를 로드합니다.

    Args:
        plm_path: 로컬 디렉터리 또는 허깅페이스 모델 식별자
    Raises:
        FileNotFoundError: 로컬 경로가 지정되었고 파일/디렉터리가 존재하지 않을 때
        Exception: transformers 로드 중 발생 에러
    """
    if plm_path is None:
        plm_path = str(MODELS_DIR / "koelectra_finetuned")
    
    if os.path.exists(plm_path) is False and not plm_path.startswith("http"):
        raise FileNotFoundError(f"PLM 경로를 찾을 수 없습니다: {plm_path}")

    MODELS.plm_tokenizer = AutoTokenizer.from_pretrained(plm_path)
    MODELS.plm_model = AutoModelForSequenceClassification.from_pretrained(plm_path)
    MODELS.plm_model.eval()
    MODELS.device = _get_device()
    MODELS.plm_model.to(MODELS.device)


def predict_plm(text: str) -> float:
    """텍스트에 대해 PLM의 '피싱' 확률을 반환합니다(0.0 ~ 1.0).

    구현 세부:
    - 모델 출력 로짓에 softmax를 적용하여 확률로 변환합니다.
    - 프로젝트에서 라벨 1을 'phishing'으로 가정합니다 (compare_all_model.py 참조).

    Returns:
        phishing_prob: float
    Raises:
        RuntimeError: PLM이 로드되지 않은 경우
    """
    if MODELS.plm_model is None or MODELS.plm_tokenizer is None:
        raise RuntimeError("PLM 모델이 로드되어 있지 않습니다. 먼저 load_plm()을 호출하세요.")

    inputs = MODELS.plm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # move inputs to device
    inputs = {k: v.to(MODELS.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = MODELS.plm_model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu()

    # ========== PLM 원본 입력/출력 로깅 (print + logger) ==========
    print("=" * 60)
    print("[PLM RAW INPUT]")
    print(text)
    print("-" * 60)
    print("[PLM RAW OUTPUT]")
    print(f"  원본 Logits: {logits.cpu().tolist()}")
    print(f"  Softmax 확률 분포: {probs.tolist()}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("[PLM RAW INPUT]")
    logger.info(text)
    logger.info("-" * 60)
    logger.info("[PLM RAW OUTPUT]")
    logger.info(f"  원본 Logits: {logits.cpu().tolist()}")
    logger.info(f"  Softmax 확률 분포: {probs.tolist()}")
    logger.info("=" * 60)
    # =============================================================

    # 안전하게 처리: 2개 클래스가 아니더라도 인덱스 1을 phishing으로 취급
    if probs.ndim == 0:
        # single value (degenerate) -> return 0.0
        return float(probs.item())

    # Ensure length
    if probs.shape[0] == 1:
        phishing_prob = float(probs[0].item())
    else:
        # label 1 -> phishing
        phishing_prob = float(probs[1].item())

    # 정규화(softmax 결과이므로 0~1 범위)
    phishing_prob = max(0.0, min(1.0, phishing_prob))
    
    print(f"[PLM FINAL] 피싱 확률: {phishing_prob:.6f} (label 1)")
    logger.info(f"[PLM FINAL] 피싱 확률: {phishing_prob:.6f} (label 1)")
    
    return phishing_prob


def load_llm(
    base_model: str = "google/gemma-2b-it",
    adapter_path: str = None,
    prefer_cuda: bool = True,
) -> None:
    """LLM(Gemma + PEFT adapter)와 토크나이저를 로드합니다.

    주의: Gemma 계열은 메모리 사용량이 큽니다. CUDA 사용 환경이면 `device_map='auto'`와 float16을 권장합니다.
    """
    if adapter_path is None:
        adapter_path = str(MODELS_DIR / "gemma_2b_finetuned")
    
    device = _get_device(prefer_cuda=prefer_cuda)

    # tokenizer (base)
    MODELS.llm_tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load base model
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if device.type == "cuda" else None,
            torch_dtype=torch.float16 if device.type == "cuda" else None,
        )
    except Exception as e:
        # fall back to cpu load
        base = AutoModelForCausalLM.from_pretrained(base_model)

    # Load PEFT adapter on top
    MODELS.llm_model = PeftModel.from_pretrained(base, adapter_path)
    MODELS.llm_model.eval()


def generate_llm(prompt: str, max_new_tokens: int = 256) -> str:
    """LLM으로부터 텍스트를 생성합니다. 반환값은 생성된 문자열(응답 부분만)입니다.

    Gemma 모델의 경우 전체 출력에서 프롬프트 부분을 제거하여 응답만 반환합니다.
    
    생성 파라미터 (Gemma 권장 설정):
    - repetition_penalty: 반복 출력 방지 (1.0 초과 시 반복 억제)
    - temperature: 출력 다양성 조절 (낮을수록 결정적)
    - top_p: nucleus sampling (확률 상위 p% 토큰만 샘플링)
    - do_sample: True여야 temperature, top_p 적용됨
    
    참고: https://huggingface.co/docs/transformers/main_classes/text_generation
    """
    if MODELS.llm_model is None or MODELS.llm_tokenizer is None:
        raise RuntimeError("LLM이 로드되어 있지 않습니다. 먼저 load_llm()을 호출하세요.")

    inputs = MODELS.llm_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(MODELS.llm_model.device) for k, v in inputs.items()}
    
    # 입력 토큰 수 저장 (응답 추출용)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        out = MODELS.llm_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            # 반복 출력 방지 (Gemma 필수 설정)
            # 참고: https://discuss.huggingface.co/t/reducing-unwanted-generation-in-gemma-3/148856
            repetition_penalty=1.15,      # 1.0 초과: 이미 생성된 토큰 반복 억제
            no_repeat_ngram_size=3,       # 3-gram 반복 금지
            # 샘플링 설정
            do_sample=True,               # 샘플링 활성화 (temperature, top_p 적용)
            temperature=0.7,              # 낮을수록 결정적, 높을수록 창의적
            top_p=0.9,                    # nucleus sampling
            top_k=50,                     # top-k sampling
            # 종료 조건
            pad_token_id=MODELS.llm_tokenizer.eos_token_id,
            eos_token_id=MODELS.llm_tokenizer.eos_token_id,
        )
    
    # 생성된 토큰에서 입력 부분 제외하고 응답만 디코딩
    response_tokens = out[0][input_length:]
    raw_text = MODELS.llm_tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # ========== LLM(Gemma) 원본 입력/출력 로깅 (print + logger) ==========
    print("=" * 60)
    print("[LLM/GEMMA RAW INPUT]")
    print(prompt)
    print("-" * 60)
    print("[LLM/GEMMA RAW OUTPUT]")
    print(f"  입력 토큰 수: {input_length}")
    print(f"  생성 토큰 수: {len(response_tokens)}")
    print("-" * 60)
    print("[LLM 원본 응답 (전체)]:")
    print(raw_text)
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("[LLM/GEMMA RAW INPUT]")
    logger.info(prompt)
    logger.info("-" * 60)
    logger.info("[LLM/GEMMA RAW OUTPUT]")
    logger.info(f"  입력 토큰 수: {input_length}")
    logger.info(f"  생성 토큰 수: {len(response_tokens)}")
    logger.info("-" * 60)
    logger.info("[LLM 원본 응답 (전체)]:")
    logger.info(raw_text)
    logger.info("=" * 60)
    # ===================================================================
    
    # 코드 블록 마커 제거 (```json ... ``` 형태로 출력될 경우)
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    return text.strip()


def analyze_and_ensemble(
    conversation_text: str,
    system_prompt_path: str = "./system_ko.txt",
    max_new_tokens: int = 256,
) -> dict:
    """LLM(심층분석) 결과에서 comprehensive_risk_score을 파싱하고 PLM 확률과 평균을 내어 반환합니다.

    Returns a dict with keys:
      - comprehensive_risk_score: averaged score (LLM + PLM) / 2
      - reasoning: LLM의 reasoning (if available)
      - key_evidence: LLM의 key_evidence list (if available)
      - plm_prob: PLM 확률
      - llm_score: LLM이 반환한 종합 점수

    Note: LLM이 올바른 JSON을 반환하지 않으면 가능한 정보를 추출하여 진행합니다.
    """
    import json
    import re

    # 1) Ensure PLM is loaded
    if MODELS.plm_model is None or MODELS.plm_tokenizer is None:
        raise RuntimeError("PLM이 로드되어 있지 않습니다. load_plm()를 먼저 호출하세요.")

    # 2) Ensure LLM is loaded
    if MODELS.llm_model is None or MODELS.llm_tokenizer is None:
        raise RuntimeError("LLM이 로드되어 있지 않습니다. load_llm()를 먼저 호출하세요.")

    # 3) Read system prompt
    sys_prompt = ""
    try:
        if system_prompt_path and os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                sys_prompt = f.read()
    except Exception:
        sys_prompt = ""

    # 4) Build LLM prompt
    prompt_parts = []
    if sys_prompt:
        prompt_parts.append(sys_prompt)
    prompt_parts.append("\n### conversation_text:\n" + conversation_text)
    prompt = "\n\n".join(prompt_parts)

    # 5) Call LLM
    llm_resp_text = generate_llm(prompt, max_new_tokens=max_new_tokens)

    # 6) Try to extract JSON from LLM text
    llm_score = 0.0
    reasoning = ""
    key_evidence = []
    try:
        # find first {...} JSON block
        m = re.search(r"\{.*\}", llm_resp_text, flags=re.S)
        if m:
            jtxt = m.group(0)
            parsed = json.loads(jtxt)
            # Expected keys per system prompt
            llm_score = float(parsed.get("comprehensive_risk_score", 0.0))
            reasoning = parsed.get("reasoning", "")
            key_evidence = parsed.get("key_evidence", []) or []
        else:
            # Fallback: try to parse a float in text
            m2 = re.search(r"([0-1](?:\.\d{1,4})?)", llm_resp_text)
            if m2:
                llm_score = float(m2.group(1))
            reasoning = llm_resp_text.strip()
            key_evidence = []
    except Exception:
        # best-effort fallback
        llm_score = 0.0
        reasoning = llm_resp_text.strip()
        key_evidence = []

    # clamp
    llm_score = max(0.0, min(1.0, llm_score))

    # 7) Call PLM
    plm_prob = predict_plm(conversation_text)

    # 8) Ensemble: simple average
    final_score = float((llm_score + plm_prob) / 2.0)

    return {
        "PLM_risk_score": float(plm_prob),
        "LLM_risk_score": float(llm_score),
        "comprehensive_risk_score": final_score,
        "reasoning": reasoning,
        "key_evidence": key_evidence,
    }


def assemble_from_llm_output(llm_output, conversation_text: str) -> dict:
    """이미 생성된 LLM 출력(문자열 JSON 또는 dict)을 받아 PLM과 앙상블하여 최종 포맷으로 반환합니다.

    llm_output: JSON 문자열 또는 이미 파싱된 dict. 기대 가능한 키:
      - comprehensive_risk_score (float, 0~1)
      - reasoning (str)
      - key_evidence (list)

    반환: dict with keys matching 사용자 요청:
      PLM_risk_score, LLM_risk_score, comprehensive_risk_score, reasoning, key_evidence
    """
    import json
    import re

    # Normalize llm_output to dict
    parsed = None
    if isinstance(llm_output, dict):
        parsed = llm_output
    elif isinstance(llm_output, str):
        # try to find JSON block
        m = re.search(r"\{.*\}", llm_output, flags=re.S)
        try:
            if m:
                parsed = json.loads(m.group(0))
            else:
                parsed = json.loads(llm_output)
        except Exception:
            # best-effort: treat entire string as reasoning and try to extract number
            parsed = {}
            parsed["reasoning"] = llm_output.strip()
            m2 = re.search(r"([0-1](?:\.\d{1,4})?)", llm_output)
            if m2:
                parsed["comprehensive_risk_score"] = float(m2.group(1))

    if parsed is None:
        raise ValueError("LLM 출력 파싱 실패")

    llm_score = float(parsed.get("comprehensive_risk_score", 0.0))
    reasoning = parsed.get("reasoning", "")
    key_evidence = parsed.get("key_evidence", []) or []

    # ensure PLM loaded
    if MODELS.plm_model is None or MODELS.plm_tokenizer is None:
        raise RuntimeError("PLM이 로드되어 있지 않습니다. load_plm()을 먼저 호출하세요.")

    plm_prob = predict_plm(conversation_text)

    final_score = float((llm_score + plm_prob) / 2.0)

    return {
        "PLM_risk_score": float(plm_prob),
        "LLM_risk_score": float(llm_score),
        "comprehensive_risk_score": float(final_score),
        "reasoning": reasoning,
        "key_evidence": key_evidence,
    }


def _llm_worker(run_id: str, conversation_text: str, system_prompt_path: str, max_new_tokens: int):
    """백그라운드 스레드에서 LLM을 호출하고 ASYNC_RUNS를 갱신합니다."""
    try:
        print(f"[LLM Worker] 시작: run_id={run_id}, text_len={len(conversation_text)}")
        logger.info(f"[LLM Worker] 시작: run_id={run_id}, text_len={len(conversation_text)}")
        ASYNC_RUNS[run_id]["status"] = "processing_llm"

        # Ensure LLM is loaded
        if MODELS.llm_model is None or MODELS.llm_tokenizer is None:
            print("[LLM Worker] LLM 모델 로드 시작...")
            logger.info("[LLM Worker] LLM 모델 로드 시작...")
            try:
                load_llm()
                print("[LLM Worker] LLM 모델 로드 완료")
                logger.info("[LLM Worker] LLM 모델 로드 완료")
            except Exception as e:
                print(f"[LLM Worker] LLM 로드 실패: {e}")
                logger.error(f"[LLM Worker] LLM 로드 실패: {e}")
                ASYNC_RUNS[run_id]["status"] = "error"
                ASYNC_RUNS[run_id]["error"] = f"LLM 로드 실패: {e}"
                return

        # read system prompt
        sys_prompt = ""
        try:
            if system_prompt_path and os.path.exists(system_prompt_path):
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    sys_prompt = f.read()
        except Exception:
            sys_prompt = ""

        # Gemma 모델용 프롬프트 구성
        # 형식: <start_of_turn>user\n{내용}<end_of_turn>\n<start_of_turn>model\n
        user_content = f"{sys_prompt}\n\n### 분석할 대화:\n{conversation_text}\n\n위 대화를 분석하고 JSON 형식으로만 응답하세요."
        prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
        
        print(f"[LLM Worker] 프롬프트 길이: {len(prompt)} chars")
        logger.info(f"[LLM Worker] 프롬프트 길이: {len(prompt)} chars")

        print(f"[LLM Worker] LLM 생성 시작...")
        logger.info(f"[LLM Worker] LLM 생성 시작...")
        llm_resp_text = generate_llm(prompt, max_new_tokens=max_new_tokens)
        print(f"[LLM Worker] LLM 생성 완료: {len(llm_resp_text)} chars")
        logger.info(f"[LLM Worker] LLM 생성 완료: {len(llm_resp_text)} chars")
        
        # ========== LLM Worker에서 원본 출력 전체 로깅 ==========
        print("=" * 60)
        print("[LLM WORKER - FULL RAW RESPONSE]")
        print(llm_resp_text)
        print("=" * 60)
        
        logger.info("=" * 60)
        logger.info("[LLM WORKER - FULL RAW RESPONSE]")
        logger.info(llm_resp_text)
        logger.info("=" * 60)
        # ======================================================

        # JSON 파싱
        import json, re
        llm_score = 0.0
        reasoning = ""
        key_evidence = []
        
        # 전처리: 코드 블록 마커 제거
        clean_text = llm_resp_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        print(f"[LLM Worker] 정제된 텍스트 (처음 300자): {clean_text[:300]}")
        logger.info(f"[LLM Worker] 정제된 텍스트 (처음 300자): {clean_text[:300]}")
        
        try:
            # 방법 1: 전체 텍스트가 JSON인 경우
            if clean_text.startswith("{"):
                try:
                    parsed = json.loads(clean_text)
                    llm_score = float(parsed.get("comprehensive_risk_score", 0.0))
                    reasoning = parsed.get("reasoning", "")
                    key_evidence = parsed.get("key_evidence", []) or []
                    print(f"[LLM Worker] JSON 직접 파싱 성공: llm_score={llm_score}")
                    logger.info(f"[LLM Worker] JSON 직접 파싱 성공: llm_score={llm_score}")
                except json.JSONDecodeError:
                    # JSON이 잘렸을 수 있음 - 정규식으로 추출 시도
                    pass
            
            # 방법 2: 텍스트 내에서 JSON 블록 추출
            if llm_score == 0.0:
                # comprehensive_risk_score를 포함한 JSON 패턴 찾기
                m = re.search(r'\{[^{}]*"comprehensive_risk_score"[^{}]*\}', clean_text, flags=re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        llm_score = float(parsed.get("comprehensive_risk_score", 0.0))
                        reasoning = parsed.get("reasoning", "")
                        key_evidence = parsed.get("key_evidence", []) or []
                        print(f"[LLM Worker] JSON 정규식 파싱 성공: llm_score={llm_score}")
                        logger.info(f"[LLM Worker] JSON 정규식 파싱 성공: llm_score={llm_score}")
                    except json.JSONDecodeError as e:
                        print(f"[LLM Worker] JSON 파싱 실패 (정규식): {e}")
                        logger.warning(f"[LLM Worker] JSON 파싱 실패 (정규식): {e}")
            
            # 방법 3: comprehensive_risk_score 값만 직접 추출
            if llm_score == 0.0:
                score_match = re.search(r'"comprehensive_risk_score"\s*:\s*([\d.]+)', clean_text)
                if score_match:
                    llm_score = float(score_match.group(1))
                    print(f"[LLM Worker] 점수 직접 추출 성공: llm_score={llm_score}")
                    logger.info(f"[LLM Worker] 점수 직접 추출 성공: llm_score={llm_score}")
                
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', clean_text)
                if reasoning_match:
                    reasoning = reasoning_match.group(1)
            
            # 방법 4: 아무것도 안 되면 원본 텍스트를 reasoning으로
            if llm_score == 0.0 and not reasoning:
                reasoning = clean_text[:500] if clean_text else llm_resp_text[:500]
                print(f"[LLM Worker] JSON 파싱 완전 실패, 원본 텍스트 사용")
                logger.warning(f"[LLM Worker] JSON 파싱 완전 실패, 원본 텍스트 사용")
                
        except Exception as parse_err:
            llm_score = 0.0
            reasoning = clean_text[:500] if clean_text else llm_resp_text[:500]
            key_evidence = []
            print(f"[LLM Worker] JSON 파싱 예외: {parse_err}")
            logger.error(f"[LLM Worker] JSON 파싱 예외: {parse_err}")

        llm_score = max(0.0, min(1.0, llm_score))

        # combine with PLM
        plm_prob = ASYNC_RUNS[run_id].get("PLM_risk_score", 0.0)
        final_score = float((llm_score + plm_prob) / 2.0)
        
        print(f"[LLM Worker] 앙상블 완료: PLM={plm_prob:.4f}, LLM={llm_score:.4f}, final={final_score:.4f}")
        logger.info(f"[LLM Worker] 앙상블 완료: PLM={plm_prob:.4f}, LLM={llm_score:.4f}, final={final_score:.4f}")

        # update run using requested keys
        ASYNC_RUNS[run_id].update({
            "status": "complete",
            "LLM_risk_score": float(llm_score),
            "PLM_risk_score": float(plm_prob),
            "comprehensive_risk_score": float(final_score),
            "reasoning": reasoning,
            "key_evidence": key_evidence,
            "llm_text": llm_resp_text,
        })
        print(f"[LLM Worker] 완료: run_id={run_id}")
        logger.info(f"[LLM Worker] 완료: run_id={run_id}")

    except Exception as e:
        print(f"[LLM Worker] 에러: {e}")
        logger.error(f"[LLM Worker] 에러: {e}")
        ASYNC_RUNS[run_id]["status"] = "error"
        ASYNC_RUNS[run_id]["error"] = str(e)


def start_async_analysis(
    conversation_text: str,
    system_prompt_path: str = "./system_ko.txt",
    max_new_tokens: int = 256,
    run_id: Optional[str] = None,
) -> str:
    """PLM은 즉시 실행하여 확률을 반환하고, LLM은 백그라운드에서 실행하도록 스케줄합니다.

    반환값: run_id (클라이언트가 이후 `get_async_status(run_id)`로 상태를 조회할 수 있음)
    """
    print(f"[start_async_analysis] 호출됨: text_len={len(conversation_text)}, run_id={run_id}")
    logger.info(f"[start_async_analysis] 호출됨: text_len={len(conversation_text)}, run_id={run_id}")
    
    if MODELS.plm_model is None or MODELS.plm_tokenizer is None:
        print("[start_async_analysis] PLM이 로드되어 있지 않습니다!")
        logger.error("[start_async_analysis] PLM이 로드되어 있지 않습니다!")
        raise RuntimeError("PLM이 로드되어 있지 않습니다. load_plm()을 먼저 호출하세요.")

    if run_id is None:
        run_id = uuid.uuid4().hex
        print(f"[start_async_analysis] 새 run_id 생성: {run_id}")
        logger.info(f"[start_async_analysis] 새 run_id 생성: {run_id}")

    # compute PLM prob immediately
    print(f"[start_async_analysis] PLM 예측 시작...")
    logger.info(f"[start_async_analysis] PLM 예측 시작...")
    plm_prob = predict_plm(conversation_text)
    print(f"[start_async_analysis] PLM 예측 완료: {plm_prob:.4f}")
    logger.info(f"[start_async_analysis] PLM 예측 완료: {plm_prob:.4f}")

    # init run entry (use requested key names)
    ASYNC_RUNS[run_id] = {
        "status": "processing_plm",
        "PLM_risk_score": float(plm_prob),
        "LLM_risk_score": None,
        "comprehensive_risk_score": None,
        "reasoning": "",
        "key_evidence": [],
        "llm_text": "",
        "created_at": time.time(),
        "conversation_text": conversation_text,
    }
    print(f"[start_async_analysis] ASYNC_RUNS에 등록 완료")
    logger.info(f"[start_async_analysis] ASYNC_RUNS에 등록 완료")

    # start background LLM worker
    print(f"[start_async_analysis] LLM 워커 스레드 시작...")
    logger.info(f"[start_async_analysis] LLM 워커 스레드 시작...")
    thread = threading.Thread(target=_llm_worker, args=(run_id, conversation_text, system_prompt_path, max_new_tokens), daemon=True)
    thread.start()
    print(f"[start_async_analysis] LLM 워커 스레드 시작됨: run_id={run_id}")
    logger.info(f"[start_async_analysis] LLM 워커 스레드 시작됨: run_id={run_id}")

    return run_id


def get_async_status(run_id: str) -> dict:
    """주어진 run_id의 현재 상태와 결과를 반환합니다."""
    entry = ASYNC_RUNS.get(run_id)
    if entry is None:
        raise KeyError(f"Run not found: {run_id}")
    # return a shallow copy to avoid external mutation
    return dict(entry)


if __name__ == "__main__":
    # 간단한 스모크 테스트(모델 파일이 실제로 존재하면 느리게 실행될 수 있음)
    print("phishing_project.inference 스모크 테스트")
    print("-- PLM 로드 시도 (경로: ./phishing_project/fine-tuned-phishing-model)")
    try:
        load_plm()
        print("PLM 로드 성공")
        sample = "안녕하세요, 귀하의 계좌에 이상한 거래가 감지되었습니다. 확인을 위해 연락처를 알려주세요."
        p = predict_plm(sample)
        print(f"샘플 PLM phishing 확률: {p:.4f}")
    except Exception as e:
        print(f"PLM 로드/예측 스모크 실패: {e}")

    print("LLM 로드는 별도로 수행하세요 (대형 모델 로드 비용 주의)")

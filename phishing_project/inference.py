"""
phishing_project.inference

모델 로드 및 예측 유틸리티.

현재 구현 범위:
- PLM(ELECTRA 계열) 로드 및 확률 예측 함수 `predict_plm`
- LLM(Gemma + PEFT adapter) 로드 및 생성 함수 `generate_llm`

시스템 프롬프트(system_prompt)는 호출자가 제공하도록 했으며 기본값은 빈 문자열입니다.
LLM의 출력 확률(점수) 산정 방식은 추후 설계하므로 현재는 생성 텍스트만 반환합니다.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import threading
import uuid
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel


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


def load_plm(plm_path: str = "./phishing_project/fine-tuned-phishing-model") -> None:
    """PLM(분류기)와 토크나이저를 로드합니다.

    Args:
        plm_path: 로컬 디렉터리 또는 허깅페이스 모델 식별자
    Raises:
        FileNotFoundError: 로컬 경로가 지정되었고 파일/디렉터리가 존재하지 않을 때
        Exception: transformers 로드 중 발생 에러
    """
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
    return phishing_prob


def load_llm(
    base_model: str = "google/gemma-2b-it",
    adapter_path: str = "./phishing_project/gemma-phishing-adapter",
    prefer_cuda: bool = True,
) -> None:
    """LLM(Gemma + PEFT adapter)와 토크나이저를 로드합니다.

    주의: Gemma 계열은 메모리 사용량이 큽니다. CUDA 사용 환경이면 `device_map='auto'`와 float16을 권장합니다.
    """
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


def generate_llm(prompt: str, max_new_tokens: int = 128) -> str:
    """LLM으로부터 텍스트를 생성합니다. 반환값은 생성된 문자열입니다.

    시스템 프롬프트는 외부에서 관리하도록 빈 문자열을 기본으로 합니다.
    """
    if MODELS.llm_model is None or MODELS.llm_tokenizer is None:
        raise RuntimeError("LLM이 로드되어 있지 않습니다. 먼저 load_llm()을 호출하세요.")

    inputs = MODELS.llm_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(MODELS.llm_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = MODELS.llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = MODELS.llm_tokenizer.decode(out[0], skip_special_tokens=True)
    return text


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
        "LLM_risk_socre": float(llm_score),
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
      PLM_risk_score, LLM_risk_socre, comprehensive_risk_score, reasoning, key_evidence
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
        "LLM_risk_socre": float(llm_score),
        "comprehensive_risk_score": float(final_score),
        "reasoning": reasoning,
        "key_evidence": key_evidence,
    }


def _llm_worker(run_id: str, conversation_text: str, system_prompt_path: str, max_new_tokens: int):
    """백그라운드 스레드에서 LLM을 호출하고 ASYNC_RUNS를 갱신합니다."""
    try:
        ASYNC_RUNS[run_id]["status"] = "processing_llm"

        # Ensure LLM is loaded
        if MODELS.llm_model is None or MODELS.llm_tokenizer is None:
            try:
                load_llm()
            except Exception as e:
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

        prompt_parts = []
        if sys_prompt:
            prompt_parts.append(sys_prompt)
        prompt_parts.append("\n### conversation_text:\n" + conversation_text)
        prompt = "\n\n".join(prompt_parts)

        llm_resp_text = generate_llm(prompt, max_new_tokens=max_new_tokens)

        # parse LLM JSON like analyze_and_ensemble
        import json, re
        llm_score = 0.0
        reasoning = ""
        key_evidence = []
        try:
            m = re.search(r"\{.*\}", llm_resp_text, flags=re.S)
            if m:
                parsed = json.loads(m.group(0))
                llm_score = float(parsed.get("comprehensive_risk_score", 0.0))
                reasoning = parsed.get("reasoning", "")
                key_evidence = parsed.get("key_evidence", []) or []
            else:
                m2 = re.search(r"([0-1](?:\.\d{1,4})?)", llm_resp_text)
                if m2:
                    llm_score = float(m2.group(1))
                reasoning = llm_resp_text.strip()
        except Exception:
            llm_score = 0.0
            reasoning = llm_resp_text.strip()
            key_evidence = []

        llm_score = max(0.0, min(1.0, llm_score))

        # combine with PLM
        plm_prob = ASYNC_RUNS[run_id].get("PLM_risk_score", 0.0)
        final_score = float((llm_score + plm_prob) / 2.0)

        # update run using requested keys
        ASYNC_RUNS[run_id].update({
            "status": "complete",
            "LLM_risk_socre": float(llm_score),
            "PLM_risk_score": float(plm_prob),
            "comprehensive_risk_score": float(final_score),
            "reasoning": reasoning,
            "key_evidence": key_evidence,
            "llm_text": llm_resp_text,
        })

    except Exception as e:
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
    if MODELS.plm_model is None or MODELS.plm_tokenizer is None:
        raise RuntimeError("PLM이 로드되어 있지 않습니다. load_plm()을 먼저 호출하세요.")

    if run_id is None:
        run_id = uuid.uuid4().hex

    # compute PLM prob immediately
    plm_prob = predict_plm(conversation_text)

    # init run entry (use requested key names)
    ASYNC_RUNS[run_id] = {
        "status": "processing_plm",
        "PLM_risk_score": float(plm_prob),
        "LLM_risk_socre": None,
        "comprehensive_risk_score": None,
        "reasoning": "",
        "key_evidence": [],
        "llm_text": "",
        "created_at": time.time(),
        "conversation_text": conversation_text,
    }

    # start background LLM worker
    thread = threading.Thread(target=_llm_worker, args=(run_id, conversation_text, system_prompt_path, max_new_tokens), daemon=True)
    thread.start()

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

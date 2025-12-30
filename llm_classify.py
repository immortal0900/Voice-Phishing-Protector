"""보이스피싱 텍스트 분류기

입력: 전사된 한국어 통화 텍스트(.txt)
동작: LLM(OpenAI 또는 Ollama)을 호출해 '보이스피싱 여부'와 근거를 판단
출력: per-file JSON, 전체 CSV 요약

사용 예(폴더 전체):
  uv run python .\llm_classify.py .\fw_stt_voice_files --model openai:gpt-4o-mini --sys-prompt system_ko.txt
  uv run python .\llm_classify.py .\fw_stt_voice_files --model ollama:llama3.1:8b --sys-prompt system_ko.txt

사용 예(단일 파일):
  uv run python .\llm_classify.py .\fw_stt_voice_files\sample.txt --model openai:gpt-4o-mini --sys-prompt system_ko.txt

환경 변수:
  OPENAI_API_KEY: OpenAI 사용 시 필요
  OLLAMA_BASE_URL: Ollama 서버 URL(기본 http://127.0.0.1:11434)
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


SUPPORTED_EXT = {".txt"}


def list_txt_files(path: str) -> List[str]:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        )
    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED_EXT:
        return [path]
    return []


def read_text_file(path: str) -> str:
    with open(path, "r", encoding = "utf-8") as f:
        return f.read().strip()


def load_system_prompt(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding = "utf-8") as f:
        return f.read()


class LLMClient:
    def __init__(self, model_spec: str, temperature: float = 0.2, base_url: Optional[str] = None):
        self.model_spec = model_spec
        self.temperature = float(temperature)
        self.base_url = base_url

        if model_spec.startswith("openai:"):
            from openai import OpenAI  # type: ignore
            self.kind = "openai"
            self.model = model_spec.split(":", 1)[1]
            self.client = OpenAI()
        elif model_spec.startswith("ollama:"):
            import requests  # type: ignore
            self.kind = "ollama"
            self.model = model_spec.split(":", 1)[1]
            self.session = requests.Session()
            self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        else:
            raise ValueError("model_spec는 'openai:<model>' 또는 'ollama:<model>' 형식이어야 합니다")

    def chat(self, system_prompt: Optional[str], user_prompt: str, max_tokens: int = 512) -> str:
        if self.kind == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            resp = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = self.temperature,
                max_tokens = max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        elif self.kind == "ollama":
            payload = {
                "model": self.model,
                "messages": (
                    ([{"role": "system", "content": system_prompt}] if system_prompt else [] )
                    + [{"role": "user", "content": user_prompt}]
                ),
                "options": {"temperature": self.temperature},
                "stream": False,
            }
            r = self.session.post(f"{self.base_url}/api/chat", json = payload, timeout = 120)
            r.raise_for_status()
            data = r.json()
            # Ollama 응답 구조에서 마지막 메시지 추출
            msg = data.get("message", {}).get("content")
            if not msg and isinstance(data.get("messages"), list) and data["messages"]:
                msg = data["messages"][-1].get("content")
            return (msg or "").strip()

        else:
            raise RuntimeError("Unknown LLM client kind")


def build_user_prompt(text: str) -> str:
    # 시스템 프롬프트는 별도 파일로 주입. 사용자 프롬프트는 데이터와 형식을 명확히.
    return (
        """
            # ROLE
            당신은 금융 사기, 특히 '보이스피싱' 탐지에 특화된 AI 분석가입니다. 당신의 임무는 실시간으로 변환된 통화 텍스트를 분석하여 대화의 위험성을 판단하고, 그 결과를 정해진 형식에 따라 출력하는 것입니다.

            ---

            # CONTEXT
            당신이 분석할 텍스트는 STT(Speech-to-Text)를 통해 변환된 원본(raw) 파일이며, 다음과 같은 특징을 가집니다.

            1.  **발화자 미구분**: 누가 말했는지 표시되어 있지 않습니다. 대화의 흐름과 문맥을 통해 각 발언의 주체를 스스로 추론해야 합니다.
            2.  **오타 및 오류 포함**: STT 변환 과정에서 발생한 오타나 잘못된 단어가 포함될 수 있습니다. 의미를 파악할 때 이를 감안하여 유연하게 해석해야 합니다.

            ---

            # OBJECTIVE
            주어진 통화 텍스트를 분석하여 아래 두 가지 중 하나로 분류하고, 위험 대화일 경우 구체적인 위험도를 측정해야 합니다.

            1.  **일상 대화 (Normal Conversation)**: 일반적인 안부, 약속, 정보 공유 등 범죄 혐의가 없는 모든 대화.
            2.  **보이스피싱 위험 대화 (Voice Phishing Risk)**: 금전이나 개인정보를 탈취하려는 목적이 의심되는 대화.

            ---

            # RISK ASSESSMENT CRITERIA
            '보이스피싱 위험 대화'의 위험도는 아래 항목들을 종합적으로 평가하여 **총 100점 만점**으로 계산합니다. 위험도 **임계점은 50점**입니다.

            1.  **권위 사칭 (25점)**: 검찰, 경찰, 금융감독원, 은행 직원 등 공신력 있는 기관이나 인물을 사칭하는가?
            2.  **긴급성 및 압박 (25점)**: "지금 당장", "시간이 없다", "오늘 처리해야 한다" 등 즉각적인 행동을 강요하며 심리적으로 압박하는가?
            3.  **금전/개인정보 요구 (30점)**: 계좌이체, 현금 전달, 상품권 구매를 직접적으로 요구하거나, 계좌번호, 비밀번호, 인증번호 등 민감한 금융 정보를 물어보는가?
            4.  **비밀 유지 및 협박 (20점)**: "가족에게도 말하면 안 된다", "주변에 알리면 당신이 공범이 된다", "구속될 수 있다" 등 비밀 유지를 강요하거나 불안감을 조성하는가?

            ---

            # INSTRUCTIONS
            아래 단계에 따라 작업을 순서대로 수행하세요.

            1.  **전체 대화 분석**: 발화자가 구분되지 않은 텍스트 전체를 읽고, 문맥을 파악하여 대화의 상황과 흐름을 이해합니다.
            2.  **1차 분류**: 대화가 '일상 대화'인지 '보이스피싱 위험'의 가능성이 있는지 판단합니다.
            3.  **위험도 채점 (필요시)**: '보이스피싱 위험' 가능성이 있다면, 상단의 `RISK ASSESSMENT CRITERIA`에 따라 각 항목의 점수를 채점하여 총점을 계산합니다. '일상 대화'의 경우 위험도는 0점입니다.
            4.  **최종 결과 생성**: 채점된 위험도가 임계점(50점)을 넘는지 확인하고, 아래 `OUTPUT FORMAT`에 맞춰 최종 결과를 생성합니다.

            ---

            # OUTPUT FORMAT
            결과는 반드시 다음 JSON 형식으로만 출력해야 합니다. 어떤 경우에도 추가적인 설명이나 일반 텍스트를 JSON 외부에 포함해서는 안 됩니다.

            ### 출력 예시 1: 일상 대화

            json
            {
              "classification": "일상 대화",
              "risk_score": 0,
              "is_above_threshold": false,
              "reasoning": "범죄 혐의점이 없는 일반적인 약속 및 안부 대화임."
            }
        """
    )


def parse_json_response(s: str) -> Tuple[bool, float, str]:
    # 모델이 JSON을 반환하지 않을 가능성도 있어 유연 파싱
    try:
        data = json.loads(s)
        # 1) 신규 스키마: classification, risk_score, is_above_threshold, reasoning
        if any(k in data for k in ("classification", "risk_score", "reasoning")):
            classification = str(data.get("classification", "")).strip()
            risk_score = float(data.get("risk_score", data.get("score", 0)))
            is_above = bool(data.get("is_above_threshold", risk_score >= 50))
            reasoning = str(data.get("reasoning", data.get("reason", "")).strip())
            # phishing 판정: 분류문자열에 '보이스피싱' 포함 또는 is_above true
            phishing = ("보이스피싱" in classification) or is_above or (risk_score >= 50)
            return phishing, risk_score, reasoning
        # 2) 기존 스키마: phishing, score, reason
        phishing = bool(data.get("phishing", False))
        score = float(data.get("score", 0))
        reason = str(data.get("reason", "")).strip()
        return phishing, score, reason
    except Exception:
        # fallback: 간단 휴리스틱
        low = s.lower()
        phishing = ("phishing" in low) or ("사기" in s) or ("검찰" in s and "계좌" in s)  # 매우 단순
        score = 70.0 if phishing else 30.0
        reason = s[:300]
        return phishing, score, reason


def ensure_out_dir(path: str) -> str:
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok = True)
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description = "보이스피싱 LLM 분류기")
    parser.add_argument("input", help = "입력 .txt 파일 또는 폴더")
    parser.add_argument("--model", required = True, help = "openai:<model> 또는 ollama:<model>")
    parser.add_argument("--sys-prompt", help = "시스템 프롬프트 텍스트 파일 경로")
    parser.add_argument("--out-dir", default = "llm_classify_out", help = "출력 폴더")
    parser.add_argument("--temperature", type = float, default = 0.2, help = "샘플링 온도")
    parser.add_argument("--max-tokens", type = int, default = 512, help = "최대 토큰 수")
    args = parser.parse_args()

    files = list_txt_files(args.input)
    if not files:
        print("[오류] 입력에 .txt 파일이 없습니다")
        sys.exit(2)

    out_dir = ensure_out_dir(args.out_dir)
    system_prompt = load_system_prompt(args.sys_prompt)
    client = LLMClient(args.model, temperature = args.temperature)

    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", encoding = "utf-8", newline = "") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "phishing", "score", "reason"])

        for p in files:
            base = os.path.basename(p)
            text = read_text_file(p)
            user_prompt = build_user_prompt(text)
            try:
                reply = client.chat(system_prompt, user_prompt, max_tokens = args.max_tokens)
                phishing, score, reason = parse_json_response(reply)
            except Exception as e:
                phishing, score, reason = False, 0.0, f"LLM 호출 실패: {e}"

            # per-file JSON 저장
            out_json = {
                "file": base,
                "phishing": phishing,
                "score": score,
                "reason": reason,
                "model": args.model,
                "temperature": args.temperature,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(os.path.join(out_dir, f"{os.path.splitext(base)[0]}.json"), "w", encoding = "utf-8") as fj:
                json.dump(out_json, fj, ensure_ascii = False, indent = 2)

            writer.writerow([base, phishing, f"{score:.1f}", reason])

    print(f"완료: {out_dir} 에 summary.csv 및 per-file JSON 저장")


if __name__ == "__main__":
    main()

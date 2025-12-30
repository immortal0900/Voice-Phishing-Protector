# FastAPI 엔드포인트 상세 문서

이 문서는 `fastapi.py` 파일에 구현된 REST API 엔드포인트를 OpenAPI/Swagger 스타일로 자세히 설명합니다.

- 파일 경로: `fastapi.py`
- 생성일: 2025-09-23
- 작성자(자동): 프로젝트 분석 도구

---

## 요약
`fastapi.py`는 업로드된 오디오 파일을 받아 백그라운드 파이프라인(오디오 청크 분할 → STT → LLM 분류)을 실행하는 FastAPI 서버입니다. 각 실행은 고유한 `run_id`를 가지며, 상태/로그/결과는 파일 시스템(`streaming_runs/<run_id>/`)에 저장됩니다.

이 문서는 다음 엔드포인트를 다룹니다:
- `POST /pipeline/start`
- `POST /pipeline/stop`
- `GET /pipeline/status`
- `GET /pipeline/log`
- `GET /pipeline/result`
- `GET /pipeline/llm_input`

---

## 공통 주의사항
- 인증: 현재 엔드포인트에는 인증이 없습니다. 내부 네트워크 또는 안전한 환경에서만 노출하십시오.
- 파일 저장: 업로드되는 오디오는 `streaming_runs/<run_id>/uploaded<ext>`로 저장됩니다. 디스크 용량을 모니터링하세요.
- 동시 실행: 여러 `run`이 동시에 실행될 경우 GPU/메모리 경쟁이 발생할 수 있습니다.

---

## POST /pipeline/start

### 설명
업로드된 오디오 파일과 파라미터로 새로운 파이프라인 실행을 생성하고 백그라운드 스레드에서 실행을 시작합니다.

### 요청
- Content-Type: `multipart/form-data`
- 필드:
  - `file` (파일, 필수)
    - 설명: 업로드할 오디오 파일(예: `.wav`, `.mp3`, `.flac`, `.ogg`)
  - `fw_model` (string, 선택, 기본값: `small`)
    - 설명: Faster-Whisper 모델 이름(예: `tiny`, `base`, `small`, `medium`, `large-v3`)
  - `device` (string, 선택, 기본값: `cuda`)
    - 설명: 실행 디바이스 (`cuda` 또는 `cpu`)
  - `compute_type` (string, 선택, 기본값: `float16`)
    - 설명: 계산 타입(예: `float16`, `int8_float16`, `int8`)
  - `chunk_sec` (int, 선택, 기본값: `5`)
    - 설명: feeder가 생성할 청크 길이(초)
  - `simulate_realtime` (bool, 선택, 기본값: `true`)
    - 설명: feeder가 청크 사이에 지연을 두어 실시간 시뮬레이션을 수행할지 여부
  - `ollama_url` (string, 선택, 기본값: `http://127.0.0.1:11434`)
    - 설명: Ollama(LLM) 서버 URL
  - `gemma_model` (string, 선택, 기본값: `gemma2:9b`)
    - 설명: LLM 모델 식별자
  - `sys_prompt` (string, 선택)
    - 설명: LLM에 전달할 시스템 프롬프트(선택)

### 응답
- 성공(HTTP 200)
```json
{ "run_id": "<hex>" }
```
- 실패: HTTP 4xx/5xx 에러

### 부수 효과
- `streaming_runs/<run_id>/` 디렉터리 생성
- 업로드 파일 저장(`uploaded<ext>`)
- 초기 `status.txt` 및 `log.txt` 기록

### 예시 (Python requests)
```python
import requests
url = "http://127.0.0.1:8000/pipeline/start"
files = {"file": ("audio.wav", open("audio.wav","rb"), "application/octet-stream")}
data = {
  "fw_model": "small",
  "device": "cuda",
  "compute_type": "float16",
  "chunk_sec": "5",
  "simulate_realtime": "true",
  "ollama_url": "http://127.0.0.1:11434",
  "gemma_model": "gemma2:9b",
  "sys_prompt": "시스템 프롬프트 내용",
}
r = requests.post(url, files=files, data=data, timeout=60)
r.raise_for_status()
print(r.json())
```

---

## POST /pipeline/stop

### 설명
지정한 `run_id`에 대해 중지 요청을 보냅니다. 실제 스레드는 현재 작업을 마치거나 루프에서 중지 플래그를 감지하면 종료합니다.

### 요청
- Content-Type: `application/x-www-form-urlencoded` 또는 `multipart/form-data`
- 필드:
  - `run_id` (string, 필수)

### 응답
- 성공(HTTP 200)
```json
{ "ok": true }
```

### 부수 효과
- `status.txt`가 `중지 요청됨`으로 갱신
- `log.txt`에 `[INFO] Stop 요청`이 추가

### 예시 (Python requests)
```python
import requests
requests.post("http://127.0.0.1:8000/pipeline/stop", data={"run_id": run_id})
```

---

## GET /pipeline/status

### 설명
지정한 `run_id`의 상태 텍스트(`status.txt`)를 반환합니다.

### 요청
- 쿼리 파라미터:
  - `run_id` (string, 필수)

### 응답
- 성공: `text/plain` — 상태 문자열(예: `처리 시작: out_dir=...`, `처리 완료`, `에러: ...`)

### 예시
```
GET /pipeline/status?run_id=<run_id>
```

---

## GET /pipeline/log

### 설명
지정한 `run_id`의 로그 파일(`log.txt`) 끝부분을 반환합니다.

### 요청
- 쿼리 파라미터:
  - `run_id` (string, 필수)
  - `tail` (int, 선택, 기본값: `200`) — 마지막 N줄

### 응답
- 성공: `text/plain` — 로그 텍스트

### 예시
```
GET /pipeline/log?run_id=<run_id>&tail=200
```

---

## GET /pipeline/result

### 설명
지정한 `run_id`의 LLM/파이프라인 최종 결과 파일(`last_result.json`)을 반환합니다. 파일이 JSON으로 파싱 가능하면 JSON 응답을, 그렇지 않으면 plain text를 반환합니다.

### 요청
- 쿼리 파라미터:
  - `run_id` (string, 필수)

### 응답
- 성공:
  - `application/json` (가능한 경우)
  - 또는 `text/plain`

### 예시
```
GET /pipeline/result?run_id=<run_id>
```

---

## GET /pipeline/llm_input

### 설명
지정한 `run_id`의 `last_llm_input.txt` 내용을 반환합니다(가장 최근 LLM 입력).

### 요청
- 쿼리 파라미터:
  - `run_id` (string, 필수)

### 응답
- 성공: `text/plain`

### 예시
```
GET /pipeline/llm_input?run_id=<run_id>
```

---

## 파일/디렉터리 관련 상세 (부수 효과 요약)
- 루트 실행 시 생성되는 디렉터리: `streaming_runs/<run_id>/`
  - `uploaded<ext>`: 업로드한 원본 오디오
  - `log.txt`: 파이프라인 로그
  - `status.txt`: 상태 텍스트
  - `last_result.json`: LLM 최종 결과(가능하면 JSON)
  - `last_llm_input.txt`: LLM에 전달한 입력
  - feeder가 생성한 청크 오디오 파일들(예: `chunk_0001.wav`)

## 운영 및 보안 권장 사항
- 인증/인가 추가: 최소한 `Authorization` 토큰 기반의 간단한 보호를 권장합니다.
- 업로드 크기 제한: `POST /pipeline/start`에 파일 크기 제한(예: 200MB) 적용 권장.
- 디스크 관리: 오래된 `streaming_runs/` 자동 삭제(예: X일 후) 및 로그 회전 정책.
- 리소스 관리: 동시 실행 제한 또는 작업 큐(특히 GPU 사용률 관리).
- 개인정보: 업로드된 오디오/텍스트에 민감정보가 포함될 수 있으므로 외부 전송 주의.

---

## 개선 가능한 OpenAPI 보강 포인트 (권장)
- 각 엔드포인트에 Pydantic 스키마로 응답/요청 모델을 명시하면 Swagger UI에 더 친절하게 노출됩니다.
- `/pipeline/status`의 반환을 현재의 자유 텍스트 대신 JSON 구조(예: `{ "status": "processing", "phase": "stt", "message": "..." }`)로 바꾸면 UI/클라이언트 처리 및 상태 판단이 용이합니다.
- Swagger 문서화 시 `file` 필드(멀티파트)의 설명과 예시를 명시적으로 추가하세요.

---
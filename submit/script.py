#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반사항 모니터링 AI 경진대회 베이스라인.

평가 서버는 이 파일을 `python script.py`로 그대로 실행합니다.
  입력   ./data/test.jsonl.gz (+ 항목표.json · 정답스키마_디코딩.json)
  출력   ./output/submission.csv  (열 = id, v1..v24, e1..e24)
         v = 위반 여부 0/1, e = 근거 문구(원문 부분문자열, 비위반은 빈칸)
  경로   PPS_DATA_DIR · PPS_OUTPUT_DIR · PPS_MODEL_DIR 환경변수 우선

전체 흐름
  데이터 로드 → 프롬프트 구성 → vLLM 배치 추론 → JSON 파싱
  → 근거 문구 검증 → submission.csv 저장 → 형식 검증

로컬 실행
  python script.py --mock          # 모델 없이 입력·출력 흐름 확인
  python script.py --limit 10      # 앞 10건 실행
"""
from __future__ import annotations

# ===== 1. 상수·경로 =====
import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

DATA_DIR = os.environ.get("PPS_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("PPS_OUTPUT_DIR", "./output")
MODEL_DIR = os.environ.get("PPS_MODEL_DIR", "/opt/models/gemma-4-26B-A4B-it")

ITEMS = [f"v{i}" for i in range(1, 25)]
EVID = [f"e{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + EVID
ABSENCE = ["v10", "v11", "v16", "v18", "v20"]          # 부재탐지 항목: 근거 문구 빈칸

DOC_ORDER = ["공고문", "규격서", "과업지시서", "제안요청서", "예외공표서", "기타"]
META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]

SEED = 20260826
MAX_MODEL_LEN = 16384                   # 1차(32768·60,000자)에서 점수가 내려가 베이스라인 길이로 되돌림
MAX_TOKENS = 3072                       # 구조화 출력 토큰 예산 (1536에서는 근거가 길면 JSON이 잘림)
PROMPT_BUDGET = MAX_MODEL_LEN - MAX_TOKENS
EVIDENCE_MAX = 500                      # 근거 문구 셀 글자 수 상한
QUANT = "int8_per_channel_weight_only"  # 평가 서버 양자화 설정
MAX_CHARS = 16000                       # 문서 글자 수 초기 상한 (베이스라인 4000 · 공고 중앙값 13k자)

# 시간 가드: 평가 서버 실행 제한 2시간. 스크립트 시작 시각 기준으로 여유를 두고 관리합니다.
T_START = time.time()
TIME_LIMIT_S = float(os.environ.get("PPS_TIME_LIMIT_S", 6300))   # 105분 안에 추론을 끝내는 것이 목표
MIN_CHARS = 6000                        # 시간이 부족할 때 줄일 수 있는 문서 글자 수 하한
EMERGENCY_CHARS = 2000                  # 마감 5분 전부터: 공고당 LLM 호출 규칙은 지키면서 입력만 최소화

# 고시금액(물품·용역, 재정경제부 고시 2026-439) — 적용 조건 게이트에 사용
GOSI_AMOUNT = 230_000_000
ONE_EOK = 100_000_000
# 지방계약 물품·일반용역 지역제한 금액(대회 공지): 시·도 3억 5천만 원, 세종·시·군·구 5억 원.
# 발주기관 단위를 확정하지 않고 둘 중 낮은 값을 써서 게이트를 느슨하게 둡니다.
LOCAL_REGION_LIMIT = 350_000_000

# 항목별 P(위반=1) 임계값. 비어 있으면 모델의 0/1 판정을 그대로 씁니다(logprob 미요청).
# 값은 dev 라벨로 정한 고정 상수이며 평가 데이터로 갱신하지 않습니다.
ITEM_THRESHOLDS: Dict[str, float] = {}


def log(msg: str) -> None:
    print(f"[baseline] {msg}", file=sys.stderr, flush=True)


# ===== 2. 데이터 로더 =====
def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


def validate_record(rec: Any) -> None:
    """레코드 1건의 최소 스키마 검사 (id · docs(공고문 1개 이상) · meta)"""
    if not isinstance(rec, dict):
        raise ValueError(f"레코드가 object가 아니다: {type(rec).__name__}")
    for k in ("id", "docs", "meta"):
        if k not in rec:
            raise ValueError(f"필수 키 없음: {k}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id가 비어 있다")
    docs = rec["docs"]
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"docs가 비어 있다 (id={rec['id']})")
    for d in docs:
        if not isinstance(d, dict) or not all(k in d for k in ("doc_id", "type", "text")):
            raise ValueError(f"docs 원소 형식 오류 (id={rec['id']})")
        if not isinstance(d["text"], str):
            raise ValueError(f"docs.text가 문자열이 아니다 (id={rec['id']})")
    if not any(d["type"] == "공고문" for d in docs):
        raise ValueError(f"공고문이 없다 (id={rec['id']})")
    if not isinstance(rec["meta"], dict):
        raise ValueError(f"meta가 object가 아니다 (id={rec['id']})")


def normalize(rec: Dict[str, Any]) -> Dict[str, Any]:
    """NFC 정규화 — macOS에서 만든 파일은 한글이 NFD로 저장될 수 있어 문자열 비교가 어긋날 수 있습니다."""
    for d in rec.get("docs", []):
        d["text"] = unicodedata.normalize("NFC", d["text"])
        if isinstance(d.get("type"), str):
            d["type"] = unicodedata.normalize("NFC", d["type"])
    return rec


def iter_records(path: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    n = 0
    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
            validate_record(rec)
            yield normalize(rec)
            n += 1
            if limit and n >= limit:
                return


def full_text(rec: Dict[str, Any]) -> str:
    """근거문구 대조용 원문 (프롬프트에 넣은 것과 같은 텍스트 · NFC)"""
    return "\n".join(d["text"] for d in rec["docs"])


def build_context(rec: Dict[str, Any], max_chars: int = 4000) -> str:
    """문서를 프롬프트용 텍스트로 구성합니다.

    공고문을 먼저 배치하고, 나머지 문서는 DOC_ORDER 순서를 따릅니다. `max_chars`를 초과하면
    뒤쪽 문서부터 제외하고 '[미수록 문서]'로 표시합니다. 첫 문서는 길이 상한에 맞게 자릅니다.
    """
    order = {t: i for i, t in enumerate(DOC_ORDER)}
    pool = sorted(rec["docs"], key=lambda d: (order.get(d["type"], len(DOC_ORDER)), d["doc_id"]))

    chunks, used, dropped, truncated = [], 0, Counter(), False
    for i, d in enumerate(pool):
        head = f"[{d['type']}:{d['doc_id']}]\n"
        body = d["text"]
        if used + len(head) + len(body) > max_chars:
            if i == 0:                                   # 첫 문서는 길이 상한에 맞게 자릅니다.
                body = body[: max(0, max_chars - len(head))]
                truncated = True
            else:
                dropped[d["type"]] += 1
                continue
        chunks.append(head + body)
        used += len(head) + len(body)

    for t, n in (rec.get("dropped_doc_counts") or {}).items():
        dropped[t] += n

    text = "\n\n".join(chunks)
    if truncated:
        text += "\n\n[절단] 공고문 뒷부분이 길이 예산으로 잘렸다"
    if dropped:
        text += "\n\n[미수록 문서] " + ", ".join(f"{t} {n}건" for t, n in sorted(dropped.items()))
    return text


def format_meta(rec: Dict[str, Any]) -> str:
    """나라장터 메타를 한 줄짜리 목록으로. 값이 없는 필드는 '미기재'로 표시합니다."""
    m = rec.get("meta", {})
    lines = []
    for k in META_FIELDS:
        if k in m:
            v = m[k]
            lines.append(f"- {k}: {'미기재' if v is None else v}")
    return "\n".join(lines)


# ===== 3. 항목표·디코딩 스키마 =====
# data/에 항목표.json·정답스키마_디코딩.json이 동봉됩니다.
# 항목명·근거조문·비고는 항목표.json 에 있으니 여기에 사본을 두지 않습니다.


def item_table(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    p = os.path.join(data_dir, "항목표.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} 가 없습니다 — data/ 를 그대로 둔 채 실행하세요.")
    return json.load(io.open(p, encoding="utf-8"))["항목"]


def decode_schema(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """베이스라인의 구조화 출력에 사용할 JSON Schema를 불러옵니다."""
    p = os.path.join(data_dir, "정답스키마_디코딩.json")
    if os.path.exists(p):
        s = json.load(io.open(p, encoding="utf-8"))
        return s["properties"]["판정"] if "판정" in s.get("properties", {}) else s
    props = {}
    for v in ITEMS:
        props[v] = {
            "type": "object", "additionalProperties": False,
            "required": ["위반여부", "근거문구"],
            "properties": {
                "위반여부": {"type": "integer", "enum": [0, 1]},
                "근거문구": {"type": "null"} if v in ABSENCE else {"type": ["string", "null"]},
            },
        }
    return {"type": "object", "additionalProperties": False, "required": list(ITEMS), "properties": props}


# ===== 4. 프롬프트 구성 =====
# 베이스라인 프롬프트는 출력 형식과 항목 목록을 구성합니다.
SYSTEM_HEAD = """당신은 공공 입찰공고의 법령 위반 여부를 점검한다.
공고문과 첨부 문서, 그리고 나라장터 입력 메타를 함께 읽고 아래 24개 항목 각각에 대해
위반 여부(1/0)와 근거 문구를 판정한다.

지켜야 할 것
1. 24개 항목 전부에 답한다. 판단이 어려운 항목도 비워 두지 말고 0으로 낸다.
2. 근거 문구는 반드시 **주어진 문서에 그대로 있는 문장**을 옮긴다. 요약하거나 고쳐 쓰지 않는다.
   원문에 없는 문구는 근거로 인정되지 않는다. 500자를 넘기지 않는다.
3. 아래 '근거 없음' 표시가 붙은 항목은 **있어야 할 문구가 없는 것**이 위반이다.
   인용할 원문이 존재하지 않으므로 근거 문구를 null로 둔다.
판정할 24개 항목"""

SYSTEM_TAIL = """
출력은 JSON 하나로만 낸다. 키는 v1~v24, 각 값은 {"위반여부": 0 또는 1, "근거문구": 문자열 또는 null}이다.
설명이나 머리말을 덧붙이지 않는다."""


# 항목별 판정 기준 요약 — 항목표·제공 법령·운영진 답변 기준. USE_ITEM_GUIDE가 켜질 때만 프롬프트에 붙습니다.
USE_ITEM_GUIDE = False
ITEM_GUIDE = {
    "v1": "법령 근거 없이 참가자격을 특정 기관(대학·산학협력단·특정 단체 회원 등)이나 과업과 무관한 특정 시설·인력 보유자로 한정",
    "v2": "추정가격이 고시금액(2억 3천만 원) 미만인데 수행실적으로 참가를 제한. 지방계약 소액수의 견적은 예외",
    "v3": "실적 제한 규모를 사업예산(추정가격)의 1배를 넘게 요구 (예: 기초금액의 130% 이상 실적)",
    "v4": "고시금액 이상 입찰에서 특정 기관의 실적만 인정하거나 특정 실적으로 참가를 제한",
    "v5": "지역제한 허용 금액(국가 고시금액, 지방 시·도 3억 5천만/시·군·구 5억 원) 이상인데 본사 소재지 등 지역으로 제한",
    "v6": "고시금액 미만 지역제한을 시·군·구(기초) 단위로 좁게 제한. 익명 토큰 '단위=기초'는 시·군·구",
    "v7": "고시금액 미만 지역제한에서 인접 지역까지 확대하는 방식이 규정에 맞지 않음",
    "v8": "실적 제한과 지역 제한을 동시에 적용(중복 제한)",
    "v9": "규격서·과업지시서 등에 특정 모델명·제조사명을 지정. '동등 이상' 표현만으로 위반 여부를 정하지 않음",
    "v10": "중소기업자간 경쟁제품 입찰인데 입찰참가자격에 직접생산확인증명서 소지 요구가 없음",
    "v11": "중소기업자간 경쟁제품 입찰인데 중소기업자(중소기업확인서)로 참가를 제한하지 않음",
    "v12": "경쟁제품이 아닌 일반제품(또는 금액 상한을 넘는 품목)에 직접생산확인을 요구",
    "v13": "경쟁제품 경쟁입찰에서 참가자격을 소기업·소상공인으로 한정(중기업 배제). 소액수의 견적은 예외",
    "v14": "추정가격 고시금액 이상의 일반 물품·용역(경쟁제품 아님)을 중소기업자로 제한",
    "v15": "추정가격 1억 이상~고시금액 미만에서 소기업·소상공인으로만 제한(중소기업자 제한이어야 함). 판로지원 예외를 공고에 명시하면 제외",
    "v16": "추정가격 1억 이상~고시금액 미만인데 중소기업자 제한이 없음. 판로지원 예외를 공고에 명시하면 제외",
    "v17": "추정가격 1억 미만 일반 물품·용역을 중소기업자(중기업 포함)로 제한(소기업·소상공인 제한이어야 함)",
    "v18": "추정가격 1억 미만인데 소기업·소상공인 제한이 없음. 확인서 제출 목록만 있고 자격 요건이 없으면 제한 없음",
    "v19": "입찰 단계(입찰 참가 시·제출 마감 전)에 제조사 물품공급·기술지원 확약서 제출·보유를 요구. 계약 시 제출은 위반 아님",
    "v20": "소프트웨어사업인데 사업금액 구간별 대기업·중견기업 참여제한 문구가 없음",
    "v21": "공동수급 구성원 최소 지분율을 기준(공동이행 10%)보다 낮게 정함",
    "v22": "협상에 의한 계약에서 현장설명회 참석 업체로 입찰 참가를 제한",
    "v23": "지방계약 협상에 의한 계약에서 현장설명회 관련 공고 기간이 규정보다 짧음",
    "v24": "공고서의 예산·계약방법·지역제한·업종이 나라장터 입력값(메타)과 다름",
}


def build_system_prompt(tbl: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for v in ITEMS:
        it = tbl[v]
        tag = "  [근거 없음 — null]" if it["부재탐지"] else ""
        note = f" ({it['비고']})" if it.get("비고") else ""
        guide = f"\n    위반 기준: {ITEM_GUIDE[v]}" if USE_ITEM_GUIDE and v in ITEM_GUIDE else ""
        lines.append(f"- {v}: {it['항목명']}{note}{tag}{guide}")
    return SYSTEM_HEAD + "\n" + "\n".join(lines) + "\n" + SYSTEM_TAIL


def build_user_prompt(rec: Dict[str, Any], max_chars: int) -> str:
    return (
        f"[공고 ID] {rec['id']}\n\n"
        f"[나라장터 입력 메타]\n{format_meta(rec)}\n\n"
        f"[문서]\n{build_context(rec, max_chars=max_chars)}\n"
    )


def build_messages(rec: Dict[str, Any], system_prompt: str, max_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(rec, max_chars)},
    ]


# ===== 4-1. 항목 그룹별 질의 (GROUP_MODE) =====
# 24항목을 한 번에 묻는 대신, 같은 문서를 공통 접두부로 두고 항목 그룹마다 질문을 뒤에 붙여 따로 묻습니다.
# system과 문서 부분이 그룹 간에 같아 vLLM prefix caching으로 문서 prefill을 재사용합니다.
GROUP_MODE = False
ITEM_GROUPS = [
    ["v1", "v2", "v3", "v4"],                 # 참가자격·실적
    ["v5", "v6", "v7", "v8"],                 # 지역 제한
    ["v9", "v10", "v11", "v12", "v13"],       # 모델명·경쟁제품·직접생산
    ["v14", "v15", "v16", "v17", "v18"],      # 금액 구간별 기업규모 제한
    ["v19", "v20", "v21", "v22", "v23", "v24"],  # 확약서·SW·공동계약·현장설명회·메타 대조
]
GROUP_SYSTEM = """당신은 공공 입찰공고의 법령 위반 여부를 점검한다.
사용자가 준 나라장터 입력 메타와 문서를 읽고, 맨 끝 [판정할 항목]에 적힌 항목만 판정한다.

지켜야 할 것
1. 지정한 항목 전부에 답한다. 판단이 어려운 항목도 비워 두지 말고 0으로 낸다.
2. 근거 문구는 반드시 **주어진 문서에 그대로 있는 문장**을 옮긴다. 요약하거나 고쳐 쓰지 않는다.
   원문에 없는 문구는 근거로 인정되지 않는다. 500자를 넘기지 않는다.
3. '근거 없음' 표시가 붙은 항목은 **있어야 할 문구가 없는 것**이 위반이다. 근거 문구를 null로 둔다.
출력은 JSON 하나로만 낸다. 설명이나 머리말을 덧붙이지 않는다."""


def group_question(tbl: Dict[str, Dict[str, Any]], group: List[str]) -> str:
    lines = []
    for v in group:
        it = tbl[v]
        tag = "  [근거 없음 — null]" if it["부재탐지"] else ""
        note = f" ({it['비고']})" if it.get("비고") else ""
        guide = f"\n    위반 기준: {ITEM_GUIDE[v]}" if v in ITEM_GUIDE else ""
        lines.append(f"- {v}: {it['항목명']}{note}{tag}{guide}")
    keys = ", ".join(group)
    return ("\n[판정할 항목]\n" + "\n".join(lines) +
            f"\n\n출력 키는 {keys}이고, 각 값은 {{\"위반여부\": 0 또는 1, \"근거문구\": 문자열 또는 null}}이다.")


def build_group_messages(rec: Dict[str, Any], tbl: Dict[str, Dict[str, Any]], group: List[str],
                         max_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": GROUP_SYSTEM},
        {"role": "user", "content": build_user_prompt(rec, max_chars) + group_question(tbl, group)},
    ]


def restrict_schema(schema: Dict[str, Any], group: List[str]) -> Dict[str, Any]:
    """24항목 스키마에서 그룹 항목만 남긴 스키마."""
    return {**schema, "required": [v for v in schema.get("required", ITEMS) if v in group],
            "properties": {v: schema["properties"][v] for v in group}}


# ===== 4-2. 사실 추출 요청 (FACTS_MODE, GROUP_MODE에서만) =====
# 모델에게 위반 여부가 아니라 "문서에 적힌 사실"만 묻고, 판정은 금액 구간·항목 정의에 따른 코드 결정표로 합니다.
# 대상: v10·v12·v13(경쟁제품·직접생산), v14~v18(금액 구간별 기업규모), v20(SW 참여제한).
FACTS_MODE = False
FACT_ITEMS = ["v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20"]
_Q = {"type": ["string", "null"], "maxLength": EVIDENCE_MAX}
FACT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "구매대상_구분": {"type": "string", "enum": ["일반", "경쟁제품", "기타", "미확인"]},
        "구매대상_인용": _Q,
        "자격문장_역할": {"type": "string", "enum": ["참가자격", "제출서류목록", "법령인용", "언급없음", "미확인"]},
        "기업규모_자격": {"type": "string", "enum": ["소기업소상공인만", "중소기업자", "제한없음", "미확인"]},
        "자격_인용": _Q,
        "자격절_전체관측": {"type": "string", "enum": ["예", "아니오"]},
        "우선조달_예외": {"type": "string", "enum": ["있음", "없음", "미확인"]},
        "직접생산_요구_인용": _Q,
        "SW사업": {"type": "string", "enum": ["예", "아니오", "미확인"]},
        "SW참여제한_인용": _Q,
    },
}
FACT_SCHEMA["required"] = list(FACT_SCHEMA["properties"])
FACT_QUESTION = """
[사실 추출] 위반 여부를 판정하지 말고, 문서에 적힌 사실만 아래 키로 답한다. 인용은 문서의 연속된 원문 한 구간(500자 이하)이며 없으면 null.
- 구매대상_구분: 실제로 구매하는 물품·용역이 아래 [경쟁제품 후보]의 중소기업자간 경쟁제품에 해당하면 경쟁제품, 해당하지 않는 일반 물품·용역이면 일반, 공사·엔지니어링·별도 규정의 SW사업 등이면 기타, 판단할 수 없으면 미확인. 후보 목록에 이름이 있다는 것만으로 경쟁제품이 되지 않는다.
- 구매대상_인용: 구매 대상을 나타내는 원문.
- 자격문장_역할: 기업규모(중소기업·소기업·소상공인) 문구가 입찰참가자격 조건이면 참가자격, 제출서류 목록에만 있으면 제출서류목록, 법령 인용·배제 조항에만 있으면 법령인용, 없으면 언급없음.
- 기업규모_자격: 참가할 수 있는 기업규모. 소기업·소상공인만(중기업 배제)이면 소기업소상공인만, 중소기업(중기업 포함)이면 중소기업자, 규모 조건이 없으면 제한없음. 나라장터 메타(조항호내용)는 근거로 쓰지 않는다.
- 자격_인용: 기업규모 조건 원문. 제한없음이면 관측한 입찰참가자격 절 원문.
- 자격절_전체관측: 입찰참가자격 절 전체를 봤으면 예.
- 우선조달_예외: 판로지원법 시행령 제2조의3의 우선조달 예외 사유가 명시돼 있으면 있음, 없으면 없음.
- 직접생산_요구_인용: 직접생산확인증명서 제출·소지 등 직접생산 요구 원문. 없으면 null.
- SW사업: 소프트웨어 개발·운영·유지보수가 계약 산출물이면 예.
- SW참여제한_인용: 대기업·중견기업 참여제한(하한제도) 적용 여부를 밝힌 원문. 없으면 null."""


def load_catalog(data_dir: str = DATA_DIR) -> List[Dict[str, str]]:
    p = os.path.join(data_dir, "법령패키지", "중기부고시", "중기부고시_경쟁제품_세부품명.csv")
    if not os.path.exists(p):
        return []
    with io.open(p, encoding="utf-8-sig", newline="") as f:
        return [{k: unicodedata.normalize("NFC", v or "") for k, v in r.items()} for r in csv.DictReader(f)]


def catalog_matches(rec: Dict[str, Any], catalog: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """메타·문서의 10자리 세부품명번호, 또는 4자 이상 세부품명 문자열이 공고에 나오는 고시 행."""
    meta_src = str((rec.get("meta") or {}).get("세부품명번호목록") or "")
    text = full_text(rec)
    codes = set(re.findall(r"(?<!\d)\d{10}(?!\d)", meta_src + "\n" + text))
    flat = re.sub(r"\s+", "", meta_src + text)
    out = []
    for row in catalog:
        name = re.sub(r"\s+", "", row.get("세부품명", ""))
        if row.get("세부품명번호") in codes or (len(name) >= 4 and name in flat):
            out.append(row)
    return out


def fact_question(rec: Dict[str, Any], catalog: List[Dict[str, str]]) -> str:
    rows = catalog_matches(rec, catalog)[:12]
    cand = "\n".join(f"- {r['세부품명번호']} {r['세부품명']}" + (f" (특이사항: {r['특이사항']})" if r.get("특이사항") else "")
                     for r in rows) or "- (공고에서 찾은 후보 없음)"
    return FACT_QUESTION + "\n\n[경쟁제품 후보]\n" + cand


def build_fact_messages(rec: Dict[str, Any], catalog: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": GROUP_SYSTEM},
        {"role": "user", "content": build_user_prompt(rec, max_chars) + "\n" + fact_question(rec, catalog)},
    ]


# ----- 직접생산 요구 문장 × 경쟁제품 고시 대조 규칙 (v11·v12, 0→1만) -----
# 참가자격으로 직접생산확인을 요구하는 문장이 지목한 품명(10자리 번호·세부품명)을 고시와 대조합니다.
# 고시 특이사항의 "추정가격 N억원 미만" 상한을 넘으면 그 품명은 경쟁제품이 아닙니다.
#   - 지목 품명이 모두 경쟁제품이 아닌데 직생을 요구 → v12(일반제품 직생 제한)
#   - 경쟁제품인데 공고 어디에도 중소기업자 참가 조건이 없음 → v11(중기간 경쟁제품 중소 없음)
# dev: v12 TP2/FP0, v11 TP2/FP1. 무라벨 발화율은 dev 대비 v12 0.07배, v11 0.53배(서버에서는 드물게 발화).
CATALOG_RULES = True
CAT_CAP = re.compile(r"추정가격\s*([\d,.]+)\s*억\s*원?\s*미만")
DP_DEMAND = re.compile(r"직접\s*생산\s*확인\s*(?:증명서|서류|기준)?[^.\n]{0,40}?"
                       r"(?:소지|보유|제출|갖추|갖춘|있는\s*자|발급)")
DP_SANCTION = re.compile(r"직접\s*생산\s*확인\s*기준을?\s*위반|직접생산\s*여부\s*확인\s*결과")
SME_CLAUSE = re.compile(r"중소기업(?:자|기본법)?[^.\n]{0,60}?(?:확인서|제한|한정|참가|자격|로서|이어야)")


def catalog_cap(row: Dict[str, str]) -> Optional[float]:
    m = CAT_CAP.search(row.get("특이사항") or "")
    return float(m.group(1).replace(",", "")) * 1e8 if m else None


def dp_demand(rec: Dict[str, Any], catalog: List[Dict[str, str]]) -> Tuple[Optional[str], set]:
    """(참가자격 직생 요구 원문, 그 주변에서 지목한 세부품명번호). 요구가 없으면 (None, 빈 집합)."""
    quote, codes = None, set()
    names = [(re.sub(r"\s+", "", r.get("세부품명", "")), r.get("세부품명번호")) for r in catalog]
    for d in rec["docs"]:
        t = d["text"]
        for m in DP_DEMAND.finditer(t):
            w = t[max(0, m.start() - 200): m.end() + 200]
            if DP_SANCTION.search(w):
                continue                                   # 계약 후 제재 안내는 참가자격 요구가 아님
            if quote is None:
                a = t.rfind("\n", 0, m.start()) + 1
                b = t.find("\n", m.end())
                quote = t[a: b if b != -1 else len(t)][:EVIDENCE_MAX]
            codes |= set(re.findall(r"(?<!\d)\d{10}(?!\d)", w))
            flat = re.sub(r"\s+", "", w)
            codes |= {c for n, c in names if len(n) >= 4 and n in flat}
    return quote, codes


def catalog_rules(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any],
                  catalog: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    if not CATALOG_RULES or not catalog:
        return judgment
    quote, codes = dp_demand(rec, catalog)
    if quote is None or not codes:
        return judgment
    by_code = {r.get("세부품명번호"): r for r in catalog}
    price = parse_amount(rec.get("meta") or {})
    competitive = False
    for c in codes:
        row = by_code.get(c)
        if row is None:
            continue
        cap = catalog_cap(row)
        if cap is not None and price is not None and price >= cap:
            continue
        competitive = True
        break
    out = dict(judgment)
    if not competitive and out["v12"]["위반여부"] != 1:
        out["v12"] = {"위반여부": 1, "근거문구": quote}
    elif competitive and out["v11"]["위반여부"] != 1 and not SME_CLAUSE.search(full_text(rec)):
        out["v11"] = {"위반여부": 1, "근거문구": ""}
    return out


# ----- 참가자격 문구 규칙 (v4·v7·v8, 0→1만) -----
# v8 실적+지역 중복 제한, v7 지역제한을 서로 다른 광역 둘 이상으로 확대(허용 금액 미만일 때),
# v4 실적을 특정 기관 발주·납품으로 한정. 배점표·서식·제출서류 문맥은 제외합니다.
# dev 규칙 단독: v8 5/0/1, v7 7/0/0, v4 6/1/0 · 무라벨 발화율은 dev 대비 0.52·0.84·0.29배.
QUAL_RULES = True
QR_PERF = re.compile(r"실적(?:을|이)?\s*(?:보유|있는|갖춘|충족)|(?:이상|초과)(?:인|의)?\s*실적|실적\s*증명서?\s*(?:보유|소지|제출)"
                  r"|수행\s*실적|준공\s*금?액[^\n]{0,40}이상|납품\s*실적")
QR_REGION = re.compile(r"본점\s*소재지|주된\s*영업소|소재지를\s*[^\n]{0,60}(?:두고|둔)|관내에\s*(?:있는|소재)"
                    r"|에\s*소재한\s*(?:업체|자)|지역\s*제한|(?:에|내에)\s*소재(?:한|하고|하는)")
QR_NOT_Q = re.compile(r"배점|평가\s*항목|평가표|정량\s*평가|정성\s*평가|가점|심사\s*기준|평가\s*기준|제안서\s*평가|서식|별지|제출\s*서류|증빙\s*서류")
QR_Q_HEAD = re.compile(r"참가\s*자격|자격\s*요건")
QR_WIDE = (r"[가-힣]{2}특별자치도|[가-힣]{2}특별자치시|[가-힣]{2,3}광역시|서울특별시|경기도|강원도|충청북도|충청남도"
        r"|전라북도|전라남도|경상북도|경상남도|제주도")
QR_QQ = r"[\s\"'“”‘’]*"
QR_REGION_PAIR = re.compile(r"(" + QR_WIDE + r")" + QR_QQ + r"[^\n]{0,60}?(?:또는|,|·|및)" + QR_QQ + r"(" + QR_WIDE + r")")
QR_REGION_ANCHOR = re.compile(r"본점\s*소재지|주된\s*영업소|소재지를\s*[^\n]{0,80}(?:두고|둔)|관할\s*구역\s*안에|지역\s*제한|소재지가")
QR_INST = (r"국가기관|공공기관|정부투자기관|지방자치단체|지자체|공기업|준정부기관|정부기관|교육청|고등학교|대학교"
        r"|대학병원|종합병원|국공립|초등학교|중학교")
QR_ORD = r"발주|시행|납품|공급|체결|수주"
QR_INST_ORD = re.compile(r"(?:" + QR_INST + r")[^\n]{0,12}?(?:이|가|에서|에게|에|,)?\s*(?:" + QR_ORD + r")(?:한|된|하는|하여)[^\n]{0,80}?실적")
QR_INST_NEAR = re.compile(r"(?:" + QR_INST + r")[^\n]{0,60}?실적")
QR_PRIVATE = re.compile(r"민간|일반\s*기업|기업체\s*포함|개인\s*포함")
QR_TAIL = re.compile(r"업체이어야|업체여야|업체만|자격이\s*있|있는\s*업체|보유한\s*업체|있어야\s*합니다|자로\s*제한|하여야\s*합니다|참가\s*자격")


def qr_in_qual(text, *positions, lookback=40):
    for pos in positions:
        a = text.rfind("\n", 0, pos) + 1
        b = text.find("\n", pos)
        if QR_NOT_Q.search(text[a: b if b != -1 else len(text)]):
            return False
        for line in reversed(text[:a].split("\n")[-lookback:]):
            if QR_Q_HEAD.search(line):
                break
            if QR_NOT_Q.search(line):
                return False
    return True


def qr_quote(text, s, e, cap=480):
    return text[max(0, s - 40): min(len(text), e + 60)][:cap].strip()


def qr_v8(rec, window=800):
    for d in rec["docs"]:
        t = d["text"]
        perf = [m.span() for m in QR_PERF.finditer(t)]
        reg = [m.span() for m in QR_REGION.finditer(t)]
        pairs = sorted(((abs(p[0] - r[0]), p, r) for p in perf for r in reg if abs(p[0] - r[0]) <= window))
        for _, p, r in pairs:
            if qr_in_qual(t, p[0], r[0]):
                return qr_quote(t, min(p[0], r[0]), max(p[1], r[1]))
    return None


def qr_v7(rec, allowed):
    if not allowed:
        return None
    for d in rec["docs"]:
        t = d["text"]
        for a in QR_REGION_ANCHOR.finditer(t):
            s0 = max(0, a.start() - 150)
            seg = t[s0: a.end() + 200]
            for m in QR_REGION_PAIR.finditer(seg):
                if m.group(1) != m.group(2):
                    return qr_quote(t, s0 + m.start(), s0 + m.end())
    return None


def qr_v4(rec):
    for d in rec["docs"]:
        t = d["text"]
        for pat, tail in ((QR_INST_ORD, False), (QR_INST_NEAR, True)):
            for m in pat.finditer(t):
                around = t[max(0, m.start() - 300): m.end() + 300]
                if QR_PRIVATE.search(around) or not qr_in_qual(t, m.start()):
                    continue
                if tail and not QR_TAIL.search(t[m.end(): m.end() + 60]):
                    continue
                return qr_quote(t, m.start(), m.end())
    return None


QR_REGION_TOKEN = re.compile(r"\[지역:[^\]]*\]")
QR_MONEY = re.compile(r"(?:(\d+(?:\.\d+)?)\s*억)?\s*(?:(\d+(?:\.\d+)?)\s*천\s*만)?\s*(?:([\d,]+)\s*만)?\s*(?:([\d,]{4,}))?\s*원")
# 실적 금액 요구 표현 확장: '실적이 3천만원 이상', '실적 5천만원 이상 보유' (v2, dev 7/4/0, 무라벨 0.27배)
QR_PERF2 = re.compile(QR_PERF.pattern + r"|실적(?:이|을)?\s*[^\n]{0,25}?\d[\d,.]*\s*(?:억|천만|만)?\s*원[^\n]{0,10}?(?:이상|초과)")
# v1 확장: 일정 규모 이상의 인력·시설·센터 보유를 참가자격으로 요구 (운영진 답변: 과업 관련성 미확인 시설·인력 보유 제한도 v1)
# dev v1 4/1/3, 무라벨 0.76배
QR_V1B = re.compile(r"(?:\d+\s*(?:명|인|대|개소|곳)\s*이상|전국|모든)[^\n]{0,40}?"
                    r"(?:인력|시설|센터|장비|사무소|지사|차량|정비소)[^\n]{0,30}?(?:보유|갖춘|갖추|있는|확보)")
QR_PLEDGE = re.compile(r"(?:물품\s*공급|기술\s*지원|공급)[^\n]{0,15}?(?:확약서|협약서|확인서)")
QR_BID = re.compile(r"입\s*찰\s*(?:참가|서|시|등록)|투\s*찰|제출\s*마감|참가\s*신청|입찰\s*참가자는")
QR_V1 = re.compile(r"(고등교육법|산학협력단|대학(?:교)?|연구기관|협회\s*회원|정부출연)[^\n]{0,60}?"
                   r"(?:만\s*(?:참여|참가|입찰)|에\s*한하여|으로\s*한정|로\s*한정|참여\s*가능)")
QR_SHARE = re.compile(r"(?:지분|출자\s*비율|분담\s*비율)[^\n]{0,40}?(\d{1,2}(?:\.\d+)?)\s*%")


def qr_money(text: str) -> List[float]:
    """'1억 5천만원', '455,000,000원' 같은 금액 표기를 원 단위로. 100만 원 미만은 버립니다."""
    vals = []
    for m in QR_MONEY.finditer(text):
        a, b, c, d = m.groups()
        v = ((float(a) * 1e8 if a else 0) + (float(b) * 1e7 if b else 0)
             + (float(c.replace(",", "")) * 1e4 if c else 0) + (float(d.replace(",", "")) if d else 0))
        if v >= 1e6:
            vals.append(v)
    return vals


def qr_region_clauses(rec: Dict[str, Any]) -> List[str]:
    """참가자격 문맥에 있는 지역 제한 문구가 들어 있는 줄."""
    out = []
    for d in rec["docs"]:
        t = d["text"]
        for m in QR_REGION.finditer(t):
            if qr_in_qual(t, m.start()):
                a = t.rfind("\n", 0, m.start()) + 1
                b = t.find("\n", m.end())
                out.append(t[a: b if b != -1 else len(t)].strip()[:480])
    return out


def qualification_rules(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not QUAL_RULES:
        return judgment
    m = rec.get("meta") or {}
    price = parse_amount(m)
    limit = LOCAL_REGION_LIMIT if "지방" in str(m.get("적용계약법") or "") else GOSI_AMOUNT
    found = {"v8": qr_v8(rec), "v7": qr_v7(rec, price is None or price < limit), "v4": qr_v4(rec)}
    # v5: 지역제한 허용 금액 이상인데 참가자격에 지역 제한 (dev 5/2/2, 무라벨 발화율 1.27배)
    # v6: 허용 금액 미만 지역제한을 시·군·구 단위(익명 토큰 '단위=기초')로 제한, 소액수의 제외 (dev 3/0/3, 0.24배)
    # v3: 참가자격 실적 문구의 같은 줄에 추정가격을 넘는 금액 '이상' 요구 (dev 7/0/1, 무라벨 발화율 0.10배)
    if price:
        for d in rec["docs"]:
            t = d["text"]
            for mm in QR_PERF.finditer(t):
                if "v3" in found or not qr_in_qual(t, mm.start()):
                    continue
                a = t.rfind("\n", 0, mm.start()) + 1
                b = t.find("\n", mm.end())
                line = t[a: b if b != -1 else len(t)]
                if "이상" in line and any(v > price for v in qr_money(line)):
                    found["v3"] = line.strip()[:480]
    # v1: 참가자격을 대학·산학협력단·연구기관·협회 회원 등 특정 기관으로 한정 (dev 2/0/5, 무라벨 0.23배)
    for d in rec["docs"]:
        t = d["text"]
        for mm in list(QR_V1.finditer(t)) + list(QR_V1B.finditer(t)):
            if "v1" not in found and qr_in_qual(t, mm.start()):
                a = t.rfind("\n", 0, mm.start()) + 1
                b = t.find("\n", mm.end())
                found["v1"] = t[a: b if b != -1 else len(t)].strip()[:480]
    # v2: 고시금액 미만(지방 소액수의 제외)인데 참가자격 실적 문구 줄에 금액 '이상' 실적 요구 (dev 4/3/3, 무라벨 0.30배)
    if price and price < GOSI_AMOUNT and not ("지방" in str(m.get("적용계약법") or "")
                                             and "소액수의" in str(m.get("낙찰방법") or "")):
        for d in rec["docs"]:
            t = d["text"]
            for mm in QR_PERF2.finditer(t):
                if "v2" in found or not qr_in_qual(t, mm.start()):
                    continue
                a = t.rfind("\n", 0, mm.start()) + 1
                b = t.find("\n", mm.end())
                line = t[a: b if b != -1 else len(t)]
                if "이상" in line and qr_money(line):
                    found["v2"] = line.strip()[:480]
    # v19: 물품공급·기술지원 확약서를 입찰 단계(같은 줄에 입찰 참가·투찰·제출 마감)에 요구 (dev 3/2/3, 0.35배)
    for d in rec["docs"]:
        t = d["text"]
        for mm in QR_PLEDGE.finditer(t):
            a = t.rfind("\n", 0, mm.start()) + 1
            b = t.find("\n", mm.end())
            line = t[a: b if b != -1 else len(t)]
            if "v19" not in found and QR_BID.search(line):
                found["v19"] = line.strip()[:480]
    # v21: 공동수급 구성원 최소 지분율을 5% 미만으로 정함 (dev 4/0/2, 0.07배). 5%는 dev 라벨이 갈려 제외
    ft = full_text(rec)
    for mm in QR_SHARE.finditer(ft):
        if float(mm.group(1)) < 5 and "공동" in ft[max(0, mm.start() - 200): mm.end()]:
            a = ft.rfind("\n", 0, mm.start()) + 1
            b = ft.find("\n", mm.end())
            found["v21"] = ft[a: b if b != -1 else len(ft)].strip()[:480]
            break
    clauses = qr_region_clauses(rec)
    if clauses and price is not None and price >= limit:
        found["v5"] = clauses[0]
    basic = [c for c in clauses if any("단위=기초" in t for t in QR_REGION_TOKEN.findall(c))]
    if basic and price is not None and price < limit and "소액수의" not in str(m.get("낙찰방법") or ""):
        found["v6"] = basic[0]
    out = dict(judgment)
    for v, q in found.items():
        if q and out[v]["위반여부"] != 1:
            out[v] = {"위반여부": 1, "근거문구": q}
    return out


# ----- 규칙 적용 방식 -----
# RULE_OVERRIDE_ITEMS: 규칙 결과로 모델 판정을 대체(규칙이 안 걸리면 0). 모델 오탐을 없애는 대신 규칙 재현율에 의존합니다.
# 나머지 항목: 규칙이 걸리면 1로만 올립니다.
# 근거: 베이스라인 모델은 일부 항목 오탐률이 매우 높고(공개 분석 v13·v17·v21 등), 규칙은 dev 정밀도가 높습니다.
# 대상 선정: 3차 서버 0.21과 비슷한 가상 모델(재현율 50%·오탐 15%, 게이트 적용)의 항목별 F1보다
# 규칙 단독 dev F1이 높은 항목(17개). 나머지 7개(v9·v10·v15·v17·v20·v23·v24)는 모델 판정을 씁니다.
RULE_OVERRIDE_ITEMS = {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v11", "v12", "v13", "v14",
                       "v16", "v18", "v19", "v21", "v22", "v23"}


# ----- 공고 본문 참가자격의 기업규모 규칙 (v14·v16) -----
# 나라장터 메타(조항호내용)가 아니라 본문 참가자격 줄에서 허용 기업규모를 읽습니다.
# 법령명·규정명·확인서 이름은 지우고 읽습니다('중소기업기본법', '중·소기업·소상공인 확인서' 등).
#   v14: 추정가격 ≥ 고시금액인 일반(비경쟁제품) 물품·용역을 중소기업·소기업으로 제한 (dev 8/5/0, 무라벨 0.53배)
#   v16: 1억 ≤ 추정가격 < 고시금액인데 기업규모 제한 문구가 없음(완전관측일 때만) (dev 5/7/1, 0.71배)
# v15·v17·v18은 같은 방식으로 dev 오탐이 커서 쓰지 않습니다.
SZ_RESTRICT = re.compile(r"제한|한정|에\s*한함|에\s*한하여|만\s*(?:참|입찰|가능)|이어야|로서|자격"
                         r"|확인서\s*(?:를|을)?\s*(?:소지|보유|발급)")
SZ_LAWNAME = re.compile(r"「[^」]*」|｢[^｣]*｣|『[^』]*』|\[[^\]]*\]")
SZ_RULENAME = re.compile(r"중소기업\s*범위\s*및\s*확인에\s*관한\s*규정"
                         r"|중?\s*[·ㆍ・/]?\s*소기업\s*[·ㆍ・/]?\s*소상공인\s*확인서|중소기업\s*확인서")
SZ_SME = re.compile(r"중소기업(?!자간)|중기업|중\s*[·ㆍ・/]\s*소")
SZ_SMALL = re.compile(r"(?<!중)소기업|소상공인")


def size_class(rec: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """본문 참가자격의 허용 기업규모: 'sme'(중기업 포함) / 'small'(소기업·소상공인만) / 'none', 그리고 근거 줄."""
    kinds, quote = set(), None
    for d in rec["docs"]:
        t = d["text"]
        for lm in re.finditer(r"[^\n]+", t):
            line = lm.group(0)
            core = SZ_RULENAME.sub("", SZ_LAWNAME.sub("", line))
            sme, small = bool(SZ_SME.search(core)), bool(SZ_SMALL.search(core))
            if not (sme or small) or not SZ_RESTRICT.search(core) or not qr_in_qual(t, lm.start()):
                continue
            kinds.add("sme" if sme else "small")
            quote = quote or line.strip()[:480]
    if not kinds:
        return "none", None
    return ("sme" if "sme" in kinds else "small"), quote


def size_rules(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any],
               catalog: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    price = parse_amount(rec.get("meta") or {})
    if price is None or price < ONE_EOK:
        return judgment
    _, codes = dp_demand(rec, catalog)
    by_code = {r.get("세부품명번호"): r for r in catalog}
    for c in codes:                                    # 경쟁제품이면 일반물품 항목(v14·v16)이 아님
        row = by_code.get(c)
        cap = catalog_cap(row) if row else None
        if row is not None and not (cap and price >= cap):
            return judgment
    kind, quote = size_class(rec)
    out = dict(judgment)
    complete = (rec.get("input_completeness") or {}).get("완전관측") is True
    if price >= GOSI_AMOUNT and kind in ("sme", "small"):
        out["v14"] = {"위반여부": 1, "근거문구": quote or ""}
    elif (ONE_EOK <= price < GOSI_AMOUNT and kind == "none" and complete
          and "수의" not in str((rec.get("meta") or {}).get("계약방법") or "")):
        out["v16"] = {"위반여부": 1, "근거문구": ""}
    return out                                          # v15(소기업만) 규칙은 dev 2/5/4로 모델보다 못해 쓰지 않음


def small_price_rules(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """v18: 추정가격 1억 미만인데 본문 참가자격에 기업규모 제한이 없음(완전관측) (dev 6/24/1)."""
    m = rec.get("meta") or {}
    price = parse_amount(m)
    if price is None or price >= ONE_EOK or "수의" in str(m.get("계약방법") or ""):
        return judgment                                 # 수의계약은 적용이 불명확(운영진 답변) — dev 오탐 14/24가 수의계약
    kind, _ = size_class(rec)
    out = dict(judgment)
    if kind == "none" and (rec.get("input_completeness") or {}).get("완전관측") is True:
        out["v18"] = {"위반여부": 1, "근거문구": ""}
    return out


QR_SITE = re.compile(r"현장\s*설명회[^\n]{0,60}?(?:참석|참가)[^\n]{0,60}?(?:업체|자)[^\n]{0,30}?"
                     r"(?:에\s*한하여|만|한함|자격|입찰\s*참가)")
# 사업·과업 설명회, '미참석·불참 업체 제외', '참석하지 아니한 업체의 입찰 참가는 허용되지 않음' (dev v22 5/0/0, 무라벨 0.13배)
QR_SITE2 = re.compile(r"(?:현장|사업|과업)\s*설명회[^\n]{0,120}?(?:미\s*참석|불\s*참|참석하지\s*(?:아니한|않은)"
                      r"|참석한\s*(?:자|업체)(?:에\s*한|만|로\s*한정)?)")
QR_SITE_NEG = re.compile(r"허용되지|제외|접수하지|불가|자격|에\s*한|만\s*(?:입찰|참가)|참석한\s*자")


# ----- v23: 지방 협상계약의 제안요청서(현장·사업) 설명 시기 -----
# 지방자치단체 입찰시 낙찰자 결정기준 제7장 제3절 2-다: 설명은 제안서 제출마감일 전일부터 기산하여
# 40일(추정가격 10억 이상)·20일(1억~10억)·10일(1억 미만) 전에 하고, 공고는 설명일 전일부터 7일 전에 해야 한다.
# 제출마감은 메타 개찰예정일자로 근사합니다. dev 4/3/1.
V23_BRIEF = re.compile(r"(?:현장|사업|과업|제안\s*요청서?)\s*설명(?:회)?")
V23_DATE = re.compile(r"(?:(20\d{2})\s*[.\-/년]\s*)?(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*[.일]?")
V23_SKIP = re.compile(r"없음|미실시|생략|하지\s*않")


def _ymd(value: Any):
    import datetime as _dt
    s_ = re.sub(r"\D", "", str(value or ""))
    try:
        return _dt.date(int(s_[:4]), int(s_[4:6]), int(s_[6:8]))
    except (ValueError, IndexError):
        return None


def v23_rule(rec: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
    """(설명 시기 위반 여부, 근거 줄). 설명일을 못 찾으면 None."""
    import datetime as _dt
    m = rec.get("meta") or {}
    if "지방" not in str(m.get("적용계약법") or "") or "협상" not in str(m.get("낙찰방법") or ""):
        return None
    post, dead = _ymd(m.get("공고게시일자")), _ymd(m.get("개찰예정일자"))
    if not post or not dead:
        return None
    p = parse_amount(m) or 0
    need = 40 if p >= 1e9 else (20 if p >= ONE_EOK else 10)
    for d in rec["docs"]:
        t = d["text"]
        for bm in V23_BRIEF.finditer(t):
            w = t[bm.end(): bm.end() + 120]
            if V23_SKIP.search(w[:30]):
                continue
            for dm in V23_DATE.finditer(w):
                y = int(dm.group(1)) if dm.group(1) else post.year
                try:
                    b = _dt.date(y, int(dm.group(2)), int(dm.group(3)))
                except ValueError:
                    continue
                if not (post <= b <= dead):
                    continue
                one = _dt.timedelta(days=1)
                short = ((dead - one) - b).days < need or ((b - one) - post).days < 7
                a = t.rfind("\n", 0, bm.start()) + 1
                e = t.find("\n", bm.end())
                return short, t[a: e if e != -1 else len(t)].strip()[:480]
    return None


def misc_rules(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any],
               catalog: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """v22: 협상계약에서 현장설명회 참석 업체로 참가 제한 (dev 1/0/4).
    v13: 경쟁제품(직생 요구 품명이 고시 대상)인데 참가자격을 소기업·소상공인으로 한정, 소액수의 제외 (dev 2/3/4)."""
    out = dict(judgment)
    m = rec.get("meta") or {}
    ft = full_text(rec)
    if "협상" in str(m.get("낙찰방법") or ""):
        sm = QR_SITE.search(ft)
        if not sm:
            for cand in QR_SITE2.finditer(ft):
                a = ft.rfind("\n", 0, cand.start()) + 1
                b = ft.find("\n", cand.end())
                if QR_SITE_NEG.search(ft[a: b if b != -1 else len(ft)]):
                    sm = cand
                    break
        if sm:
            a = ft.rfind("\n", 0, sm.start()) + 1
            b = ft.find("\n", sm.end())
            out["v22"] = {"위반여부": 1, "근거문구": ft[a: b if b != -1 else len(ft)].strip()[:480]}
    r23 = v23_rule(rec)
    if r23 and r23[0]:
        out["v23"] = {"위반여부": 1, "근거문구": r23[1]}
    if "소액수의" not in str(m.get("낙찰방법") or ""):
        _, codes = dp_demand(rec, catalog)
        price = parse_amount(m)
        by_code = {r.get("세부품명번호"): r for r in catalog}
        comp = any(c in by_code and not (catalog_cap(by_code[c]) and price and price >= catalog_cap(by_code[c]))
                   for c in codes)
        if not comp:
            for row in catalog_matches(rec, catalog):
                cap = catalog_cap(row)
                if not (cap and price and price >= cap):
                    comp = True
                    break
        kind, quote = size_class(rec)
        if comp and kind == "small":
            out["v13"] = {"위반여부": 1, "근거문구": quote or ""}
    return out


def apply_rules(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any],
                catalog: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    zero = {v: {"위반여부": 0, "근거문구": ""} for v in ITEMS}
    ruled = qualification_rules(catalog_rules(zero, rec, catalog), rec)
    ruled = misc_rules(small_price_rules(size_rules(ruled, rec, catalog), rec), rec, catalog)
    out = {}
    for v in ITEMS:
        if v in RULE_OVERRIDE_ITEMS:
            out[v] = ruled[v]
        elif ruled[v]["위반여부"] == 1 and judgment[v]["위반여부"] != 1:
            out[v] = ruled[v]
        else:
            out[v] = judgment[v]
    return out


def restore_quote(quote: Optional[str], src: str) -> Optional[str]:
    """공백·줄바꿈만 다른 인용을 원문 표기로 되돌립니다. 못 찾으면 None. (8자 미만은 오일치 위험으로 제외)"""
    if not quote:
        return None
    flat_q = re.sub(r"\s+", "", quote)
    if len(flat_q) < 8:
        return None
    idx = [i for i, ch in enumerate(src) if not ch.isspace()]
    flat = "".join(src[i] for i in idx)
    k = flat.find(flat_q)
    if k == -1:
        return None
    return src[idx[k]: idx[k + len(flat_q) - 1] + 1]


def parse_facts(text: str) -> Optional[Dict[str, Any]]:
    obj = extract_json(text)
    if not isinstance(obj, dict) or not all(k in obj for k in FACT_SCHEMA["properties"]):
        return None
    return obj


def decide_from_facts(facts: Optional[Dict[str, Any]], rec: Dict[str, Any],
                      catalog: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """사실 → 항목 판정. 확정할 수 없으면 그 항목은 비워 두어 그룹 판정을 그대로 씁니다."""
    if not facts:
        return {}
    src = unicodedata.normalize("NFC", full_text(rec))

    def quoted(q):
        if not (isinstance(q, str) and q.strip()):
            return False
        q = unicodedata.normalize("NFC", q).strip()
        return q in src or restore_quote(q, src) is not None

    kind = facts["구매대상_구분"]
    if kind == "미확인" or not quoted(facts["구매대상_인용"]):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    complete = (facts["자격절_전체관측"] == "예"
                and (rec.get("input_completeness") or {}).get("완전관측") is True
                and not any((rec.get("dropped_doc_counts") or {}).values()))
    matched = bool(catalog_matches(rec, catalog))
    dp = facts["직접생산_요구_인용"]

    # 역할과 기업규모가 모순이면 기업규모를 미확인으로 둡니다.
    size, role = facts["기업규모_자격"], facts["자격문장_역할"]
    if role == "참가자격" and size not in ("소기업소상공인만", "중소기업자"):
        size = "미확인"
    if role in ("제출서류목록", "법령인용", "언급없음") and size != "제한없음":
        size = "미확인"
    size_ok = size != "미확인" and quoted(facts["자격_인용"])

    # v10·v12·v13 (경쟁제품·직접생산)
    if kind == "일반" and quoted(dp):
        out["v12"] = {"위반여부": 1, "근거문구": dp}
    if kind == "경쟁제품" and complete:
        out["v10"] = {"위반여부": 0 if quoted(dp) else 1, "근거문구": None}
    if kind == "경쟁제품" and matched and size_ok and size == "소기업소상공인만" \
            and "소액수의" not in str((rec.get("meta") or {}).get("낙찰방법") or ""):
        out["v13"] = {"위반여부": 1, "근거문구": facts["자격_인용"]}

    # v20 (SW 참여제한 문구 부재)
    if facts["SW사업"] == "예" and complete:
        out["v20"] = {"위반여부": 0 if quoted(facts["SW참여제한_인용"]) else 1, "근거문구": None}

    # v14~v18 (일반 물품·용역의 금액 구간별 기업규모)
    band = ["v14", "v15", "v16", "v17", "v18"]
    if kind in ("경쟁제품", "기타"):             # 고시 이름 일치만으로 내리지 않음 (특이사항 조건 등으로 과대 일치)
        out.update({v: {"위반여부": 0, "근거문구": None} for v in band})
        return out
    price = parse_amount(rec.get("meta") or {})
    if price is None or not size_ok:
        return out
    hit = None
    if size == "제한없음":
        if not complete or facts["우선조달_예외"] != "없음":
            return out
        hit = "v18" if price < ONE_EOK else ("v16" if price < GOSI_AMOUNT else None)
    elif price >= GOSI_AMOUNT:
        hit = "v14"
    elif price >= ONE_EOK:
        hit = "v15" if size == "소기업소상공인만" else None
    else:
        hit = "v17" if size == "중소기업자" else None
    out.update({v: {"위반여부": 0, "근거문구": None} for v in band})
    if hit:
        out[hit] = {"위반여부": 1, "근거문구": None if hit in ABSENCE else facts["자격_인용"]}
    return out


def merge_group_outputs(texts: List[str], groups: List[List[str]]) -> str:
    """그룹별 출력을 24항목 JSON 하나로 합칩니다. 그룹 밖 항목이 섞여 나와도 무시합니다."""
    merged: Dict[str, Any] = {}
    for text, group in zip(texts, groups):
        parsed, missing = parse_judgment(text)
        for v in group:
            if v not in missing:
                merged[v] = parsed[v]
    return json.dumps(merged, ensure_ascii=False)


# ===== 5. 모델 러너 (vLLM offline / mock) =====
class VLLMRunner:
    """평가 서버의 모델을 vLLM offline API로 실행합니다."""

    def __init__(self, schema: Dict[str, Any], model_dir: str = MODEL_DIR, quant: Optional[str] = QUANT,
                 max_tokens: int = MAX_TOKENS, seed: int = SEED, gpu_mem: float = 0.92, tp: int = 1):
        t0 = time.time()
        import vllm                                    # --mock 실행 시 vllm이 없어도 되도록 지연 import
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        log(f"vllm {vllm.__version__} · 모델 {model_dir} · quant={quant} · max_model_len={MAX_MODEL_LEN}")
        kw = dict(model=model_dir, tokenizer=model_dir, max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=gpu_mem, seed=seed, tensor_parallel_size=tp, dtype="auto")
        if quant:
            kw["quantization"] = quant
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        def make_sp(sch):
            return SamplingParams(
                temperature=0.0, max_tokens=max_tokens, seed=seed,
                structured_outputs=StructuredOutputsParams(json=sch, disable_any_whitespace=True),
                logprobs=5 if ITEM_THRESHOLDS else None,
            )
        self._make_sp = make_sp
        self._sp_cache: Dict[int, Any] = {}
        self.sp = make_sp(schema)
        self.probs: Dict[int, Dict[str, float]] = {}   # id(messages) → 항목별 P(위반=1)
        self.load_seconds = time.time() - t0

    def sampling_params(self, schema: Optional[Dict[str, Any]]):
        if schema is None:
            return self.sp
        if id(schema) not in self._sp_cache:
            self._sp_cache[id(schema)] = self._make_sp(schema)
        return self._sp_cache[id(schema)]

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:   # transformers 버전에 따라 dict가 반환되는 경우
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def chat(self, batch: List[List[Dict[str, str]]], schemas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """schemas를 주면 요청마다 그 스키마로 제약 디코딩합니다(그룹 질의)."""
        sp = self.sp if schemas is None else [self.sampling_params(s) for s in schemas]
        outs = self.llm.chat(batch, sampling_params=sp, use_tqdm=False)
        if ITEM_THRESHOLDS:
            for m, o in zip(batch, outs):
                self.probs[id(m)] = item_probs(o.outputs[0]) if o.outputs else {}
        return [o.outputs[0].text if o.outputs else "" for o in outs]


VALUE_KEY = '"위반여부":'
ITEM_KEY = re.compile(r'"(v\d{1,2})":\{')


def item_probs(completion) -> Dict[str, float]:
    """생성 토큰을 따라가며 각 항목 `"위반여부":` 다음 자리의 P(1) = p1/(p0+p1)을 구합니다.

    같은 호출의 0/1 토큰 확률에 항목별 고정 임계값을 거는 후처리용입니다(학습한 분류기가 아님).
    """
    out: Dict[str, float] = {}
    buf = ""
    for tid, lp in zip(completion.token_ids, completion.logprobs or []):
        if buf.endswith(VALUE_KEY):
            m = list(ITEM_KEY.finditer(buf))
            if m:
                cand = {x.decoded_token.strip(): x.logprob for x in lp.values() if x.decoded_token is not None}
                l0, l1 = cand.get("0", -1e9), cand.get("1", -1e9)
                out[m[-1].group(1)] = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, l0 - l1))))
        tok = lp.get(tid)
        buf += tok.decoded_token if tok is not None and tok.decoded_token is not None else ""
    return out


def apply_thresholds(judgment: Dict[str, Dict[str, Any]], probs: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """항목별 임계값으로 위반 여부를 다시 정합니다. 확률이 없는 항목은 모델 판정을 그대로 둡니다.
    0→1로 바뀐 항목은 모델이 근거를 내지 않았으므로 근거 문구가 빈칸입니다."""
    if not ITEM_THRESHOLDS or not probs:
        return judgment
    out = {}
    for v in ITEMS:
        cell = judgment[v]
        if v in probs and v in ITEM_THRESHOLDS:
            hit = int(probs[v] >= ITEM_THRESHOLDS[v])
            cell = {"위반여부": hit, "근거문구": cell["근거문구"] if hit else ""}
        out[v] = cell
    return out


class MockRunner:
    """모델 없이 입력·출력 및 제출 형식을 확인합니다."""
    load_seconds = 0.0

    def __init__(self, schema: Dict[str, Any], **_):
        pass

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(m["content"]) for m in messages) // 2     # Mock 실행용 간이 추정치

    def _one(self, _messages: List[Dict[str, str]]) -> str:
        out = {v: {"위반여부": 0, "근거문구": None} for v in ITEMS}
        return json.dumps(out, ensure_ascii=False)

    def chat(self, batch: List[List[Dict[str, str]]], schemas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        return [self._one(m) for m in batch]


def fit_to_budget(rec: Dict[str, Any], system_prompt: str, runner, max_chars: int,
                  budget: int = PROMPT_BUDGET, builder=None) -> Tuple[List[Dict[str, str]], int, int]:
    """설정된 토큰 예산에 맞게 문서 글자 수를 조정합니다. builder(max_chars)를 주면 그것으로 메시지를 만듭니다."""
    while True:
        msgs = builder(max_chars) if builder else build_messages(rec, system_prompt, max_chars)
        n = runner.count_tokens(msgs)
        if n <= budget or max_chars <= 2000:
            return msgs, n, max_chars
        max_chars = int(max_chars * min(0.85, budget / n * 0.95))


def run_chunk(runner, batch: List[List[Dict[str, str]]],
              schemas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """배치 실패 시 건별로 재시도하고, 처리하지 못한 건은 빈 출력으로 반환합니다."""
    try:
        return runner.chat(batch, schemas) if schemas else runner.chat(batch)
    except Exception as e:
        log(f"  ! 청크({len(batch)}건) 실패 → 건 단위 재시도: {type(e).__name__}: {str(e)[:160]}")
    outs = []
    for k, m in enumerate(batch):
        try:
            outs.append((runner.chat([m], [schemas[k]]) if schemas else runner.chat([m]))[0])
        except Exception as e:
            log(f"  ! 건 단위 실패 → 빈 출력: {type(e).__name__}: {str(e)[:160]}")
            outs.append("")
    return outs


# ===== 6. 파싱·후처리 =====
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    for cand in (text, *(m.group(1) for m in FENCE.finditer(text))):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


ITEM_CELL = re.compile(
    r'"(v\d{1,2})"\s*:\s*\{\s*"위반여부"\s*:\s*([01])\s*,\s*"근거문구"\s*:\s*(null|"((?:[^"\\]|\\.)*)")\s*\}')


def salvage_items(text: str) -> Optional[Dict[str, Any]]:
    """max_tokens에 걸려 JSON이 끊긴 출력에서 완결된 항목 셀만 건집니다.

    전체 파싱이 실패해도 앞쪽 항목은 온전히 나와 있으므로, 그 판정까지 버리지 않습니다.
    """
    out: Dict[str, Any] = {}
    for m in ITEM_CELL.finditer(text or ""):
        ev = None
        if m.group(3) != "null":
            try:
                ev = json.loads('"' + m.group(4) + '"')
            except json.JSONDecodeError:
                ev = None
        out[m.group(1)] = {"위반여부": int(m.group(2)), "근거문구": ev}
    return out or None


def parse_judgment(text: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """모델 출력을 24항목 판정으로 정리합니다. 빠진 항목은 0/None으로 채우고 결손 목록을 함께 반환합니다."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        obj = salvage_items(text)
    if isinstance(obj, dict) and isinstance(obj.get("판정"), dict):
        obj = obj["판정"]
    out, missing = {}, []
    for v in ITEMS:
        raw = obj.get(v) if isinstance(obj, dict) else None
        if not isinstance(raw, dict):
            missing.append(v)
            out[v] = {"위반여부": 0, "근거문구": None}
            continue
        hit = raw.get("위반여부", raw.get("violation", 0))
        if isinstance(hit, bool):
            hit = int(hit)
        if isinstance(hit, str):
            hit = 1 if hit.strip() in ("1", "위반", "true", "True") else 0
        if hit not in (0, 1):
            hit = 1 if hit else 0
        ev = raw.get("근거문구", raw.get("evidence"))
        if ev is not None and not isinstance(ev, str):
            ev = str(ev)
        out[v] = {"위반여부": int(hit), "근거문구": ev}
    return out, missing


def clean_evidence(ev: Optional[str], src: str) -> str:
    """근거문구 셀 규약: NFC · 앞뒤 공백 제거 · 500자 상한 · 수식 접두(=,+,@)면 빈칸 ·
    원문 부분문자열이 아니면 빈칸(원문에 없는 근거는 채점에서 인정되지 않습니다)."""
    if not ev:
        return ""
    ev = unicodedata.normalize("NFC", ev).replace("\r", "").strip()
    if not ev or ev[0] in "=+@":
        return ""
    if ev not in src:                              # 공백·줄바꿈만 다르면 원문 표기로 복원
        ev = restore_quote(ev, src) or ""
    return ev[:EVIDENCE_MAX]


def postprocess(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """후처리: ① 부재탐지 5항목 근거 빈칸 고정 ② 위반이 아니면 근거 빈칸 ③ 근거문구 원문 대조(NFC)"""
    src = unicodedata.normalize("NFC", full_text(rec))
    out = {}
    for v in ITEMS:
        cell = dict(judgment.get(v, {"위반여부": 0, "근거문구": None}))
        hit = 1 if cell.get("위반여부") == 1 else 0
        ev = "" if (hit == 0 or v in ABSENCE) else clean_evidence(cell.get("근거문구"), src)
        out[v] = {"위반여부": hit, "근거문구": ev}
    return out


# ===== 6-1. 메타 기반 적용 조건 게이트 =====
# 항목표의 적용 조건상 성립할 수 없는 항목을 0으로 둡니다(1→0만, 새 양성은 만들지 않음).
# 금액 경계는 메타 금액과 공고 금액 차이를 감안해 여유(margin)를 둡니다.
def parse_amount(meta: Dict[str, Any]) -> Optional[float]:
    """입찰추정가격(없으면 배정예산금액/1.1). 알 수 없으면 None — 게이트를 적용하지 않습니다."""
    def num(x: Any) -> Optional[float]:
        if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0:
            return float(x)
        if isinstance(x, str):
            s = re.sub(r"[^\d.]", "", x)
            try:
                return float(s) if s and float(s) > 0 else None
            except ValueError:
                return None
        return None
    est = num(meta.get("입찰추정가격"))
    if est is not None:
        return est
    bud = num(meta.get("배정예산금액"))
    return bud / 1.1 if bud is not None else None


GATE_MODEL_HINT = re.compile(r"모델\s*명|제조사|제조\s*회사|상표|브랜드|[A-Za-z]{2,}[\s-]?\d{2,}|동등\s*(?:이상|품)")


def gate_closed(rec: Dict[str, Any]) -> Dict[str, str]:
    """성립 불가 항목 → 사유. dev 200건에서 정답 양성을 막지 않는 것을 확인한 조건만 둡니다."""
    m = rec.get("meta") or {}
    closed: Dict[str, str] = {}
    award = str(m.get("낙찰방법") or "")
    law = str(m.get("적용계약법") or "")
    if award and "협상" not in award:
        closed["v22"] = "협상 계약 아님"
        closed["v23"] = "협상 계약 아님"
    if law and "지방" not in law:
        closed["v23"] = "지방계약법 아님"
    # v9(특정 모델명 명시): 물품이 아니고, 규격서가 없고, 모델명·제조사·상표·영문+숫자 모델 표기도 없으면 성립 불가
    # (dev 막힌 문서 35%, 막힌 정답 양성 0, 무라벨 52%)
    if ("물품" not in str(m.get("업무구분") or "") and not any(d["type"] == "규격서" for d in rec["docs"])
            and not GATE_MODEL_HINT.search(full_text(rec))):
        closed["v9"] = "모델명 단서 없음"
    amt = parse_amount(m)
    if amt is not None:
        if amt < GOSI_AMOUNT * 0.9:                      # 고시금액 이상 항목
            closed["v14"] = "고시금액 미만"
        region_limit = LOCAL_REGION_LIMIT if "지방" in law else GOSI_AMOUNT
        if amt < region_limit * 0.9:                     # 지역제한 허용 금액 이상 항목
            closed["v5"] = "지역제한 허용 금액 미만"
        if amt < ONE_EOK * 0.9 or amt >= GOSI_AMOUNT * 1.1:   # 1억 이상~고시금액 미만 항목
            closed["v15"] = closed["v16"] = "1억~고시금액 구간 밖"
    # v15(소기업·소상공인만 제한)는 본문 참가자격에 기업규모 제한 문구가 있어야 성립 (dev 막힌 정답 양성 0)
    if "v15" not in closed and size_class(rec)[0] == "none":
        closed["v15"] = "본문 기업규모 제한 없음"
        if amt >= ONE_EOK * 1.1:                         # 1억 미만 항목
            closed["v17"] = closed["v18"] = "1억 이상"
    return closed


PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|％|퍼센트)")
BID_STAGE = re.compile(r"입\s*찰|투\s*찰|견\s*적\s*서?\s*제\s*출|제\s*안\s*서\s*제\s*출|마\s*감")
CONTRACT_STAGE = re.compile(r"계\s*약\s*(?:시|체\s*결|전)|낙\s*찰\s*자")


def evidence_refutes(item: str, ev: str) -> bool:
    """모델이 인용한 근거가 그 항목의 위반 조건을 스스로 부정하면 True (1→0).

    v19 물품공급 확약서: 위반은 '입찰 시' 제출·보유 요구다. 근거가 계약 단계만 말하고
        입찰 단계 표현이 없으면 위반이 아니다.
    v21 공동계약 지분: 공동이행 구성원 최소지분율 10%(20% 범위 가감 → 8%)보다 낮게 정한 것이 위반이다.
        근거의 지분율이 모두 8% 이상이면 위반이 아니다. 수치가 없으면 판단하지 않는다.
    """
    if not ev:
        return False
    if item == "v19":
        return bool(CONTRACT_STAGE.search(ev)) and not BID_STAGE.search(ev)
    if item == "v21":
        pcts = [float(x) for x in PERCENT.findall(ev)]
        return bool(pcts) and min(pcts) >= 8
    return False


def apply_gates(judgment: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    closed = gate_closed(rec)
    out = {}
    for v in ITEMS:
        cell = judgment[v]
        drop = v in closed or (cell["위반여부"] == 1 and evidence_refutes(v, cell["근거문구"]))
        out[v] = {"위반여부": 0, "근거문구": ""} if drop else cell
    return out


def to_row(rec_id: str, judgment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        row[v] = judgment[v]["위반여부"]
        row[f"e{i}"] = judgment[v]["근거문구"]
    return row


def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """제출 규약에 맞게 한 행을 정리합니다: v는 정수 0/1, 비위반·부재탐지의 e는 빈칸,
    e는 NFC·개행 제거·앞 공백과 수식 접두(=,+,@) 제거·500자 상한."""
    out = {"id": row["id"]}
    for i, v in enumerate(ITEMS, 1):
        try:
            hit = 1 if int(row.get(v) or 0) == 1 else 0
        except (TypeError, ValueError):
            hit = 0
        ev = row.get(f"e{i}") or ""
        ev = unicodedata.normalize("NFC", str(ev)).replace("\r", "")   # 개행은 유지(원문 부분문자열 성질)
        ev = ev.lstrip(" \t=+@").strip()[:EVIDENCE_MAX]
        if not hit or v in ABSENCE:
            ev = ""
        out[v], out[f"e{i}"] = hit, ev
    return out


def empty_row(rec_id: str) -> Dict[str, Any]:
    return to_row(rec_id, {v: {"위반여부": 0, "근거문구": ""} for v in ITEMS})


# ===== 7. submission.csv 저장·자가검증 =====
def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:   # UTF-8(BOM 없음) · RFC4180 quoting
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})


def validate_csv(path: str, expected_ids: List[str]) -> List[str]:
    """자가검증: 열 49 · 행 수 = 입력 건수 · id 유일·일치 · v 0/1 · e 500자 이하 · 부재탐지 e 빈칸"""
    errs: List[str] = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = list(rd)
    if header != COLUMNS:
        errs.append(f"헤더 불일치: {len(header or [])}열 (기대 {len(COLUMNS)})")
        return errs
    if len(rows) != len(expected_ids):
        errs.append(f"행 수 {len(rows)} ≠ 입력 {len(expected_ids)}")
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("id 중복")
    if set(ids) != set(expected_ids):
        errs.append(f"id 집합 불일치 (누락 {len(set(expected_ids) - set(ids))})")
    absence_idx = {COLUMNS.index("e" + v[1:]) for v in ABSENCE}
    for r in rows:
        if len(r) != len(COLUMNS):
            errs.append(f"{r[0]}: 열 수 {len(r)}")
            continue
        if any(x not in ("0", "1") for x in r[1:25]):
            errs.append(f"{r[0]}: 위반여부에 0/1 아닌 값")
        if any(len(x) > EVIDENCE_MAX for x in r[25:]):
            errs.append(f"{r[0]}: 근거문구 {EVIDENCE_MAX}자 초과")
        if any(r[j] for j in absence_idx):
            errs.append(f"{r[0]}: 부재탐지 항목에 근거문구")
        if any(x.startswith(("=", "+", "@")) for x in r[25:]):
            errs.append(f"{r[0]}: 수식 접두 근거문구")
    return errs


# ===== 8. 실행 =====
def run(input_path: str, out_path: str, runner_cls, limit: Optional[int], chunk: int,
        max_chars: int, data_dir: str, **runner_kw) -> Dict[str, Any]:
    t_all = time.time()
    recs = list(iter_records(input_path, limit=limit))
    log(f"입력 {len(recs)}건 ← {input_path}")
    if not recs:
        write_csv([], out_path)
        return {"건수": 0}

    tbl, schema = item_table(data_dir), decode_schema(data_dir)
    system_prompt = build_system_prompt(tbl)
    group_schemas = [restrict_schema(schema, g) for g in ITEM_GROUPS]
    catalog = load_catalog(data_dir) if (FACTS_MODE or CATALOG_RULES) else []
    facts_used = 0
    runner = runner_cls(schema, **runner_kw)
    log(f"모델 로드 {runner.load_seconds:.1f}s · GROUP_MODE={GROUP_MODE} · 임계값 {len(ITEM_THRESHOLDS)}개")

    # 청크 단위로 메시지를 만들고 추론합니다. 남은 시간이 부족하면 이후 청크의 문서 글자 수를 줄입니다.
    t_inf = time.time()
    texts: List[str] = []
    probs: List[Dict[str, float]] = []
    ntok, shrunk, skipped, cur_chars = [], 0, 0, max_chars
    deadline = T_START + TIME_LIMIT_S
    for s in range(0, len(recs), chunk):
        part = recs[s:s + chunk]
        done, left = len(texts), len(recs) - len(texts)
        if done:
            per_rec = (time.time() - t_inf) / done
            remain = deadline - time.time()
            if remain < 30:                              # 최후 수단 — 남은 건은 전항목 0
                texts.extend([""] * left)
                probs.extend([{}] * left)
                skipped += left
                log(f"  ! 시간 부족 → 남은 {left}건 추론 생략")
                break
            need = per_rec * left                        # 축소 전 속도 기준이라 보수적인 추정
            if remain < 300:                             # 공고마다 LLM 호출은 유지하되 입력을 최소로
                cur_chars = EMERGENCY_CHARS
            elif need > remain * 0.9 and cur_chars > MIN_CHARS:
                cur_chars = max(MIN_CHARS, int(cur_chars * remain * 0.8 / need))
                log(f"  ! 예상 {need:.0f}s > 남은 {remain:.0f}s → 문서 상한 {cur_chars:,}자로 축소")
        if GROUP_MODE:
            # 공고마다 그룹 질의를 이어 붙여(공고 순서대로) 한 번에 보냅니다 — 같은 문서가 연달아 와서 prefix 캐시가 맞습니다.
            longest = max(ITEM_GROUPS, key=lambda g: len(group_question(tbl, g)))
            batch, sch, owner = [], [], []
            for k, rec in enumerate(part):
                _, n, mc = fit_to_budget(rec, system_prompt, runner, cur_chars,
                                         builder=lambda c, r=rec: build_group_messages(r, tbl, longest, c))
                ntok.append(n)
                shrunk += int(mc < max_chars)
                for gi, g in enumerate(ITEM_GROUPS):
                    batch.append(build_group_messages(rec, tbl, g, mc))
                    sch.append(group_schemas[gi])
                    owner.append(k)
                if FACTS_MODE:
                    batch.append(build_fact_messages(rec, catalog, mc))
                    sch.append(FACT_SCHEMA)
                    owner.append(k)
            # 1단계: 공고마다 첫 요청만 보내 문서 prefill을 한 번씩 계산·캐시
            # 2단계: 나머지 요청을 보내 캐시된 문서 접두부를 재사용
            per = len(ITEM_GROUPS) + int(FACTS_MODE)
            first = [j for j in range(len(batch)) if j % per == 0]
            rest = [j for j in range(len(batch)) if j % per != 0]
            outs = [""] * len(batch)
            for sel in (first, rest):
                res = run_chunk(runner, [batch[j] for j in sel], [sch[j] for j in sel])
                for j, t in zip(sel, res):
                    outs[j] = t
            rp = getattr(runner, "probs", {})
            for k in range(len(part)):
                idx = [j for j, o in enumerate(owner) if o == k]
                merged = merge_group_outputs([outs[j] for j in idx[:len(ITEM_GROUPS)]], ITEM_GROUPS)
                if FACTS_MODE:
                    decided = decide_from_facts(parse_facts(outs[idx[-1]]), part[k], catalog)
                    if decided:
                        m = json.loads(merged)
                        m.update(decided)
                        merged = json.dumps(m, ensure_ascii=False)
                        facts_used += 1
                texts.append(merged)
                pk: Dict[str, float] = {}
                for j in idx:
                    pk.update(rp.pop(id(batch[j]), {}))
                probs.append(pk)
        else:
            batch = []
            for rec in part:
                m, n, mc = fit_to_budget(rec, system_prompt, runner, cur_chars)
                batch.append(m)
                ntok.append(n)
                shrunk += int(mc < max_chars)
            texts.extend(run_chunk(runner, batch))
            rp = getattr(runner, "probs", {})
            probs.extend(rp.pop(id(m), {}) for m in batch)
        log(f"  {len(texts)}/{len(recs)}건 … {time.time() - t_inf:.0f}s (전체 {time.time() - T_START:.0f}s)")
    inf_seconds = time.time() - t_inf
    if ntok:
        log(f"프롬프트 토큰 중앙값 {sorted(ntok)[len(ntok) // 2]:,} · 최대 {max(ntok):,} · 예산 축소 {shrunk}건 · 생략 {skipped}건")

    # 파싱·후처리 → 행
    rows, invalid, filled, ev_kept, ev_dropped = [], 0, 0, 0, 0
    for rec, text, pr in zip(recs, texts, probs):
        try:
            parsed, missing = parse_judgment(text)
            invalid += int(len(missing) == 24)
            filled += len(missing)
            before = sum(1 for v in ITEMS if parsed[v]["근거문구"] and parsed[v]["위반여부"] == 1 and v not in ABSENCE)
            final = apply_gates(apply_rules(apply_thresholds(postprocess(parsed, rec), pr), rec, catalog), rec)
            kept = sum(1 for v in ITEMS if final[v]["근거문구"])
            ev_kept += kept
            ev_dropped += before - kept
            rows.append(to_row(rec["id"], final))
        except Exception as e:                           # 한 건의 실패가 전체 실행을 막지 않도록
            log(f"  ! {rec['id']} 후처리 실패 → 전항목 0: {type(e).__name__}: {e}")
            rows.append(empty_row(rec["id"]))
    if len(rows) != len(recs):                          # 방어: 행 수가 어긋나면 누락 id를 전항목 0으로 채움
        have = {r["id"] for r in rows}
        rows += [empty_row(r["id"]) for r in recs if r["id"] not in have]
    rows = [sanitize_row(r) for r in rows]

    write_csv(rows, out_path)
    errs = validate_csv(out_path, [r["id"] for r in recs])
    if errs:                                            # 최후 방어: 근거 문구를 모두 비우고 다시 씀
        log(f"[주의] 자가검증 실패 → 근거 문구 제거 후 재작성: {errs[:5]}")
        rows = [{**r, **{e: "" for e in EVID}} for r in rows]
        write_csv(rows, out_path)
        errs = validate_csv(out_path, [r["id"] for r in recs])
    report = {
        "건수": len(recs), "모델로드_s": round(runner.load_seconds, 1), "추론_s": round(inf_seconds, 1),
        "건당_s": round(inf_seconds / len(recs), 2), "전체_s": round(time.time() - t_all, 1),
        "유효JSON": len(recs) - invalid, "메운_항목수": filled,
        "근거_유지": ev_kept, "근거_원문불일치_폐기": ev_dropped, "사실판정_적용": facts_used,
        "출력": out_path, "자가검증": "PASS" if not errs else errs,
    }
    log(json.dumps(report, ensure_ascii=False))
    if invalid:
        log("[주의] 유효 JSON이 아닌 출력이 있습니다. 구조화 출력 설정과 JSON Schema를 확인하세요.")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="24개 항목의 법령 위반 여부 판정 베이스라인")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--input", default=None, help="기본 = <data-dir>/test.jsonl.gz")
    ap.add_argument("--model-dir", default=MODEL_DIR, help="로컬에서는 HF ID(google/gemma-4-26B-A4B-it)도 가능")
    ap.add_argument("--quantization", default=os.environ.get("PPS_QUANT", QUANT),
                    help="채점 서버 = int8_per_channel_weight_only · 'none'이면 미양자화")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=None,
                    help="LLM.chat 한 번에 넘길 공고 수(시간 가드 점검 단위). 기본 64, GROUP_MODE는 16(문서 KV 캐시 보존)")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS, help="문서 글자 수의 초기 상한(토큰 예산에 맞춰 자동 조정)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="모델 없이 흐름만 확인")
    a = ap.parse_args()

    input_path = a.input or os.path.join(a.data_dir, "test.jsonl.gz")
    out_path = os.path.join(a.output_dir, "submission.csv")
    quant = None if str(a.quantization).lower() in ("none", "") else a.quantization
    runner_kw = {} if a.mock else dict(model_dir=a.model_dir, quant=quant, max_tokens=a.max_tokens,
                                       seed=SEED, gpu_mem=a.gpu_mem, tp=a.tp)
    report = run(input_path, out_path, MockRunner if a.mock else VLLMRunner,
                 limit=a.limit, chunk=a.chunk or (16 if GROUP_MODE else 64),
                 max_chars=a.max_chars, data_dir=a.data_dir, **runner_kw)
    if report.get("자가검증") not in ("PASS", None):
        log(f"[주의] 자가검증 오류가 남았지만 제출 파일은 작성됨: {report.get('자가검증')}")
    return 0                                            # CSV를 썼으면 정상 종료(비정상 종료 코드로 제출 전체가 오류 처리되는 것을 방지)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)                                      # vLLM 종료 처리 중 예외가 종료 코드를 바꾸지 않도록

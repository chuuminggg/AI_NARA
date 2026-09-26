#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU(vLLM)에서 제출 script.py의 프롬프트·모델 설정 그대로 dev를 추론하고 원응답을 캐시합니다.

  python tools/gpu_dev_run.py --script submissions/002_20260926/script.py --tag s002
  → cache/api/<tag>.jsonl ({"id","text","prompt_hash","finish"}) · 이후 tools/dev_eval.py --tag <tag>로 CPU 채점

제출 코드가 아닙니다. 평가 서버와 같은 vLLM 0.26.0 · L40S에서 돌려 시간·출력 잘림도 함께 기록합니다.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_runner import ROOT, load_script  # noqa: E402


VALUE_KEY = '"위반여부":'
ITEM_KEY = re.compile(r'"(v\d{1,2})":\{')


def item_probs(completion, id0: int, id1: int) -> Dict[str, float]:
    """생성 토큰을 따라가며 각 항목 `"위반여부":` 바로 다음 자리의 P(1) = p1/(p0+p1)을 구한다.

    logprob은 제약 디코딩 마스크 전의 모델 분포다. 항목별 임계값을 dev로 정하는 데 쓴다.
    """
    out, buf = {}, ""
    lps = completion.logprobs or []
    for tid, lp in zip(completion.token_ids, lps):
        if buf.endswith(VALUE_KEY):
            m = list(ITEM_KEY.finditer(buf))
            if m:
                # 토크나이저의 숫자 토큰 표기(앞 공백 등)에 기대지 않도록 후보의 디코드 문자열로 찾는다
                cand = {x.decoded_token.strip(): x.logprob for x in lp.values() if x.decoded_token is not None}
                l0 = cand.get("0", lp[id0].logprob if id0 in lp else -1e9)
                l1 = cand.get("1", lp[id1].logprob if id1 in lp else -1e9)
                out[m[-1].group(1)] = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, l0 - l1))))
        buf += lp[tid].decoded_token if tid in lp and lp[tid].decoded_token is not None else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=os.path.join(ROOT, "submit", "script.py"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--input", default=os.path.join(ROOT, "open", "dev.jsonl.gz"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "open", "data"))
    ap.add_argument("--model-dir", default=os.environ.get("PPS_MODEL_DIR", "/workspace/models/gemma-4-26B-A4B-it"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--logprobs", type=int, default=5, help="항목별 P(1) 기록용 상위 logprob 개수 (0이면 끔)")
    ap.add_argument("--set", action="append", default=[], help="script 상수 덮어쓰기 KEY=JSON (실험용)")
    a = ap.parse_args()

    s = load_script(a.script)
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(s, k, json.loads(v))
        print(f"override {k}={v}", file=sys.stderr)
    t0 = time.time()
    recs = list(s.iter_records(a.input, limit=a.limit))
    tbl, schema = s.item_table(a.data_dir), s.decode_schema(a.data_dir)
    system_prompt = s.build_system_prompt(tbl)
    runner = s.VLLMRunner(schema, model_dir=a.model_dir, quant=s.QUANT, max_tokens=s.MAX_TOKENS, seed=s.SEED)
    if a.logprobs:
        runner.sp.logprobs = a.logprobs
    id0, id1 = (runner.tok.encode(d, add_special_tokens=False)[-1] for d in ("0", "1"))
    group_mode = bool(getattr(s, "GROUP_MODE", False))
    groups = s.ITEM_GROUPS if group_mode else [None]
    gsch = [s.restrict_schema(schema, g) for g in s.ITEM_GROUPS] if group_mode else [None]
    msgs, sps, owner, ntok = [], [], [], []
    for k, rec in enumerate(recs):
        if group_mode:
            longest = max(s.ITEM_GROUPS, key=lambda g: len(s.group_question(tbl, g)))
            _, n, mc = s.fit_to_budget(rec, system_prompt, runner, s.MAX_CHARS,
                                       builder=lambda c, r=rec: s.build_group_messages(r, tbl, longest, c))
            for gi, g in enumerate(s.ITEM_GROUPS):
                msgs.append(s.build_group_messages(rec, tbl, g, mc))
                sps.append(runner.sampling_params(gsch[gi]))
                owner.append(k)
        else:
            m, n, _ = s.fit_to_budget(rec, system_prompt, runner, s.MAX_CHARS)
            msgs.append(m)
            sps.append(runner.sp)
            owner.append(k)
        ntok.append(n)
    if a.logprobs:
        for sp in {id(x): x for x in sps}.values():
            sp.logprobs = a.logprobs
    t_inf = time.time()
    outs = runner.llm.chat(msgs, sampling_params=sps, use_tqdm=True)
    inf_s = time.time() - t_inf

    path = os.path.join(ROOT, "cache", "api", f"{a.tag}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    truncated = 0
    with open(path, "w", encoding="utf-8") as f:
        for k, rec in enumerate(recs):
            idx = [j for j, o in enumerate(owner) if o == k]
            comps = [outs[j].outputs[0] if outs[j].outputs else None for j in idx]
            finishes = [getattr(c, "finish_reason", None) for c in comps]
            truncated += sum(x == "length" for x in finishes)
            texts = [c.text if c else "" for c in comps]
            text = s.merge_group_outputs(texts, groups) if group_mode else texts[0]
            probs: Dict[str, float] = {}
            if a.logprobs:
                for c in comps:
                    if c:
                        probs.update(item_probs(c, id0, id1))
            h = hashlib.sha1(json.dumps([msgs[j] for j in idx], ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
            f.write(json.dumps({"id": rec["id"], "text": text, "prompt_hash": h,
                                "finish": finishes if group_mode else finishes[0],
                                "out_tokens": sum(len(c.token_ids) for c in comps if c),
                                "p1": probs}, ensure_ascii=False) + "\n")
    rep = {"tag": a.tag, "script": os.path.relpath(a.script, ROOT), "건수": len(recs),
           "모델로드_s": round(runner.load_seconds, 1), "추론_s": round(inf_s, 1),
           "건당_s": round(inf_s / len(recs), 3), "전체_s": round(time.time() - t0, 1),
           "프롬프트토큰_중앙값": sorted(ntok)[len(ntok) // 2], "프롬프트토큰_최대": max(ntok),
           "출력잘림(length)": truncated,
           "1853건_추론_추정_s": round(inf_s / len(recs) * 1853, 0)}
    with open(path.replace(".jsonl", ".report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()

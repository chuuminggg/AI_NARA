#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dev 검증용: submit/script.py와 같은 프롬프트로 Gemma API(OpenAI 호환)를 호출하고 응답을 캐시합니다.

제출 코드가 아닙니다. 평가 서버에서는 script.py가 vLLM으로 같은 프롬프트를 실행합니다.

환경변수
  GEMMA_API_BASE   예) https://openrouter.ai/api/v1 · https://generativelanguage.googleapis.com/v1beta/openai
  GEMMA_API_KEY
  GEMMA_MODEL      예) google/gemma-4-26b-a4b-it (제공처 표기에 맞춤)

사용
  python tools/api_runner.py --tag exp1 --input open/dev.jsonl.gz --workers 4
  → cache/api/exp1.jsonl  (한 줄 = {"id", "text", "prompt_hash"}) · 이미 받은 id는 건너뜁니다.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(path=os.path.join(ROOT, "submit", "script.py")):
    spec = importlib.util.spec_from_file_location("pps_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call_api(messages, base, key, model, max_tokens, retries=6):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError) as e:
            wait = min(120, 5 * 2 ** i)
            print(f"  ! {type(e).__name__}: {str(e)[:120]} → {wait}s 후 재시도", file=sys.stderr)
            time.sleep(wait)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="실험 이름 (cache/api/<tag>.jsonl)")
    ap.add_argument("--input", default=os.path.join(ROOT, "open", "dev.jsonl.gz"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "open", "data"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=None, help="기본 = script.py의 MAX_CHARS")
    a = ap.parse_args()

    base, key, model = (os.environ.get(k) for k in ("GEMMA_API_BASE", "GEMMA_API_KEY", "GEMMA_MODEL"))
    if not (base and key and model):
        sys.exit("GEMMA_API_BASE · GEMMA_API_KEY · GEMMA_MODEL 환경변수를 설정하세요.")

    s = load_script()
    max_chars = a.max_chars or s.MAX_CHARS
    system_prompt = s.build_system_prompt(s.item_table(a.data_dir))
    counter = s.MockRunner({})                           # 토크나이저 없이 글자 수 기반으로 예산을 맞춥니다
    recs = list(s.iter_records(a.input, limit=a.limit))

    out_path = os.path.join(ROOT, "cache", "api", f"{a.tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)}건 중 {len(todo)}건 호출 → {out_path}", file=sys.stderr)

    lock = threading.Lock()

    def work(rec):
        msgs, _, _ = s.fit_to_budget(rec, system_prompt, counter, max_chars)
        h = hashlib.sha1(json.dumps(msgs, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        text = call_api(msgs, base, key, model, s.MAX_TOKENS)
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rec["id"], "text": text, "prompt_hash": h}, ensure_ascii=False) + "\n")
        return rec["id"], bool(text)

    t0, n = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(work, r) for r in todo]):
            rid, ok = fut.result()
            n += 1
            print(f"  {n}/{len(todo)} {rid} {'ok' if ok else 'EMPTY'} · {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

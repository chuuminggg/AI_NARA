#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캐시된 API 응답(cache/api/<tag>.jsonl)을 script.py의 파싱·후처리·게이트로 채점합니다 (CPU만 사용).

  python tools/dev_eval.py --tag exp1              # 게이트 적용
  python tools/dev_eval.py --tag exp1 --no-gates   # 게이트 효과 비교
  python tools/dev_eval.py --gates-only            # 모델 없이 게이트가 정답 양성을 막는지 확인
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_runner import ROOT, load_script  # noqa: E402


def f1(tp, fp, fn):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--input", default=os.path.join(ROOT, "open", "dev.jsonl.gz"))
    ap.add_argument("--labels", default=os.path.join(ROOT, "open", "dev_labels.csv"))
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--gates-only", action="store_true")
    a = ap.parse_args()

    s = load_script()
    labels = {r["id"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
    recs = {r["id"]: r for r in s.iter_records(a.input)}

    if a.gates_only:
        for v in s.ITEMS:
            closed = [i for i, r in recs.items() if v in s.gate_closed(r)]
            killed = [i for i in closed if labels[i][v] == "1"]
            if closed:
                print(f"{v:>4} 게이트 {len(closed):>3}건 · 막힌 정답양성 {len(killed)} {killed}")
        return

    texts = {}
    with open(os.path.join(ROOT, "cache", "api", f"{a.tag}.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                texts[d["id"]] = d["text"]
    ids = [i for i in recs if i in texts]
    print(f"채점 {len(ids)}/{len(recs)}건 · 빈 응답 {sum(1 for i in ids if not texts[i])}건")

    stat = {v: [0, 0, 0] for v in s.ITEMS}
    ev_ok = ev_all = 0
    for i in ids:
        parsed, _ = s.parse_judgment(texts[i])
        final = s.postprocess(parsed, recs[i])
        if not a.no_gates:
            final = s.apply_gates(final, recs[i])
        for v in s.ITEMS:
            p, y = final[v]["위반여부"], int(labels[i][v])
            stat[v][0] += p and y
            stat[v][1] += p and not y
            stat[v][2] += (not p) and y
            if p and v not in s.ABSENCE:
                ev_all += 1
                ev_ok += bool(final[v]["근거문구"])
    scores = {v: f1(*stat[v]) for v in s.ITEMS}
    print(f"Macro F1 = {sum(scores.values()) / 24:.4f}  (게이트 {'OFF' if a.no_gates else 'ON'})")
    print(f"근거문구 원문 일치 {ev_ok}/{ev_all}")
    for v in s.ITEMS:
        tp, fp, fn = stat[v]
        print(f"{v:>4}  F1 {scores[v]:.3f}  TP {tp:>2} FP {fp:>3} FN {fn:>2}")


if __name__ == "__main__":
    main()

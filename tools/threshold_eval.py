#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""항목별 P(위반=1)(gpu_dev_run.py가 기록한 p1)로 임계값 판정의 dev 효과를 추정합니다 (CPU).

  python tools/threshold_eval.py --tag s002 [--script submit/script.py]

비교
  argmax   : 모델이 실제로 낸 0/1 (후처리·게이트 적용) — 기준
  global t : 모든 항목에 같은 임계값
  item t   : 항목별 임계값. dev 양성이 항목당 5~8건뿐이라 그대로 맞추면 과대추정되므로
             5-fold 교차검증 점수(CV)를 함께 낸다. 제출에 쓸 값은 CV로 확인된 방식으로만 고른다.
게이트(gate_closed)로 막힌 항목은 임계값과 무관하게 0.
"""
import argparse
import csv
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_runner import ROOT, load_script  # noqa: E402

GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def f1(tp, fp, fn):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def item_f1(ids, v, pred, labels):
    tp = sum(1 for i in ids if pred[i][v] and labels[i][v] == "1")
    fp = sum(1 for i in ids if pred[i][v] and labels[i][v] == "0")
    fn = sum(1 for i in ids if not pred[i][v] and labels[i][v] == "1")
    return f1(tp, fp, fn), (tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--script", default=None)
    ap.add_argument("--out", default=None, help="항목별 임계값 JSON 저장 경로")
    a = ap.parse_args()
    s = load_script(a.script) if a.script else load_script()
    labels = {r["id"]: r for r in csv.DictReader(io.open(os.path.join(ROOT, "open", "dev_labels.csv"), encoding="utf-8"))}
    recs = {r["id"]: r for r in s.iter_records(os.path.join(ROOT, "open", "dev.jsonl.gz"))}
    rows = [json.loads(l) for l in io.open(os.path.join(ROOT, "cache", "api", f"{a.tag}.jsonl"), encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    closed = {i: s.gate_closed(recs[i]) for i in ids}

    # argmax 기준 (script의 파싱·후처리·게이트 그대로)
    base = {}
    for r in rows:
        parsed, _ = s.parse_judgment(r["text"])
        fin = s.apply_gates(s.postprocess(parsed, recs[r["id"]]), recs[r["id"]])
        base[r["id"]] = {v: fin[v]["위반여부"] for v in s.ITEMS}
    p1 = {r["id"]: r.get("p1") or {} for r in rows}
    cover = sum(len(p) for p in p1.values()) / (24 * len(ids))
    print(f"응답 {len(ids)}건 · p1 기록률 {cover:.1%}")

    def pred_with(th):
        return {i: {v: int(v not in closed[i] and p1[i].get(v, 0.0) >= th[v]) for v in s.ITEMS} for i in ids}

    def macro(pred, sub=None):
        sub = sub or ids
        return sum(item_f1(sub, v, pred, labels)[0] for v in s.ITEMS) / 24

    print(f"argmax(현재 방식) Macro {macro(base):.4f}")
    best_g = max(GRID, key=lambda t: macro(pred_with({v: t for v in s.ITEMS})))
    for t in GRID:
        print(f"  global t={t:<4} Macro {macro(pred_with({v: t for v in s.ITEMS})):.4f}{'  ←' if t == best_g else ''}")

    def fit_items(train):
        th = {}
        for v in s.ITEMS:
            th[v] = max(GRID, key=lambda t: (item_f1(train, v, pred_with({**{x: 1.1 for x in s.ITEMS}, v: t}), labels)[0], -abs(t - best_g)))
        return th

    th_all = fit_items(ids)
    print(f"item t (dev 전체로 맞춤, 과대추정) Macro {macro(pred_with(th_all)):.4f}")
    # 5-fold 교차검증: 각 fold를 나머지로 맞춘 임계값으로 예측해 합친다
    rng = random.Random(0)
    order = ids[:]
    rng.shuffle(order)
    folds = [order[k::5] for k in range(5)]
    cv_pred = {}
    for k in range(5):
        train = [i for j, f in enumerate(folds) if j != k for i in f]
        th = fit_items(train)
        cv_pred.update({i: pred_with(th)[i] for i in folds[k]})
    print(f"item t 5-fold CV Macro {macro(cv_pred):.4f}   (global t={best_g}와 비교해 제출 방식 결정)")
    print("\n항목  argmax TP/FP/FN   global      item t (값)")
    g = pred_with({v: best_g for v in s.ITEMS})
    it = pred_with(th_all)
    for v in s.ITEMS:
        print(f"{v:>4}  {item_f1(ids, v, base, labels)[1]!s:<14} {item_f1(ids, v, g, labels)[1]!s:<11} {item_f1(ids, v, it, labels)[1]} ({th_all[v]})")
    if a.out:
        json.dump({"global": best_g, "item": th_all}, io.open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""제출 버전별 CPU 테스트 (모델 없이).

  python tests/test_versions.py                 # 001, 002 스냅샷 비교
  python tests/test_versions.py 001_20260925    # 특정 스냅샷만

검사
  T1 mock 실행: dev 200건·샘플 10건 자가검증 PASS
  T2 정답 재생: 정답 라벨·근거를 모델 출력으로 넣었을 때 후처리·게이트가 정답을 지우지 않는지 (Macro F1 = 1)
  T3 전부 위반: 모든 항목을 1로 낸 모델 → 게이트가 오탐을 얼마나 지우는지 (하한)
  T4 잡음 모델: 정답 + 무작위 오탐(셀당 10%, 근거는 원문 문장) → 게이트·근거 부정 규칙의 효과
  T5 잘린 출력: max_tokens에 걸려 JSON이 중간에 끊긴 응답 → 행이 어떻게 나오는지
  T6 시간 가드: 가짜 시계로 축소·생략 경로를 실행해 CSV가 유효한지
  T7 재현율 50% 모델: 정답 양성의 절반을 놓치는 가상 모델 → 규칙이 놓친 양성을 되살리는지
T3·T4는 라벨로 만든 가상 모델이다. 실제 Gemma 점수가 아니다.
"""
import csv
import gzip
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")
ITEMS = [f"v{i}" for i in range(1, 25)]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def macro(preds, labels):
    f1s = {}
    for v in ITEMS:
        tp = sum(1 for i in labels if preds[i][v] == 1 and labels[i][v] == "1")
        fp = sum(1 for i in labels if preds[i][v] == 1 and labels[i][v] == "0")
        fn = sum(1 for i in labels if preds[i][v] == 0 and labels[i][v] == "1")
        f1s[v] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return sum(f1s.values()) / 24, f1s


def replay(s, recs, outputs):
    """모델 출력 텍스트 → script.py 파싱·후처리·게이트 → {id: {v: 0/1}}"""
    preds = {}
    catalog = s.load_catalog(os.path.join(OPEN, "data")) if hasattr(s, "catalog_rules") else []
    for rid, rec in recs.items():
        parsed, _ = s.parse_judgment(outputs[rid])
        final = s.postprocess(parsed, rec)
        if hasattr(s, "apply_rules"):                       # script의 run()과 같은 순서
            final = s.apply_rules(final, rec, catalog)
        else:
            if hasattr(s, "catalog_rules"):
                final = s.catalog_rules(final, rec, catalog)
            if hasattr(s, "qualification_rules"):
                final = s.qualification_rules(final, rec)
        if hasattr(s, "apply_gates"):
            final = s.apply_gates(final, rec)
        preds[rid] = {v: final[v]["위반여부"] for v in ITEMS}
    return preds


def as_json(cells):
    return json.dumps({v: {"위반여부": h, "근거문구": e} for v, (h, e) in cells.items()}, ensure_ascii=False)


def sentences(rec):
    txt = "\n".join(d["text"] for d in rec["docs"])
    return [x.strip() for x in re.split(r"[\n.]", txt) if 15 <= len(x.strip()) <= 200]


def t1_mock(path):
    ok = True
    for inp in (os.path.join(OPEN, "dev.jsonl.gz"), os.path.join(OPEN, "data", "test.jsonl.gz")):
        out = tempfile.mkdtemp(prefix="t1_")
        p = subprocess.run([sys.executable, path, "--mock", "--data-dir", os.path.join(OPEN, "data"),
                            "--input", inp, "--output-dir", out],
                           capture_output=True, text=True, encoding="utf-8",
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        ok &= p.returncode == 0 and '"자가검증": "PASS"' in p.stderr
    return ok


def t5_truncated(s, rec):
    full = as_json({v: (1 if v in ("v1", "v2") else 0, None) for v in ITEMS})
    cut = full[: len(full) // 2]                 # 중간에서 끊긴 JSON
    parsed, missing = s.parse_judgment(cut)
    return len(missing), sum(parsed[v]["위반여부"] for v in ITEMS)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


def t6_guard(path, step_s):
    """청크마다 가짜 시계를 step_s초 전진시켜 run()을 끝까지 돌린다."""
    s = load(path, f"guard_{step_s}")
    clock = FakeClock()
    s.time = clock                                # 모듈의 time.time()을 가짜 시계로
    s.T_START, s.TIME_LIMIT_S = 0.0, 6300.0
    logs = []
    s.log = logs.append

    class Runner(s.MockRunner):
        def chat(self, batch, schemas=None):
            clock.now += step_s
            return super().chat(batch)

    out = os.path.join(tempfile.mkdtemp(prefix="t6_"), "submission.csv")
    rep = s.run(os.path.join(OPEN, "dev.jsonl.gz"), out, Runner, limit=None, chunk=64,
                max_chars=s.MAX_CHARS, data_dir=os.path.join(OPEN, "data"))
    shrunk = sum("문서 상한" in m for m in logs)
    skipped = next((m for m in logs if "추론 생략" in m), "")
    return rep["자가검증"], shrunk, skipped


def main():
    names = sys.argv[1:] or ["001_20260925", "002_20260926"]
    labels = {r["id"]: r for r in csv.DictReader(io.open(os.path.join(OPEN, "dev_labels.csv"), encoding="utf-8"))}
    recs = {}
    with gzip.open(os.path.join(OPEN, "dev.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["id"]] = r

    gold = {i: as_json({v: (int(labels[i][v]), labels[i]["e" + v[1:]] or None) for v in ITEMS}) for i in recs}
    ones = {i: as_json({v: (1, None) for v in ITEMS}) for i in recs}
    rng = random.Random(0)
    noisy = {}
    for i, rec in recs.items():
        sents = sentences(rec) or [""]
        cells = {}
        for v in ITEMS:
            if labels[i][v] == "1":
                cells[v] = (1, labels[i]["e" + v[1:]] or None)
            elif rng.random() < 0.10:
                cells[v] = (1, rng.choice(sents))
            else:
                cells[v] = (0, None)
        noisy[i] = as_json(cells)
    # T7: 재현율 50% 모델 — 정답 양성의 절반을 놓치고 음성 셀의 5%를 오탐(근거는 원문 문장)
    rng7 = random.Random(1)
    half = {}
    for i, rec in recs.items():
        sents = sentences(rec) or [""]
        cells = {}
        for v in ITEMS:
            if labels[i][v] == "1":
                cells[v] = (1, labels[i]["e" + v[1:]] or None) if rng7.random() < 0.5 else (0, None)
            elif rng7.random() < 0.05:
                cells[v] = (1, rng7.choice(sents))
            else:
                cells[v] = (0, None)
        half[i] = as_json(cells)

    results = {}
    for name in names:
        path = os.path.join(ROOT, "submissions", name, "script.py")
        s = load(path, "s_" + name)
        for i in recs:
            s.normalize(recs[i])
        r = {"T1 mock PASS": t1_mock(path)}
        m, _ = macro(replay(s, recs, gold), labels)
        r["T2 정답 재생 Macro"] = round(m, 4)
        m, f = macro(replay(s, recs, ones), labels)
        r["T3 전부 위반 Macro"] = round(m, 4)
        m, f = macro(replay(s, recs, noisy), labels)
        r["T4 잡음 모델 Macro"] = round(m, 4)
        r["T4 항목 F1"] = {v: round(x, 3) for v, x in f.items()}
        m, f7 = macro(replay(s, recs, half), labels)
        r["T7 재현율 50% 모델 Macro"] = round(m, 4)
        r["T7 항목 F1"] = {v: round(x, 3) for v, x in f7.items()}
        miss, hits = t5_truncated(s, next(iter(recs.values())))
        r["T5 잘린 출력: 결손 항목/남은 위반"] = f"{miss}/24, {hits}"
        r["T6 가드(2000s/청크): 검증·축소·생략"] = t6_guard(path, 2000)
        r["T6 가드(3000s/청크): 검증·축소·생략"] = t6_guard(path, 3000)
        results[name] = r

    for name, r in results.items():
        print(f"\n## {name}")
        for k, v in r.items():
            if k not in ("T4 항목 F1", "T7 항목 F1"):
                print(f"  {k}: {v}")
    if len(names) == 2:
        for t in ("T4", "T7"):
            a, b = (results[n][f"{t} 항목 F1"] for n in names)
            diff = {v: (a[v], b[v]) for v in ITEMS if a[v] != b[v]}
            print(f"\n## {t} 항목별 차이 ({names[0]} → {names[1]}): {diff or '없음'}")


if __name__ == "__main__":
    main()

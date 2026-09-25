#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""threshold_eval.py --out 으로 저장한 임계값을 submit/script.py의 ITEM_THRESHOLDS 상수로 씁니다.

  python tools/set_thresholds.py --json output/th_s003.json --mode global   # 모든 항목 같은 값
  python tools/set_thresholds.py --json output/th_s003.json --mode item     # 항목별 값
  python tools/set_thresholds.py --off                                      # 끄기
"""
import argparse
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "submit", "script.py")
LINE = re.compile(r"^ITEM_THRESHOLDS: Dict\[str, float\] = .*$", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--mode", choices=["global", "item"], default="global")
    ap.add_argument("--off", action="store_true")
    a = ap.parse_args()
    if a.off:
        th = {}
    else:
        d = json.load(io.open(a.json, encoding="utf-8"))
        th = {f"v{i}": d["global"] for i in range(1, 25)} if a.mode == "global" else d["item"]
    src = io.open(SCRIPT, encoding="utf-8").read()
    if not LINE.search(src):
        raise SystemExit("ITEM_THRESHOLDS 줄을 찾지 못했습니다")
    new = LINE.sub("ITEM_THRESHOLDS: Dict[str, float] = " + json.dumps(th, ensure_ascii=False).replace("\\", "\\\\"), src, count=1)
    io.open(SCRIPT, "w", encoding="utf-8", newline="").write(new)
    print(f"ITEM_THRESHOLDS ← {len(th)}개 항목" + (f" (예: v1={th.get('v1')})" if th else " (꺼짐)"))


if __name__ == "__main__":
    main()

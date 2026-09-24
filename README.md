# AI_NARA

나라장터 자체입찰 공고 법령 위반사항 모니터링 AI 경진대회 (DACON 236754) 작업 저장소입니다.

| 경로 | 내용 |
|---|---|
| `docs/competition_rules_and_evaluation.md` | 대회 평가·규칙 정리 |
| `submit/` | 제출용 `script.py`, `requirements.txt` (zip 루트에 두는 파일) |
| `tools/api_runner.py` | dev 검증용 Gemma API 호출과 응답 캐시 |
| `tools/dev_eval.py` | 캐시된 응답의 dev Macro F1 채점, 게이트 검사 |
| `submissions/<번호>_<날짜>/` | 제출별 코드 사본과 기록(`RECORD.md`) |
| `experiments.md` | 제출·실험 요약표 |

대회 배포 자료(`open/`: 공고 원문·라벨·법령 패키지)는 재배포하지 않으므로 저장소에 포함하지 않습니다.
대회 페이지에서 받아 `open/`에 두고 사용합니다.

```bash
python submit/script.py --mock --data-dir open/data --input open/dev.jsonl.gz --output-dir output
python tools/dev_eval.py --gates-only
```

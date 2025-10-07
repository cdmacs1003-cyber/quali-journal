D3 — 점수/룰 고정 + 재현성 체크 (생성: 2025-10-04 13:48 KST)

[구성]
- tools/classify_and_rank.py  : 점수 계산/결정적 정렬/섹션 분류
- tools/make_daily_report.py  : 하루 리포트 MD 생성 (archive/reports/YYYY-MM-DD_report.md)
- tests/d3_regression_check.py: 간이 회귀 테스트 (최소 개수/섹션 유효성)

[빠른 실행]
1) python orchestrator.py --collect
2) python orchestrator.py --publish
3) python tools/make_daily_report.py --date YYYY-MM-DD --keyword "IPC-A-610"
4) python tests/d3_regression_check.py --date YYYY-MM-DD --keyword "IPC-A-610"

[목표]
- 같은 데이터라면 항상 같은 순서(결정적 정렬)를 보장
- 표준/글로벌/AI/커뮤니티 섹션 자동 분류
- 일일 리포트를 자동으로 만들고, 최소 품질을 경고로 알려줌

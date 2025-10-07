
QualiJournal Admin — 패치 요약 (2025-10-05, KST)

1) 통합 UI
- 위치: admin_quali/index.html
- 라이트/다크 테마를 하나의 파일에서 토글(🌓 버튼). OS 선호도(prefers-color-scheme) 자동 인식 + localStorage 저장.

2) API 정렬(server_quali.py)
- 신규 엔드포인트: /api/enrich/selection, /api/enrich/keyword, /api/report, /api/export/md, /api/export/csv
- flow/keyword가 date 필드도 수용(현재 orchestrator는 내부 today 사용).
- index("/")는 admin_quali/index.html을 우선 서빙.

3) 안전 클린업
- tools/cleanup_legacy.py (기본 드라이런) — --force로 실제 삭제.

4) 실행
- uvicorn admin_quali.server_quali:app --port 8010 --reload
- 브라우저에서 admin_quali/index.html 열고 버튼 사용.

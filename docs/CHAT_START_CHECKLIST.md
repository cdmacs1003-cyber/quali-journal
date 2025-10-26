# CHAT_START_CHECKLIST.md

다음 채팅 시작 시 항상 이 순서:

1) **SSOT 확인**: repo=cdmacs1003-cyber/quali-journal, branch=main
2) **인벤토리 최신화**: `/_inventory/inventory.json` 유무·최신 SHA 확인
3) **DoD 확인(오늘)**: health 200, /api/status 200(토큰), /api/report 200(토큰)
4) **작업 범위 확정**: admin / orchestrator / CI / docs 중 선택
5) **앵커 기준 패치**: 파일경로→앵커→범위→교체/삽입 코드→저장/테스트
6) **PR 생성**: 리뷰어·라벨 지정, CI 통과 확인

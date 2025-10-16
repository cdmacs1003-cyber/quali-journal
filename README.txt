QualiNews 완성 템플릿 (생성: 2025-10-04 13:26 KST)

[배치 위치]
- 이 폴더의 파일들을 프로젝트 루트로 복사
- feeds/* 는 그대로 feeds/ 폴더에 두세요

[바로 실행]
1) python orchestrator.py --collect-community
2) python orchestrator.py --approve-top 20
3) python orchestrator.py --publish-community --format all
4) python orchestrator.py --collect
5) python orchestrator.py --publish

[설명]
- config.json : 승인 게이트/경로/신뢰 도메인/커뮤니티 점수 가중치
- feeds/community_sources.json : Reddit+포럼 수집 설정/필터
- feeds/editor_rules.json : 도메인 가중치(좋은 소스가 위로)
- smt_sources.json : SMT/PCB 공식 뉴스·매거진
- supply_sources.json : EMS/부품/산업 동향 소스
- 커뮤니티_키워드.txt : 커뮤니티 수집 키워드 목록(1줄 1키워드)
- keyword_synonyms.json : 키워드 변형/동의어 매핑


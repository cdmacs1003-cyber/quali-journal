D4 — 원문 요약과 선택적 한국어 번역

[무엇을 하는가]
- tools/enrich_cards.py는 작업 JSON의 원문 요약을 보충합니다.
- 기본 실행은 OpenAI를 호출하지 않습니다. 키가 있어도 같습니다.
- --translate를 명시해야 누락된 한국어 요약을 API로 번역합니다.
  이 옵션은 기존 OPENAI_API_KEY를 사용하며 API 사용료가 발생할 수 있습니다.
- 이미 있는 한국어 요약은 재사용합니다. 한국어 제목만 있다고 영어 요약을
  한국어로 간주하지 않습니다.
- 영어만 있는 옛 summary_ko / ko_summary / summary_kr 및 '요약: English'는
  summary_ko_rejected에 원래 값을 보관하고 한국어 칸에서 제외합니다.
- 번역 실패 시 영어 원문을 한국어 칸에 복사하지 않습니다.

[실행: 저장소 루트에서]
1) API 호출 없이 선정본 요약 정리
   python -m tools.enrich_cards --mode selection

2) API 호출 없이 키워드 작업본 정리
   python -m tools.enrich_cards --mode keyword

3) 실제 번역이 필요한 경우에만 사용
   python -m tools.enrich_cards --mode selection --translate

선정본: selected_articles.json
키워드 작업본: data/selected_keyword_articles.json
지정한 파일이 없으면 기존 동작대로 다른 작업본을 찾습니다.
표준 출력의 source 필드로 실제 처리 파일을 확인하세요.
지원하지 않는 --selection / --date / --keyword / --max는 오류로 처리합니다.
JSON은 임시 파일을 쓴 뒤 교체합니다. 자동 .bak 생성 기능은 없습니다.
기존 알 수 없는 필드는 보존합니다.

[번역 상태: summary_ko_status]
- existing: 기존 한국어 값 재사용. API 번역 성공을 뜻하지 않습니다.
- source_korean: 원문 요약이 이미 한국어여서 복사했습니다.
- translated: 이번 API 호출에서 완결된 한국어 응답을 받았습니다.
- not_requested: --translate를 지정하지 않았습니다.
- missing_key: 기존 API 키가 설정되지 않았습니다.
- disabled: QUALI_DISABLE_OPENAI=1로 호출을 껐습니다.
- no_source: 번역할 원문이 없습니다.
- unavailable: 호출 실패 또는 비정상·잘린·영어 응답입니다.

출력 JSON의 ok=true는 파일 처리가 완료되었다는 뜻입니다.
한국어 요약이 모두 번역됐다는 뜻이 아닙니다. korean_available과 각 기사의
summary_ko_status를 함께 보세요. 한글 포함 여부는 최소 언어 검사이며
번역의 사실성·정확성까지 보증하는 검사는 아닙니다.

[기존 유틸과 엔진]
- tools/_d4_utils.py의 enrich_item()/to_ko()를 직접 호출하면 기존처럼
  키가 있는 경우 번역을 시도합니다. 실패하면 한국어 결과는 빈 값입니다.
- D4 모델 기본값 gpt-4o-mini, engine_core.py 기본값 gpt-4o는 유지합니다.
  OPENAI_TRANSLATE_MODEL 또는 기존 엔진 features.translate_model 설정을 사용합니다.
- 엔진의 본문·제목·요약·편집장 문구 API 호출은 발행 모드에서만 가능합니다.
  QUALI_DISABLE_OPENAI=1 또는 키 없음이면 모든 경로에서 호출하지 않습니다.
- 편집장 문구는 API 사용 불가 시 기존 규칙 기반 문구를 사용합니다.
- HTTP 401/429/500, 연결 실패, 잘린 응답을 성공한 번역으로 저장하지 않으며
  자동 재시도를 추가하지 않았습니다. 키나 원문은 오류 로그에 출력하지 않습니다.

[외부 호출 없는 검증]
기존 실행 의존성(requests, beautifulsoup4, pydantic)이 있는 환경에서:
   python -m unittest discover -s tools/tests -p test_translation_integrity.py -v
합성 데이터와 모의 응답을 사용하며 네트워크 연결은 테스트에서 차단합니다.

[배포 범위]
이 소스 수정은 운영 서버 반영이나 실제 API 성공의 증거가 아닙니다.
docs/infra/cloudbuild.yaml, GitHub 배포 워크플로, 루트 Dockerfile의 관계와
운영 전 확인 사항은 docs/SSOT_Admin_DoD.md의 '뉴스 번역 경로' 절을 따릅니다.

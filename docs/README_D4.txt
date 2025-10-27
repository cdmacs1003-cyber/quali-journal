D4 — 요약/번역(선택) 파이프 (생성: 2025-10-04 14:25 KST)

[설명]
- 기사 본문/요약 텍스트에서 간단한 '추출 요약(Extractive)'을 만들어 summary_en 필드에 넣습니다.
- 환경변수 OPENAI_API_KEY가 있으면, 같은 내용을 한국어로 짧게 번역(summary_ko)을 자동 생성합니다.
- 결과는 원본 JSON을 백업(.bak)한 뒤, 제자리(overwrite)로 저장합니다.

[파일]
- tools/_d4_utils.py        : 문장 분리/추출 요약/번역(OpenAI 옵션) 유틸
- tools/enrich_cards.py      : CLI. selection/keyword JSON을 받아 카드에 summary_en/summary_ko 필드 추가

[실행(초딩 버전)]
1) '선정본(selected_articles.json)' 카드 요약/번역 붙이기
   python -m tools.enrich_cards --selection selected_articles.json --max 20

2) '키워드 특집(YYYY-MM-DD_KEYWORD.json)' 카드 요약/번역 붙이기
   python -m tools.enrich_cards --date 2025-10-04 --keyword "IPC-A-610" --max 20

3) 결과 확인
   - JSON 파일을 열어 카드에 summary_en / summary_ko 필드가 생겼는지
   - 번역은 OPENAI_API_KEY가 있어야 생성(없으면 영어 요약만)

[참고]
- 번역 모델 기본값은 gpt-4o-mini 입니다. 환경변수 OPENAI_TRANSLATE_MODEL로 바꿀 수 있습니다.

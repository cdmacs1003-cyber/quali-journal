QualiJournal Admin — 고도화 대비 운영 안내 (Complete)

# 0. 개요
- 목적: 하루 한 키워드 뉴스의 “수집 → 승인(HiTL) → 동기화 → 발행”을 단일 관리자에서 빠르고 안정적으로 수행.
- 이 문서는 설치/실행부터 UI/API, 원클릭 자동 플로우, 접근성(WCAG), HTTP 표준(RFC 7231), 보안/장애 대응까지 ‘한 파일’로 완결합니다.
- 모든 절차는 승인 ≥ 15(품질 게이트) 원칙을 유지하며, 자동화는 ‘사람 검수 보조’로 동작합니다.
  (기존 README의 목표·흐름을 유지하면서 라이트/다크 UI, 원클릭 플로우, 승인 UI 연동을 정리) 〔근거: 기존 관리자 README 요지〕 〔ref: turn15file0〕
  (접근성·표준·API 구조·비동기·버저닝 권고 반영) 〔ref: turn15file1〕

────────────────────────────────────────────────────────────────────────

# 1. 설치
Python 3.10+ 권장
> (PowerShell)
pip install fastapi uvicorn pydantic

# 2. 폴더 배치(권장)
project-root/
  orchestrator.py
  selected_articles.json
  data/selected_keyword_articles.json
  archive/selected_community.json
  archive/YYYY-MM-DD_KEYWORD.{json|md|html}
  tools/ (rebuild_kw_from_today.py, sync_selected_for_publish.py 등)
  admin/                 ← 관리자 전용 폴더 (이 문서 기준)
    server_quali.py
    index_lite.html

- 관리자 서버는 ‘루트’를 기준으로 JSON/도구를 찾습니다. (루트에서 실행 권장)
- index_lite.html은 라이트/다크 테마를 토글하며, API 기본 주소는 http://127.0.0.1:8010 입니다. 〔ref: turn15file0〕

# 3. 실행
(루트에서)
uvicorn admin.server_quali:app --port 8010 --reload
브라우저: http://127.0.0.1:8010

TIP
- 모듈 인식 불가 시: set PYTHONPATH=%CD%
- 포트 충돌 시: --port 8765 등 변경

────────────────────────────────────────────────────────────────────────

# 4. UI(관리자) 사용 흐름
(1) [새로고침] → KPI 확인: 선정본 총량/승인수/커뮤니티 후보/키워드 특집 수, 게이트(승인≥15) 통과 여부 표시.
(2) [원클릭(커뮤니티)] 또는 [커뮤니티 수집] 실행.
(3) [승인 UI 열기] → 127.0.0.1:8765 에서 체크박스/코멘트 입력 → [저장].
(4) [원클릭(키워드)] 또는 [발행]으로 키워드 특집 산출물 생성.
(5) 결과 파일: archive/YYYY-MM-DD_KEYWORD.(json|md|html)

- UI는 단일 파일(index_lite.html)로 라이트/다크 테마를 제공(유지보수성↑). 〔ref: turn15file1〕
- 버튼들은 POST(상태 변경) / GET(조회) 의미를 지켜 호출합니다. 〔ref: turn15file1〕

────────────────────────────────────────────────────────────────────────

# 5. API 요약 (v1)
※ HTTP 메서드 의미 준수(조회=GET, 상태변경=POST). 〔ref: turn15file1〕

GET  /api/community?only_pending=true
  - 커뮤니티 후보 조회(루트 selected_community.json → archive 후보 → 작업본 community).
  - 응답: {date, keyword, total, approved, pending, articles:[...]}

POST /api/community/save
  - 승인/코멘트 저장, 저장 직후 발행본(selected_articles.json) 동기화 수행.
  - 입력: {changes:[{id,approved,editor_note}]}
  - 응답: {saved, synced, sync_log}

GET  /api/status?date=YYYY-MM-DD&keyword=...
  - KPI/경로: selection_total/approved, community_total, keyword_total, gate_pass 등.

POST /api/publish-keyword
  - 키워드 즉시 발행(동일 키워드 중복 발행 시 이전본을 HHMM 타임스탬프로 백업 후 새 결과 저장).

원클릭 플로우(운영 단축):
POST /api/flow/daily       → 커뮤니티 수집 → TopN 자동 승인 → 동기화
POST /api/flow/community   → 커뮤니티 수집만
POST /api/flow/keyword     → (옵션)외부 RSS 포함 키워드 수집 → TopN 승인 → 동기화 → 발행

권장 네이밍/버전:
- 장기적으로 /api/v1/... 형태 도입(버저닝), 명사형 리소스 경로 선호(/api/runs/daily 대신 /api/tasks/daily 등). 〔ref: turn15file1〕

비동기 확장(선택):
- 현재는 subprocess.run 동기 실행. 대기시간 감소를 위해 asyncio.create_subprocess_exec로 비동기 전환 가능(응답 즉시, 완료는 폴링/WebSocket). 〔ref: turn15file1〕

────────────────────────────────────────────────────────────────────────

# 6. 하루 운용 레시피(초간단)
[키워드 기준]
1) python orchestrator.py --collect-keyword "IPC-A-610" --use-external-rss
2) python tools/rebuild_kw_from_today.py
3) python orchestrator.py --approve-top 20
4) python tools/sync_selected_for_publish.py
5) python orchestrator.py --publish-keyword "IPC-A-610"
[UI 기준]
1) 새로고침 → KPI 확인
2) 승인 UI 열기 → 체크/코멘트 후 저장
3) 발행 버튼(또는 원클릭[키워드])

────────────────────────────────────────────────────────────────────────

# 7. 접근성·UI 품질 가드 (WCAG 2.1)
- 텍스트: 배경 대비 비율 최소 4.5:1(대텍스트 3:1 허용).
- 버튼/입력: 동일 대비 규칙 적용, 비활성/참고 문구도 3:1 이상 권고.
- 스크린리더: 모든 입력에 <label for="...">, 모든 버튼에 aria-label/type="button".
- prefers-color-scheme 감지 → 기본 테마 자동 적용 + localStorage 저장.
※ 위 가이드는 WCAG 원칙에 따른 권장 사항입니다. 〔ref: turn15file1〕

────────────────────────────────────────────────────────────────────────

# 8. 보안·권한
- 내부망/개발기: CORS 전체 허용 가능.
- 외부 공개 시: 
  1) 관리자 페이지와 API를 토큰(헤더) 기반으로 보호(OAuth2/JWT 권장),
  2) HTTPS(프록시) 필수,
  3) 중요 엔드포인트(발행·삭제)는 2단계 확인/로그 남김. 〔ref: turn15file1〕

────────────────────────────────────────────────────────────────────────

# 9. 장애/오류 대응(현장 스크립트)
A) JSON BOM 오류
   - 증상: “Unexpected UTF-8 BOM”.
   - 조치: UTF-8(BOM 없음)로 재저장. 코드에서는 utf-8-sig 리더로 내성 확보.

B) 경로 문제(0건 표기)
   - 서버를 루트에서 실행하거나, 코드에서 루트 고정 경로(BASE.parent) 사용.
   - community JSON은 루트/archive 후보 순서로 탐색.

C) 중복 발행 덮어쓰기
   - 발행 전 기존본을 HHMM 타임스탬프로 자동 백업.

D) API 응답 지연
   - 비동기 프로세스 실행(권장), 프런트에서 진행률 표시/폴링.

────────────────────────────────────────────────────────────────────────

# 10. 체크리스트 (운영/품질 게이트)
□ 승인 ≥ 15 충족(게이트 통과)  
□ 저장 직후 발행본 동기화(log 확인)  
□ 같은 날 재발행 시 기존본 타임스탬프 백업  
□ UI 대비/레이블/aria 점검(라이트/다크 모두)  
□ 외부 배포 시 토큰/HTTPS/로그 활성화

────────────────────────────────────────────────────────────────────────

# 11. 변경 이력 (요약)
- 2025-10-05: 관리자 고도화 가이드 통합(접근성·HTTP 표준·비동기 권고·원클릭 플로우·경로/인코딩 내성).
- 2025-10-04: Lite Admin(Black) 정착, 원클릭 플로우/승인 UI 연동, 기본 KPI 위젯. 〔ref: turn15file0〕


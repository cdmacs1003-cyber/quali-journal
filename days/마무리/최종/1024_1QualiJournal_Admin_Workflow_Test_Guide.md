# 퀄리저널 관리자 페이지 **실전 워크플로우 시험 가이드북** (v2025‑10‑22)

> 대상: 운영자(초보/초딩 버전), 개발자, SRE  
> 시험 환경: **https://admin.standardai.co.kr** (Cloud Run 도메인 매핑)  
> 인증 토큰(테스트): `32776f034dc84d4ba71613e76feec991`  
> 이 가이드는 **실제 기사 수집·선정·보고서 생성**을 최종 도메인에서 검증하기 위한 _실전 테스트_ 절차입니다.  
> 배경과 필요 수정사항은 내부 보고서에 정리되어 있습니다. (인증/새로고침, KPI 동기화, 배포/도메인/캐시 등)【turn11file4】, UI/API 동기화·파일 점검 항목은 개발 파일 분석 보고서 참조【turn11file3】.

---

## 0) 합격 기준(운영 투입 OK)

- **인증**: 토큰 저장 후 배지 “인증됨(초록)” 표시, `GET /api/status` 200.  
- **KPI**: 새로고침/자동새로고침 동작, Gate 슬라이더 조정 → KPI 반영.  
- **키워드 파이프라인**: 키워드 실행 → `status`/`flow` 스트림 로그 도착 → 보고서/요약/선정 파일 생성 경로가 화면에 표시.  
- **Export**: `Export Markdown`/`Export CSV` 정상 다운로드(비어있지 않음).  
- **배포/도메인**: 최신 리비전이 100% 트래픽, admin 도메인에서 최신 HTML/JS 서빙.  

불합격 사유(예): 인증 미갱신, 401 연속, KPI 0 고정, 버튼 무반응, 보고서/CSV 빈 파일, 옛 HTML 서빙 등【turn11file4】.

---

## 1) 준비물 (복붙만 하면 되는 PowerShell)

> **Windows PowerShell** 기준. (관리자 권장)

```powershell
# ==== 환경 값: 프로젝트/리전/서비스 ====
$PROJECT   = "quali-journal-prod"
$REG_TYO   = "asia-northeast1"     # 도쿄 (도메인 매핑 지원)
$REG_SEO   = "asia-northeast3"     # 서울
$LIVE      = "quali-admin-domap"   # 운영 도메인 가리키는 Cloud Run 서비스
$ADMIN     = "quali-journal-admin" # 백오피스 API 서비스(소스 빌드)
$TOKEN     = "32776f034dc84d4ba71613e76feec991"

# ==== 편의 함수 ====
function Code($x){ Write-Host ">> $x" -ForegroundColor Cyan }

# ==== 헬스체크: 서비스 URL / 토큰 / 상태 ====
Code "서비스 실 URL"
$URL = (gcloud run services describe $LIVE --region $REG_TYO --format="value(status.url)")
$URL

Code "헬스체크 (200 기대)"
curl.exe -s -o NUL  -w "%{http_code}\n"  "$URL/health"

Code "상태 조회 (토큰 필요, 200 기대)"
curl.exe -s -o NUL  -w "%{http_code}\n"  -H "Authorization: Bearer $TOKEN" "$URL/api/status"
```

**기대값**
- `/health` 200, `/api/status` 200 → 인증·API 정상.  
- 401이면 토큰 미적용/미인증 상태. UI/HTML에서 토큰 저장/갱신 이슈가 과거에 보고됨【turn11file4】.

---

## 2) **도메인 최신화** & 캐시 회피

클라우드 런 최신 리비전에 100% 트래픽을 보내야 **새 HTML/JS**가 노출됩니다. 옛 리비전/캐시 때문에 버튼이 안 먹은 사례가 있었습니다【turn11file4】.

```powershell
# 도쿄 리전(도메인 매핑) 최신 리비전에 100% 트래픽
gcloud run services update-traffic $LIVE --region $REG_TYO --to-latest

# 브라우저 강제 갱신용 쿼리스트링 (캐시 회피)
$R = Get-Random
Start-Process "https://admin.standardai.co.kr/?v=$R"
```

> 실패 시: `update-traffic` 후에도 UI가 구버전처럼 보이면, 다시 강제 새로고침(CTRL+F5) 또는 `?v={랜덤}`을 붙여 열어 확인.

---

## 3) **초딩 버전** UI 시험 (마우스만)

1. 페이지 열기: `https://admin.standardai.co.kr/?v=랜덤숫자`  
2. 상단 **ADMIN_TOKEN** 칸에 토큰 붙여넣기 → **[입력]** → 배지 “**인증됨**” 확인.  
   - 배지/상태가 안 바뀌면 **[토큰 검증]**/**[해제]** 버튼도 눌러보고, 다시 입력.  
   - 과거 UI 버그: 저장 직후 UI 갱신 누락·버튼 바인딩 누락 보고됨【turn11file4】.  
3. 왼쪽 **KPI 패널**에서 **[KPI 새로고침]** / 상단 자동 새로고침 시간이 줄어드는지 확인.  
4. **게이트 슬라이더**를 15→10으로 바꿔보기 → 상단 KPI “승인본 수/게이트 통과 여부(Ready)”가 즉시 반영되는지 확인. (서버는 `gate_required`와 상태 카운트를 반환해야 함【turn11file3】)  
5. 가운데 **키워드 특집** → 키워드 입력(예: `ipc-a-610`) → **[수집→승인Top20→발행]** 클릭.  
   - 하단 **비동기 작업 로그**에 “collect… approve… publish…” 단계 로그가 쭉 찍혀야 정상.  
   - 오른쪽 개발자도구 Network에서 `status`/`flow`/`report` 이벤트가 순서대로 도착하면 합격.
6. 결과 링크 확인: 중앙 결과 창에 `archive/...md` / `_selected.md` / `.csv` 경로가 나타남 → 눌러서 내용이 **비어있지 않은지** 확인.  
7. **Export Markdown / Export CSV** 버튼도 눌러 파일을 내려 받아 내용 확인 (제목·출처·요약 포함).

**실패 패턴과 즉시 조치**
- 모든 API가 401 → 토큰 저장/전송 실패. UI 토큰 저장·갱신 코드를 점검(3.개발 파일 패치 체크 참조).  
- KPI 0 고정/슬라이더 반영 안 됨 → `/api/status`/`/api/config/gate_required` 응답 확인(섹션 4).  
- 결과 파일이 비어 있음 → 수집 파이프라인/선정 로직이 데이터를 못 만들고 있음(섹션 5).

---

## 4) **API 단위 검증** (복붙 PowerShell)

```powershell
# 날짜/키워드 변수
$DATE = (Get-Date -Format "yyyy-MM-dd")
$KW   = "ipc-a-610"

# 상태
curl.exe -s "$URL/api/status?date=$DATE&keyword=$KW" -H "Authorization: Bearer $TOKEN"

# 게이트 설정 (예: 12)
curl.exe -s -X PATCH "$URL/api/config/gate_required" `
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" `
  -d "{\"value\":12}"

# 키워드 전체 파이프라인 (수집→승인→발행)
curl.exe -s -X POST "$URL/api/keyword/run?keyword=$KW" -H "Authorization: Bearer $TOKEN"

# 요약/번역 (전체·선정)
curl.exe -s -X POST "$URL/api/enrich/keyword?keyword=$KW"  -H "Authorization: Bearer $TOKEN"
curl.exe -s -X POST "$URL/api/enrich/selection?keyword=$KW" -H "Authorization: Bearer $TOKEN"

# 보고서 생성
curl.exe -s -X POST "$URL/api/report?keyword=$KW" -H "Authorization: Bearer $TOKEN"

# 생성물 조회(예)
curl.exe -s "$URL/api/items?state=ready&date=$DATE&keyword=$KW" -H "Authorization: Bearer $TOKEN"
```

**기대**: `status`는 `selected/approved` 수, `state_counts={candidate,ready,rejected}`, `gate_required`, `gate_pass`를 포함해 반환해야 합니다【turn11file3】.

---

## 5) 파이프라인 실동작 점검 (서버 로그 시나리오)

> 관리자 보고서에 따르면, 실제 운영에서 보고서와 CSV가 비어있던 사례가 있었습니다【turn11file4】. 아래를 통해 원인 분기합니다.

1) **수집 성공 여부**  
- `selected_keyword_articles.json`/`selected_articles.json` 파일이 갱신돼야 합니다.  
- `/api/status`의 `state_counts.ready`가 증가해야 합니다【turn11file3】.

2) **선정/발행 로직**  
- Gate(예: 12) 미달 시 발행 비활성화/경고가 UI에 보여야 합니다.  
- 선정본이 0이면 보고서 생성 시 “수집된 기사 없음” 경고를 출력하도록 UX 반영 권장【turn11file4】.

3) **로그 가시성**  
- 하단 비동기 로그가 단계별로 이어져야 하며, 실패 시 **원인 메시지**가 보여야 합니다(네트워크, 크롤러 차단 등).  
- 서버는 상세 로그를 파일로 남기고, `/api/tasks`로 현재 작업 상태를 노출하는 구조 권장【turn11file3】.

---

## 6) **배포/도메인 체크리스트**

- **트래픽 최신화**: `gcloud run services update-traffic $LIVE --region $REG_TYO --to-latest` 필수.  
- **캐시 회피**: 운영 검증 시 URL에 `?v={랜덤}`를 붙여 새 HTML을 강제 로드.  
- **HTML 포함 빌드**: `.gcloudignore` / `.dockerignore`에 **배제 규칙**이 있으면 해제하세요.  
  - (예) 배제 해제 룰:  
    ```text
    !admin/index.html
    !admin/index_lite_black.html
    !admin/**/*.html
    ```
  - 과거 “버튼 무반응”은 구버전 HTML이 배포되거나 빌드에 포함되지 않아 발생【turn11file4】.

---

## 7) **개발 파일 패치 체크(요약)**

> 자세한 항목과 코드 위치는 _개발 파일 분석 및 수정 제안_ 문서 참고【turn11file3】.

- **admin/index.html**  
  - `DOMContentLoaded`에서 `btnRefresh`, `btnKpiRefresh` → `refresh()` 바인딩.  
  - `saveAdminTokenJJ()`에서 `updateAuthUI()`를 **즉시** 호출, `testToken()` 실패 시 경고 표시.  
  - `clearAdminTokenJJ()`는 새로고침 없이 localStorage만 비우도록.  
- **admin/server_quali.py**  
  - `authorize()` 실패 응답을 **일관된 JSON(401)** 으로: `{"detail":"invalid admin token"}`.  
  - `calc_status()` 분리: `state_counts`/`gate_required`/`gate_pass` 계산 정확화.  
  - 비동기 **중복 실행 방지** 및 `/api/tasks` 제공(작업 조회).  
- **파이프라인(orchestrator.py)**  
  - `--collect-approve-publish-keyword` 같이 **원클릭 순차 실행** 제공, 실패 시 중단.  
  - PID/레지스트리로 **동일 키워드 중복 실행 방지**.  
- **포트/런타임**  
  - 컨테이너는 **`PORT` 환경변수**로 바인딩: `uvicorn(..., host="0.0.0.0", port=int(os.getenv("PORT",8080)))`  
  - 오류 메시지 *“failed to start and listen on the port PORT=8080”* 발생 시 진단 포인트.

---

## 8) **문제 대처 빠른 레시피**

- 401 반복 → 토큰 저장/전송/배지 갱신 확인 → `/api/status`를 헤더 포함으로 직접 호출해 비교.  
- KPI 0 고정 → `status` 응답 필드 점검(`state_counts`, `gate_required`) → 슬라이더 PATCH 후 재조회.  
- 버튼 무반응 → 최신 리비전 100%/캐시회피/HTML 포함 빌드 여부 재확인.  
- 보고서·CSV 빈 파일 → 수집/선정 단계부터 상태 값 확인 → 로그에서 실패 지점 파악, 소스/규칙/유의어 최신화.

---

## 9) 부록 – “정상” 예시 화면/응답

- Network 탭에 `status`(200) 연속, `flow` SSE 로그, `report` 응답 `{ ok:true, path:"archive/..." }`.  
- 화면 중앙 “결과: 링크”에 `archive/..._ALL.md`, `_selected.md`, `...csv` 노출 후 클릭 시 내용 확인.  
- KPI 카드 수치가 Gate 반영과 함께 변동.

---

### 참고 문서
- _운영 적합성 평가 보고서 (2025‑10‑22)_: 인증/새로고침/KPI/배포 문제와 개선안【turn11file4】  
- _개발 파일 분석 및 수정 제안_: UI/서버/파이프라인/설정 파일 점검·수정 항목 상세【turn11file3】

---

**이 문서로 테스트를 마친 뒤에도 막히는 단계가 있다면, 막힌 단계 번호와 화면/응답(HTTP 코드 포함)을 알려주세요. 거기서 이어서 진단·패치까지 바로 연결하겠습니다.**

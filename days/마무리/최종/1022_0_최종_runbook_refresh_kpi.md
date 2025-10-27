# QualiJournal Admin — 새로고침/KPI/Ready 점검 **재현 런북**

본 문서는 **한국어 전용**이며, 버튼 배선·API 호출·데이터 경로를 **재현 가능한 절차**로 점검한다.  
성공/실패 기준과 관찰 포인트를 단계별로 명시했다.

---

## 0) 전제(테스트 환경·성공 기준)

- 페이지: `https://admin.standardai.co.kr`  
- 토큰: 제공된 문자열(상단 **ADMIN_TOKEN** 입력창에 붙여넣기)  
- **성공 기준**:  
  - 버튼 클릭 시 **`/api/status`**가 호출되고, 응답 JSON으로 KPI DOM이 갱신(토스트 “상태 갱신 완료”).  
  - **새로고침** / **KPI 새로고침** 버튼 모두 `refresh()` 경로를 타야 함.  【772†source】  
  - Ready 목록은 **`/api/items?state=ready`** 호출로 갱신.  【772†source】

---

## 1) 토큰 저장·헤더 자동 부착 확인(프런트)

1) 페이지 열기 → F12 → **Console**:
```js
localStorage.getItem('ADMIN_TOKEN')     // 저장 여부
```
2) **Network** 탭에서 임의의 `/api/*` 요청 선택 → **Request Headers**에  
   `Authorization: Bearer <토큰>`이 자동 세팅되어야 한다.  
   → 프런트가 `window.fetch`를 래핑하여 `/api/` 경로일 때 토큰 헤더를 주입함. 【772†source】

> 백엔드는 `Authorization: Bearer …`와 `X-Admin-Token`을 모두 허용한다.  
> 토큰(ADMIN_TOKEN 또는 API_TOKEN)이 설정되어 있으면 통과, 없으면 401. 【788†source】

---

## 2) “새로고침” 버튼 배선 확인(프런트)

1) **Elements**에서 `#btnRefresh` 선택 후 Console 실행:
```js
document.getElementById('btnRefresh').click() // 수동 클릭
```
2) **Network**에 **`/api/status?...`** 즉시 발생해야 한다.  
   이 버튼은 `refresh()` 호출에 연결되어 있으며, `refresh()`는 날짜/키워드를 읽어 `/api/status`를 `fetch`하고 받은 JSON으로 KPI DOM을 갱신한다(에러 시 “상태 오류:” 로그). 【772†source】

---

## 3) “KPI 새로고침” 버튼 배선 확인(프런트)

1) **Elements**에서 `#btnKpiRefresh` 선택 후 Console 실행:
```js
document.getElementById('btnKpiRefresh').click()
```
2) **Network**에 **`/api/status`** 호출이 떠야 한다(동일 `refresh()` 경로). 【772†source】

---

## 4) 자동 새로고침 동작 확인(첫 로드 직후)

- DOM 초기화 시 `testToken()`이 호출되고, `AUTO_REFRESH` 기본값을 `1`로 세팅한다(최초 방문).  
  응답이 OK이면 `startAutoRefresh()`가 **30초 간격으로 `refresh()`**를 수행한다. 【772†source】
- 확인 절차: 페이지 새로고침 후 30초 내 `Network`에서 **주기적인 `/api/status`** 호출 발생 여부 확인.

---

## 5) Ready 목록 갱신 흐름(프런트)

- Ready 패널의 **“Ready 불러오기”** 버튼 클릭 → 내부에서 `loadReady()`가 **`/api/items?state=ready&date=...&keyword=...`**를 호출해 카드 리스트를 렌더링한다(실패 시 “Ready 불러오기 실패:” 로그). 【772†source】

> Console 강제 호출 예:
```js
await loadReady()
```

---

## 6) `/api/status`가 KPI를 만드는 방식(백엔드 관점)

- `/api/status`는 내부 **스냅샷**을 읽어 KPI 수치를 계산해 반환한다.  
  - 커뮤니티·키워드 선정 데이터 파일 경로는 **고정 경로 + 후보 경로** 조합(예: `selected_keyword_articles.json`, `archive/selected_community.json`).  
  - 값이 없으면 0으로 폴백한다(항상 안전한 JSON 반환). 【773†source】
- 커뮤니티 스냅샷 로딩은 복수 경로(CAND_COMM)에서 먼저 찾고, 없으면 워크 파일로 폴백한다. 기사마다 id/approved 등의 보정 필드를 채운다. 【788†source】

🟡 **확인 포인트**: 서버에 위 스냅샷 파일이 실제로 존재/갱신되는지(없으면 KPI가 항상 0).

---

## 7) 수집 작업(커뮤니티/키워드) 실행 경로 확인

- 프런트의 실행 버튼들은 서버에서 **`orchestrator.py`**를 **서브프로세스**로 실행한다. 서버 측 `_run_orch()`가 명령행 인자로 작업을 구동하고 stdout/stderr를 수집한다. 【788†source】  
- 오케스트레이터는 인자로 `--collect-community`, `--collect-keyword` 등을 받아 실제 크롤링·선정·쓰기 작업을 수행한다. 【775†source】  
- 외부 사이트 접근은 `requests` 네트워크 호출에 의존하며 타임아웃이 지정되어 있다. **아웃바운드 네트워크가 차단**되면 수집에 실패하고 결과 파일이 생성되지 않는다. 【775†source】

---

## 8) 문제 재현 → 원인 분해 트리(현상별 관찰값)

1) 버튼은 눌리지만 **KPI 0 지속**  
   - 관찰: `/api/status`는 **200**, 그러나 수치는 0 → **스냅샷 데이터 부재/미갱신** 의심. 【773†source】  
2) **Ready 비어 있음**  
   - 관찰: `/api/items?state=ready`는 **200**, `articles: []` → **수집 작업 실패**로 결과 파일 미생성. 【772†source】  
3) **수집 작업 실패**  
   - 관찰: 서버에서 `_run_orch()`는 정상 호출되나, **외부 접근 제한/소스 설정 누락**으로 **stderr**에 오류(결과 파일 없음). 【788†source】

---

## 9) 즉시 해결 절차(우선순위)

### 9-1) 데이터 파일(스냅샷) 유효성 확보
- 다음 파일이 서버 루트에 실제 존재하는지 확인/보강:  
  - `data/selected_keyword_articles.json`, `archive/selected_community.json` (또는 `selected_community.json`)  
- 파일이 유효하면 `/api/status` 반환 수치가 즉시 달라지고, 프런트는 새 JSON으로 DOM을 갱신한다. 【788†source】

### 9-2) 외부 네트워크 통로 확보 또는 오프라인 대체
- `requests`가 접근하는 도메인에 대한 **egress 허용/프록시 설정**.  
- 불가 시 **오프라인 크롤링 결과를 동일 경로에 주기 공급**하여 KPI/Ready 표시를 유지. 【775†source】

### 9-3) 커뮤니티 소스 설정 제공
- `feeds/community_sources.json` 또는 `community_sources.json`을 배치(오케스트레이터의 후보 탐색 경로). 없으면 수집 입력이 비어 작업이 무의미. 【775†source】

### 9-4) 실패 원인 가시화(운영성)
- 서버 `_run_orch()`의 **stderr**를 API 응답/토스트에 요약 표기 → 운영자가 즉시 원인 파악 가능(네트워크/설정/타임아웃 등). 【788†source】

---

## 10) 프런트 배선 요약(검증 체크리스트)

- `#btnRefresh` → `refresh()` 직접 연결, 클릭 시 `/api/status` 호출. 【772†source】  
- `#btnKpiRefresh` → `refresh()` 동일 연결(두 버튼 모두 같은 코드 경로). 【772†source】  
- 자동 새로고침: `AUTO_REFRESH='1'`이면 30초마다 `refresh()` 수행. 【772†source】  
- Ready 갱신: `loadReady()` → `/api/items?state=ready` 호출 후 카드 렌더. 【772†source】  
- `/api/*` 요청은 `window.fetch` 래퍼가 토큰 헤더 자동 주입. 【772†source】

---

## 11) 현장 테스트용 콘솔 스니펫

```js
// 1) 강제 KPI 새로고침
await refresh()   // Network에 /api/status 200, “상태 갱신 완료” 토스트 기대
// 2) 강제 Ready 갱신
await loadReady() // /api/items?state=ready 200, 카드 렌더 또는 "Ready 항목이 없습니다."
// 3) 토큰 주입 정상 확인(200이면 대개 통신 정상)
(await fetch('/api/status')).status
```

---

## 12) 성공 판정

- 두 버튼 모두 `/api/status`를 호출하고, 응답 값으로 KPI DOM이 변한다.  
- Ready 카드가 존재하면 리스트가 그려지고, 발행 버튼을 눌러도 `refresh()` 재호출에 문제가 없다.

---

### 핵심 요약

- **버튼/배선은 정상**(`refresh()` 공통 경로).  
- **증상은 데이터 부재/수집 실패에서 기인**.  
- **즉시 조치**: 스냅샷 파일 존재 확인 + 외부 네트워크/소스 설정 확보 → KPI/Ready 반응 회복.

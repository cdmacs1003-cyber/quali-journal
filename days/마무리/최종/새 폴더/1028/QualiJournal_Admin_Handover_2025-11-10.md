# QualiJournal Admin — 운영 인수인계(Handover) 문서
_작성일: 2025-11-10_

> **요약(TL;DR)**  
> 이 문서는 **새 채팅창**에서 바로 작업을 이어갈 수 있도록, 현재까지의 결정/스크립트/운영 Runbook/미해결 이슈를
> 한 번에 복구할 수 있는 **핵심 맥락**과 **재현 가능한 절차**를 정리한 인수인계 자료입니다.  
> 운영 기준·실전 시험 절차·워크플로우 구현 가이드의 근거는 내부 문서에 정리되어 있습니다
> (각 섹션에 근거 파일을 함께 표기).

---

## 1) 시스템 개요 & 환경 변수

- **도메인(운영 URL)**: `https://admin.standardai.co.kr`
- **GCP Project**: `quali-journal-prod`
- **리전**  
  - `asia-northeast1` (도쿄) — *도메인 매핑 서비스 운영 리전*  
  - `asia-northeast3` (서울) — *소스/백오피스 서비스 리전(빌드/배포)*
- **Cloud Run 서비스명**
  - `quali-admin-domap` — **도메인 매핑** 대상 운영 서비스(도쿄)
  - `quali-journal-admin` — **백오피스(API/소스 빌드)** 서비스
- **테스트 토큰(운영 시험 전용)**: `32776f034dc84d4ba71613e76feec991`  
  _※ 절대 리포지터리에 커밋 금지 (GitHub Secret scanning에 의해 push 차단됨)_

---

## 2) 현재 상태(성공/진행/주의)

### 2.1 최근에 **성공**한 것
- WIF(OIDC) 기반 배포 파이프라인 정비 + SA Key **폴백** 단계 구성.
- Service Worker 캐시 무효화 전략 도입:  
  - HTML(`/`, `/index.html`)은 `no-store, no-cache, must-revalidate`  
  - 해시된 정적 자산(`/assets/*.hash.*`)은 `public, max-age=31536000, immutable`
- 브라우저/프록시 캐시 혼선 방지를 위한 `?v=랜덤` 강제 새로고침 및 `to-latest` 트래픽 전환 습관화.
- GitHub **비밀키 차단** 이슈 해결: `admin/key.json` 제거, `.gitignore` 반영, 재커밋/PR/머지.

### 2.2 **진행 중/주의** 이슈
- **Cloud Run update-traffic 실패 (컨테이너 기동 지연/포트 리슨)**  
  현상: `The user-provided container failed to start and listen on the port defined by PORT=8080 ...`  
  조치 가이드: (a) 최근 리비전 **로그** 확인 → (b) 앱이 `0.0.0.0:$PORT`로 리슨하는지 확인(uvicorn/gunicorn 인자) →  
  (c) 초기 구동 시간이 긴 경우 `--timeout 300` 및 `--cpu-boost`/`--min-instances` 등으로 완화 →  
  (d) 비정상 시 **직전 정상 리비전**으로 트래픽 롤백 후 원인 분석.

---

## 3) 새 채팅창 **퀵스타트** (10분 체크리스트)

1. **도메인 최신화**  
   ```powershell
   $PROJECT="quali-journal-prod"
   $REG_TYO="asia-northeast1"
   $LIVE="quali-admin-domap"

   gcloud run services update-traffic $LIVE --region $REG_TYO --to-latest
   $URL=(gcloud run services describe $LIVE --region $REG_TYO --format="value(status.url)")
   Start-Process "$URL/?v=$(Get-Random)"
   ```

2. **헬스·인증 점검**  
   ```powershell
   $TOKEN="32776f034dc84d4ba71613e76feec991"
   curl.exe -s -o NUL -w "%{http_code}`n" "$URL/health"             # 200 기대
   curl.exe -s -o NUL -w "%{http_code}`n" -H "Authorization: Bearer $TOKEN" "$URL/api/status"  # 200 기대
   ```

3. **UI(초딩 버전) 시험** — 주소창에 `?v=랜덤`을 붙여 열고, 페이지 상단 **ADMIN_TOKEN** 입력 → **[입력]**  
   - 배지 “인증됨(초록)” 표시 확인
   - KPI **새로고침/자동새로고침** 동작
   - **게이트 슬라이더** 조정 → KPI 반영
   - **키워드 수집→승인→발행** 버튼 실행, 하단 스트림 로그(`status/flow/report`) 순서 확인
   - **Export Markdown/CSV**가 **비어있지 않은지** 확인

4. **캐시·SW 검증**  
   ```powershell
   $BASE="https://admin.standardai.co.kr"
   curl.exe -sI "$BASE/" | findstr /R /I "^HTTP|^Cache-Control"
   curl.exe -sI "$BASE/service-worker.js?v=$(Get-Random)" -o NUL | findstr /R /I "^HTTP|^Cache-Control"
   ```

5. **API 단위 점검(선택)**  
   ```powershell
   $DATE=(Get-Date -Format "yyyy-MM-dd"); $KW="ipc-a-610"
   curl.exe -s "$URL/api/status?date=$DATE&keyword=$KW" -H "Authorization: Bearer $TOKEN"
   curl.exe -s -X PATCH "$URL/api/config/gate_required" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{"value":12}"
   curl.exe -s -X POST  "$URL/api/keyword/run?keyword=$KW" -H "Authorization: Bearer $TOKEN"
   ```

---

## 4) 운영 Runbook(필수)

### A. **운영 합격 기준** (UI/파이프라인)  
- 인증 배지 OK, `/api/status` 200  
- KPI 새로고침 및 슬라이더 반영  
- 키워드 파이프라인 로그가 **수집→재구성→승인→발행**으로 순차 도착, 결과 파일 생성/미리보기 가능  
- Export Markdown/CSV 내용 충분

### B. **배포/도메인 최신화 원칙**  
- 새 리비전 배포 후 **반드시** `--to-latest`로 최신 리비전에 100% 트래픽 전환  
- 사용자 단에서 `?v=랜덤`을 붙여 **강제 캐시 무효화** 확인

### C. **캐시 정책**  
- `/` 및 `/index.html` → `no-store, no-cache, must-revalidate`  
- 정적 해시 자산 → `immutable` + 1년

> 운영 기준/시험 절차/구현 가이드는 내부 문서의 정의와 동일합니다.

---

## 5) CI/CD & 보안 주의

- GitHub Actions: WIF(OIDC) 인증을 **우선**, 실패 시 SA Key **폴백** 단계가 있어야 함.  
- `admin/key.json` 등 **비밀키**는 커밋 금지(Secret scanning이 push 차단).  
- Artifact Registry 이미지 읽기 권한: 런타임 서비스계정에 `roles/artifactregistry.reader` 명시 부여.

---

## 6) 자주 보는 장애 유형 & 즉시 액션

| 증상 | 1차 확인 | 2차 조치 |
|---|---|---|
| update-traffic 실패(포트 리슨) | 서비스 로그, ENTRYPOINT/PORT | `--timeout 300`, uvicorn/gunicorn 인자 확인, 필요 시 롤백 |
| UI 전면 무반응 | 토큰 저장/전송(Authorization+X-Admin-Token) | 토큰 재저장, fetch 오버라이드/버튼 배선 스크립트 점검 |
| Export 빈 파일 | 파이프라인 스크립트 누락(orchestrator) | 컨테이너 빌드 컨텍스트 복사·재배포 |
| 캐시 갱신 안 됨 | `to-latest`, `?v=랜덤` | SW 파일/헤더 정책 재확인 |

---

## 7) 개발 참고(요지)

- **fetch 재정의/토큰 부착**: 모든 `/api/` 요청에 `Authorization`과 `X-Admin-Token` 동시 부착.  
- **버튼 배선 폴백**: `DOMContentLoaded` 시점에 QA 버튼 클릭 핸들러 연결 보강.  
- **빌드 컨텍스트**: `orchestrator.py`, `tools/`, `feeds/`, `data/`가 이미지에 포함되도록 `.dockerignore/.gcloudignore` 예외 추가.  
- **FastAPI 헤더**: 루트/HTML **no-store**, 해시 자산 **immutable**, `/service-worker.js`는 **no-store**.

---

## 8) 다음 액션(우선순위)

1. 최신 리비전에서 포트 리슨/초기화 시간 이슈 재현 → 로그 캡처 및 `--timeout 300` 재설정.  
2. 빌드 컨텍스트 점검: `admin/` 이미지에 파이프라인 및 SW 라우트가 포함되는지 확인.  
3. 도메인 최신화 후 UI 시험 전체 시나리오 재검증(섹션 3).

---

## 9) 새 채팅창에서 **컨텍스트 복구** 방법

- 이 파일(**Handover.md**)을 새 채팅창에 **업로드**하고 “**핵심 요약만 실행**”이라고 지시하면,  
  바로 **퀵스타트(섹션 3)**부터 재개 가능합니다.
- 추가로, 운영 기준·시험 절차·구현 가이드를 함께 업로드하면 더 정확하게 자동으로 연결됩니다.

---

### 부록: 커맨드 치트시트

```powershell
# 최신 리비전 100%
gcloud run services update-traffic quali-admin-domap --region asia-northeast1 --to-latest

# URL/헬스
$URL=(gcloud run services describe quali-admin-domap --region asia-northeast1 --format="value(status.url)")
curl.exe -s -o NUL -w "%{http_code}`n" "$URL/health"

# 타임아웃/부스트(예시)
gcloud run services update quali-admin-domap --region asia-northeast1 --timeout 300 --cpu-boost

# 캐시 확인
$BASE="https://admin.standardai.co.kr"
curl.exe -sI "$BASE/" | findstr /R /I "^HTTP|^Cache-Control"
curl.exe -sI "$BASE/service-worker.js?v=$(Get-Random)" -o NUL | findstr /R /I "^HTTP|^Cache-Control"
```

# 퀄리저널 Admin — 배포/커밋 런북 (재현 가능 버전)
**버전**: 2025-11-09 12:18  
**대상**: QualiJournal *Admin* (Cloud Run, Service Worker, 캐시 정책, WIF 기반 CI/CD)

---

## 0) 항목별 한 줄 요약
- **코드 변경** → 항상 **기능 브랜치**에서 커밋 → **Draft PR** → 머지. (main에 직접 푸시 금지)
- **배포** → GitHub Actions 워크플로 `deploy-admin-domap.yml` 실행 → 완료 후 **to‑latest**로 트래픽 전환.
- **검증** → 운영 도메인에서 *3줄 점검*으로 Cache‑Control과 SW 본문 확인.
- **롤백** → Cloud Run **이전 리비전으로 트래픽 이동**.

---

## 1) 사전 준비
- git, gh(GitHub CLI), gcloud(Cloud SDK) 설치 및 로그인
  ```powershell
  gh auth login --web --scopes repo,workflow
  gcloud auth login
  gcloud config set project quali-journal-prod
  ```
- 레포 루트( `.git` 이 있는 폴더 )에서 작업

**환경 값(기본값)**  
- `PROJECT=quali-journal-prod`  
- `REGION=asia-northeast1`  
- `SERVICE=quali-admin-domap`  
- `DOMAIN=https://admin.standardai.co.kr`  
- 워크플로 파일: `.github/workflows/deploy-admin-domap.yml`

---

## 2) 커밋/푸시 — 표준 흐름
> 스크립트: `fix-commit-push-v2.ps1` (레포 루트에 둠)

```powershell
# 세션 실행권한 (새 콘솔이면 1회)
Set-ExecutionPolicy -Scope Process Bypass -Force

# 커밋/푸시 + Draft PR 자동 생성(메시지만 바꿔서 사용)
.ix-commit-push-v2.ps1 -Commit "fix: SW route+headers; include admin static"
```
**무엇을 함?**
- origin 확인, rebase‑pull, 보호브랜치 감지 시 `fix/wif‑YYYYMMDD‑HHmmss` 브랜치 생성
- 변경 **add/commit** → **push -u** → **Draft PR** 자동 생성

> 에러/알림
- `index.lock` 경고 →  
  `taskkill /F /IM git.exe 2>$null; Remove-Item -Force .git\index.lock`
- 403/권한 → `gh auth login --web --scopes repo,workflow`

---

## 3) WIF 시크릿 보정 + 배포 워크플로 실행
> 스크립트: `setup-wif-and-run-v2.ps1`

```powershell
.\setup-wif-and-run-v2.ps1 `
  -Project "quali-journal-prod" `
  -Pool "github-wif" -Provider "github-oidc" `
  -Repo "cdmacs1003-cyber/quali-journal" `
  -SA "github-deploy@quali-journal-prod.iam.gserviceaccount.com" `
  -WorkflowPath ".github/workflows/deploy-admin-domap.yml"
```
**무엇을 함?**
- WIF Provider **정답 경로** 조회/생성 → `GCP_WORKLOAD_IDENTITY_PROVIDER` 시크릿 저장
- `GCP_PROJECT`, `GCP_SERVICE_ACCOUNT` 시크릿 저장
- **workflow_dispatch** 트리거 → 최근 run **watch**

---

## 4) Cloud Run 트래픽 최신 리비전으로 전환
워크플로가 성공하면 아래 1줄로 안전 확인:
```powershell
gcloud run services update-traffic quali-admin-domap --to-latest --region asia-northeast1 --project quali-journal-prod
```

(도메인 매핑 서비스 이름 확인용)
```powershell
gcloud beta run domain-mappings describe --domain admin.standardai.co.kr --region asia-northeast1 --format "value(spec.routeName)"
```

---

## 5) 운영 검증 — 3줄 점검
```powershell
$BASE="https://admin.standardai.co.kr"
curl.exe -sI "$BASE/"                                  | findstr /R "^Cache-Control"
curl.exe -sI "$BASE/service-worker.js?v=$(Get-Random)" | findstr /R "^HTTP\|^Cache-Control"
curl.exe -s  "$BASE/service-worker.js?v=$(Get-Random)" -o sw.js; `
  Select-String -Path .\sw.js -Pattern "BUILD|skipWaiting|clients\.claim"
```
**기대값**
- `/` & `/service-worker.js` → `Cache-Control: no-store, no-cache, must-revalidate`
- `sw.js`에 `BUILD` · `skipWaiting` · `clients.claim` 문자열 확인

> 브라우저 캐시 잔상 시(화면이 안 바뀌면)
```js
navigator.serviceWorker.getRegistrations()
 .then(rs => Promise.all(rs.map(r => r.unregister())))
 .then(() => location.reload());
```

---

## 6) 코드 구성 핵심(이미 반영됨)

- **서비스워커 라우트 단일화**
  - 경로: `/service-worker.js`
  - 헤더: `no-store, no-cache, must-revalidate`, `Service-Worker-Allowed: "/"`
  - 빌드ID 치환: `COMMIT_SHA` → 없으면 `BUILD_ID` → 없으면 `K_REVISION` → `dev`

- **캐시 정책**
  - `/`, `/index.html`, `/service-worker.js` → **no-store**
  - `/assets/<hash>.(js|css|img|font)` → **immutable + 1년**

- **정적 포함 보장**
  - `.gcloudignore`/`.dockerignore` 예외:
    ```
    !admin/index.html
    !admin/service-worker.js
    !admin/**.html
    !admin/**.js
    !admin/dist/**
    ```

- **워크플로 배포 명령(옵션 추가)**
  - `gcloud run deploy`에
    `--set-env-vars BUILD_ID=${ github.sha },COMMIT_SHA=${ github.sha }`

---

## 7) 롤백(필요 시)
최근 리비전 목록 확인:
```powershell
gcloud run revisions list --service quali-admin-domap --region asia-northeast1 --project quali-journal-prod
```
원하는 리비전에 100% 트래픽 전환:
```powershell
gcloud run services update-traffic quali-admin-domap --region asia-northeast1 --project quali-journal-prod --to-revisions <REVISION>=100
```

---

## 8) 자주 묻는 장애 대응 요약
- **WIF invalid_target / audience** → 3단계 스크립트로 시크릿 재설정
- **main 보호 브랜치 푸시 거절** → `fix-commit-push-v2.ps1` 로 기능 브랜치/PR 흐름 유지
- **서비스워커 404** → `/api/debug/sw-path` 로 선택 경로 확인 → 해당 경로에 `service-worker.js` 존재 보장
- **헤더가 no-store로 안 나옴** → 미들웨어 분기(`/`, `/index.html`, `/service-worker.js`) 확인

---

## 9) 원클릭(선택) — 빠른 실행 래퍼
레포 루트에 `quali-admin-deploy-quick.ps1` 를 만들어 아래처럼 실행:
```powershell
.\quali-admin-deploy-quick.ps1 -Commit "fix: ..." 
```

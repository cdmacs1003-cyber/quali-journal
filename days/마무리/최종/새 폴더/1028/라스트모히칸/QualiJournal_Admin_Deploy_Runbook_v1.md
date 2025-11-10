# QualiJournal Admin Deploy Runbook (Reproducible)
**Version:** v1  
**Generated:** 2025-11-10 01:01 UTC

본 문서는 *퀄리저널 Admin* 배포·검증 과정을 **항상 같은 결과로 재현**할 수 있게 만든 표준 런북입니다.  
(Windows PowerShell 7 기준 예시)

---

## 🔌 빠른 사용(요약)
아래 6줄만 순서대로 실행하면 **성공 판정 → 최신 리비전 전환**까지 끝납니다.

```powershell
$BASE="https://admin.standardai.co.kr"
curl.exe -sD - "$BASE/" -o NUL | findstr /R /I "^HTTP" "^Cache-Control"
curl.exe -sD - "$BASE/service-worker.js" -o NUL | findstr /R /I "^HTTP" "^Cache-Control"
curl.exe -s -H "Accept-Encoding: identity" "$BASE/service-worker.js" -o sw.js
Select-String -Path .\sw.js -Pattern "skipWaiting|clients\.claim" -CaseSensitive:$false
gcloud run services update-traffic quali-admin-domap --to-latest --region asia-northeast1 --project quali-journal-prod
Remove-Item .\sw.js -Force
```

**정상 판정:**  
- 루트와 서비스워커의 **HTTP 200** + **Cache-Control** 확인됨  
- `sw.js` 본문에서 `skipWaiting` 과 `clients.claim` 매치됨  
- (선택) 트래픽 **100% LATEST**로 전환 완료

---

## 1) 준비물
- PowerShell 7, `git`, `gh`(GitHub CLI), `gcloud`(Google Cloud CLI), `curl`
- GCP Project: `quali-journal-prod` / Region: `asia-northeast1` / Cloud Run Service: `quali-admin-domap`
- 프로덕트 도메인: `https://admin.standardai.co.kr`

환경 변수(선택):
```powershell
$BASE   = "https://admin.standardai.co.kr"
$REGION = "asia-northeast1"
$PROJ   = "quali-journal-prod"
$SERVICE= "quali-admin-domap"
```

> 🔐 **WIF/인증**은 CI/CD 워크플로에 이미 구성되어 있다고 가정합니다. 필요한 경우 기존 스크립트(`setup-wif-and-run-v2.ps1`)나 워크플로를 사용해 사전 점검을 수행하세요.

---

## 2) 배포 트리거(선택)
CI/CD로 배포를 트리거하는 대표 옵션 2가지입니다.

### A. 준비된 스크립트로 트리거
```powershell
.\setup-wif-and-run-v2.ps1 `
  -Project "quali-journal-prod" `
  -Pool "github-oidc" `
  -Repo "cdmacs1003-cyber/quali-journal" `
  -WorkflowPath ".github/workflows/deploy-admin-domap.yml"
# 성공 상태를 확인했다면 3단계로 이동
```

### B. GitHub CLI로 직접 트리거
```powershell
gh workflow run .github/workflows/deploy-admin-domap.yml -r main
gh run watch
```

> ✅ 워크플로가 초록색 **success**로 끝났다면 다음 단계로 진행합니다.

---

## 3) 배포 결과 검증(캐시 & 서비스워커)

### 3-1. 루트(HTML) 헤더 확인
```powershell
curl.exe -sD - "$BASE/" -o NUL | findstr /R /I "^HTTP" "^Cache-Control"
```
- 기대: `HTTP/2 200` (또는 `HTTP/1.1 200`) 과 `Cache-Control` 정책이 함께 보임

### 3-2. 서비스워커 헤더 확인(※ 쿼리스트링 **없이**)
```powershell
curl.exe -sD - "$BASE/service-worker.js" -o NUL | findstr /R /I "^HTTP" "^Cache-Control"
```
- 기대: `200` + `Cache-Control`

### 3-3. 서비스워커 본문 토큰 확인
압축 해제 효과를 위해 **identity**로 받아서 체크합니다.
```powershell
curl.exe -s -H "Accept-Encoding: identity" "$BASE/service-worker.js" -o sw.js
Select-String -Path .\sw.js -Pattern "skipWaiting|clients\.claim" -CaseSensitive:$false
```
- 기대: 두 패턴이 각각 한 줄 이상 매치됨  
- 정리: `Remove-Item .\sw.js -Force`

> ⚠️ **주의/FAQ**  
> - 일부 WAF/CDN은 `HEAD` 요청을 **403**으로 차단합니다. 헤더 확인은 `curl -sD -`(GET 헤더)로 진행하세요.  
> - `service-worker.js?v=...` 형태로 **쿼리스트링이 붙으면** 일부 환경에서 **404**(짧은 JSON)로 응답될 수 있습니다. **쿼리스트링 없이** 점검하세요.  
> - 본문이 gzip/br 인코딩이면 매치가 안 될 수 있으니, 반드시 `Accept-Encoding: identity`로 받으세요.

---

## 4) 트래픽을 최신 리비전으로 전환
```powershell
gcloud run services update-traffic $SERVICE --to-latest --region $REGION --project $PROJ
```
- 기대: `Traffic: 100% LATEST (currently quali-admin-domap-xxxxx-yyy)`

(참고) 도메인 매핑이 가리키는 **routeName** 확인:
```powershell
gcloud beta run domain-mappings describe --domain admin.standardai.co.kr `
  --region $REGION --project $PROJ --format "value(spec.routeName)"
```

---

## 5) 성공 판정 체크리스트
- [ ] 루트와 서비스워커 **HTTP 200 OK**  
- [ ] 두 응답 모두 **Cache-Control** 정책 확인  
- [ ] `sw.js` 본문에서 **skipWaiting / clients.claim** 패턴 확인  
- [ ] (필수) 트래픽 **100% LATEST**로 전환 완료

---

## 6) 태깅(선택, 재현 포인트 고정)
```powershell
git tag -a runbook-v1 -m "Admin deploy runbook v1"
git push origin runbook-v1
```

---

## 7) 롤백 가이드(요약)
1) 최근 리비전 목록 확인
```powershell
gcloud run revisions list --service $SERVICE --region $REGION --project $PROJ
```
2) 특정 리비전에 100% 전환(예: `quali-admin-domap-000130-abc`)
```powershell
gcloud run services update-traffic $SERVICE `
  --to-revisions=quali-admin-domap-000130-abc=100 `
  --region $REGION --project $PROJ
```

---

## 8) 자주 겪는 이슈 & 해결
- **HEAD 403**: 프록시/WAF 차단 → `curl -sD -` (GET 헤더)로 확인  
- **SW 404(JSON)**: `?v=`가 붙은 요청 → **쿼리 제거** 후 재시도  
- **본문 매치 불가**: 압축 인코딩 때문 → `Accept-Encoding: identity` 사용  
- **Git lock 에러**: `taskkill /F /IM git.exe` 후 `.git\index.lock` 삭제

---

## 부록 A) 최종 점검 단락(복붙용)
아래 블록을 내부 문서 어디든 붙여 넣으면, 1분 내에 필수 점검을 끝낼 수 있습니다.

```markdown
<!-- RUNBOOK: FINAL-CHECKS v1 -->
### 운영 최종 점검(항상 보이는 버전)

```powershell
$BASE="https://admin.standardai.co.kr"

# 1) HTML(루트) 헤더: HTTP 상태 + Cache-Control
curl.exe -sD - "$BASE/" -o NUL | findstr /R /I "^HTTP" "^Cache-Control"

# 2) 서비스워커 헤더: 쿼리스트링 없이(HEAD가 막힌 환경이 있어 GET 헤더로 점검 권장)
curl.exe -sD - "$BASE/service-worker.js" -o NUL | findstr /R /I "^HTTP" "^Cache-Control"

# 3) 서비스워커 본문 토큰 존재 확인(압축 해제를 위해 identity 사용)
curl.exe -s -H "Accept-Encoding: identity" "$BASE/service-worker.js" -o sw.js
Select-String -Path .\sw.js -Pattern "skipWaiting|clients\.claim" -CaseSensitive:$false
```

**정상 판정**
- (1)(2) 모두 `200` 과 `Cache-Control`이 보인다.
- (3)에서 `skipWaiting` 과 `clients.claim` 라인이 보인다.

> **주의/FAQ**
> - 일부 WAF/CDN은 **HEAD** 요청을 403으로 차단하거나, `?v=` 파라미터가 붙은 SW에 404를 줄 수 있습니다.  
>   → 헤더 점검은 쿼리스트링 없이 `curl -sD -`로 확인하고, 본문 점검은 `Accept-Encoding: identity`로 받은 후 **두 토큰**만 확인하세요.
> - 점검 후 `Remove-Item .\sw.js` 로 임시 파일을 정리하세요.
<!-- /RUNBOOK: FINAL-CHECKS v1 -->
```

---

**끝. 항상 같은 절차로, 같은 결과를 얻을 수 있도록 설계된 표준 런북입니다.**

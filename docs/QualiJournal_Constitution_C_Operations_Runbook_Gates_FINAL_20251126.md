# QualiJournal 운영·런북·게이트 헌법 (SSOT)

### 1. 개요(목적/적용 범위)

이 헌법은 **운영자·SRE·플랫폼 엔지니어**가

- 오늘 배포해도 되는지,  
- Traffic 50/50 전환을 어떻게 하는지,  
- `/api/standards/reviews` / Quick Tools / Cloud SQL Stop / SSOT Check 를  
  어떤 순서로 실행·검증해야 하는지

를 한 문서로 파악할 수 있도록 정리한 “운영 실행 헌법”입니다.

- 대상 환경: Cloud Run domap(prod) + Cloud SQL `quali-pg` + GitHub Actions

---

### 2. 본문

#### 2.1 게이트 스택(레벨별 Gate 개요)

운영 관점에서 게이트를 **L0 ~ L5** 계층으로 이해하면 편합니다.

- **L0 – 인프라 상태 게이트**
  - Cloud SQL Stop Policy / Cloud Run 서비스 상태 / 도메인 Ready 상태
- **L1 – Ready/Status 게이트**
  - `/health`, `/api/status`, `/api/ready`
- **L2 – Release DoD 스모크**
  - `/api/report`, `/api/standards/score-test`
- **L3 – 표준 리뷰 A1 게이트**
  - `/api/standards/reviews` 1회 + 10회 스모크 + 상태머신 pytest(`tests/test_standards_reviews_a1_state_machine.py`) + CI `ci-test-admin.yml` / `test-admin`

- **L4 – Quick Tools Gate**
  - `/api/tools/approve_top`, `/api/tools/repair` 정상 동작 여부
- **L5 – PR/CI SSOT 게이트**
  - Admin Tests / SSOT Check / domap 배포 파이프라인 구조

운영자는 “L0 → L1 → L2 → L3 → L4 → L5” 순으로 체크한다고 생각하면 됩니다.

##### 2.1.1 A1 운영 세션 워크플로우(0~6단계)

운영자는 QualiJournal Admin 관련 모든 세션(배포/장애/점검/테스트)을  
다음 **A1 운영 세션 워크플로우(0~6단계)** 기준으로 진행합니다.

0) **준비(Preparation)**  
   - 운영자 SSOT 카드(프로젝트/서비스/도메인/버킷/시크릿/스케줄러 7축)를 확인하고,  
   - PowerShell 공통 헤더(프로젝트/리전/서비스/도메인/버킷/토큰)를 세션 시작 시 항상 실행한다.

1) **Gate 스냅샷 (L0~L5 Before)**  
   - L0 인프라: Cloud Run SRC/LIVE, 도메인, Scheduler `daily-report`, Secret `ADMIN_TOKEN`, 버킷 존재 여부를 확인한다.  
   - L1 Ready: `/health`, `/api/status`, `/api/ready` 3종 스모크를 실행해 HTTP 코드와 상태를 기록한다.  
   - L2 Release: `/api/report` 및 표준/큐레이션 스모크(`/api/standards/score-test` 등)를 실행해 Release DoD를 확인한다.  
   - L3~L5는 필요 시(리뷰/도구/CI 이슈가 있을 때) A1 Runbook 해당 절을 참고해 추가 스모크를 수행한다.

2) **문제 정의 및 가설 정리**  
   - 이슈 정의 카드 A1에 “어떤 Gate(L0~L5)를 다루는 세션인지”와  
     현재 증상/로그/에러 코드를 요약해 적는다.  

3) **전략 A/B/C + 롤백 설계**  
   - 전략 A/B/C(예: 인프라 우선, 문서/SSOT 우선, 혼합 전략)를 정리하고,  
     실패 시 사용할 롤백 방법(이전 리비전/이전 설정 복원)을 함께 적어 둔다.

4) **실행 루프 (Git/배포/수정/롤백)**  
   - Git 브랜치·테스트 헌법(B)에 따라 테스트 브랜치를 만들고,  
     수정 후 CI/게이트 결과를 확인하면서 필요한 경우 Traffic50 Runbook에 따라 트래픽을 전환한다.  

5) **Gate 재검증 (L0~L5 After)**  
   - 동일한 스모크 세트를 다시 실행해 L0~L5 상태를 Before/After로 비교하고,  
     Gate 관점에서 PASS/HOLD/BAD 여부를 판단한다.

6) **인수인계 정리**  
   - 세션 종료 시, `QualiJournal_Admin_Handover_YYYYMMDD.md` 템플릿에  
     - Gate 상태 Before/After  
     - 실행한 명령/변경 사항  
     - 확정된 SSOT/정책 변경  
     - 다음 세션 TODO  
     를 정리해 남긴다.

---

##### 2.1.2 이슈 정의 카드 A1 템플릿(운영 세션용)

운영·배포·장애 대응 세션에서 사용하는 **이슈 정의 카드 A1** 템플릿은 다음과 같습니다.

1) **기본 정보**  
- 이슈 제목: (예) `QualiJournal Admin prod – domap /api/status 500`  
- 이슈 유형: 인프라 / 코드 / 표준 큐레이션 / CI·게이트 / 문서 중 선택 또는 복수 선택  
- 세션 목적·배경: 왜 지금 이 이슈를 다루는지 한 줄 요약  

2) **목표 / DoD**  
- 이 세션에서 “끝났다고 인정할 기준(정량/정성)”을 3~5줄로 명시  
- 예: `/health 200 / /api/status 200 / /api/report 200`, Traffic50 전환 완료, 문서/SSOT 정렬 등

3) **Gate 레벨(L0~L5)**  
- 이슈에 직접 관련된 Gate에 체크  
  - L0 인프라 / L1 Ready / L2 Release / L3 Reviews / L4 Quick Tools / L5 CI SSOT

4) **원인 후보(2~3개)**  
- 로그/증상을 바탕으로 가능한 원인 후보를 2~3개 나열  
  - 예: 도메인 인증서 지연, ADMIN_TOKEN 만료, GCS 버킷 권한 문제 등

5) **전략 A/B/C + 추천안 + 롤백**  
- 전략 A/B/C 요약(예: 인프라 우선, 코드 패치 우선, 문서/SSOT 개정 우선)  
- 추천 전략 1개 선택  
- 실패 시 롤백 플랜(OLD_REV 100%, 이전 설정 복원, SSOT 재되돌리기 등)을 함께 기술

---

#### 2.2 Cloud SQL Stop Policy – 운영 Runbook 관점

1) **일일 스모크(매일 아침 3줄)**

- 목적: “지금은 Cloud SQL OFF 상태가 기본 모드”가 잘 유지되는지 확인
- 실행:

```bash
gcloud sql instances list   --format="table(name,region,state,settings.activationPolicy)"

gcloud run services describe quali-admin-domap   --region asia-northeast1   --format="value(status.traffic)"

Invoke-WebRequest "https://admin.standardai.co.kr/?v=$(Get-Random)" `
  -Headers @{"Cache-Control"="no-cache"} `
  -OutFile "deployed_index.html"
```

- 기대 결과:
  - SQL: `STOPPED / NEVER`
  - Run: 하나의 리비전에 100% 트래픽
  - HTML: 정상 다운로드

2) **플래그 파일 관리**

- `infra/FLAGS/CLOUDSQL_STOPPED.yaml` 예시:
  - `instance: quali-pg`
  - `intent: stopped`
  - `since: 2025-11-03T12:00+09:00`
  - `until: 2025-12-31`
  - `owner: platform`
  - `reason: cost_hold`

3) **배포 전 가드레일 (CI 훅)**

```bash
STATE=$(gcloud sql instances describe quali-pg --format="value(state)")
if [ "$STATE" = "STOPPED" ]; then
  echo "::error ::Cloud SQL is STOPPED; deployment blocked."
  exit 1
fi
```

4) **정책·SSOT 간 유의점**

- Stop Policy 문서는 STOP/NEVER를 “기본 모드”로 규정.  
- SSOT_Check_Guide 4.2.1 에서는 domap 배포 설명 시 `state=RUNNABLE, activationPolicy=ALWAYS` 를 가정.  

→ **운영자 관점**에서는

- 배포/마이그레이션이 필요할 때만 `ALWAYS + RUNNABLE` 로 전환 → 작업 → 다시 STOP  
- CI 가드가 STOP 상태에서의 “실수 배포”를 막아준다고 이해하면 됩니다.  
- 장기 정책은 오너와 재합의 필요(`TODO`).

---

#### 2.3 domap Traffic 50/50 A1 Runbook

Traffic50 Runbook A1 기준 domap 트래픽 전환 절차:

1) **사전 준비**

- OLD_REV: 현재 안정 리비전 이름 (예: `quali-admin-domap-00173-vvq`)
- NEW_REV: 새 표준 리뷰 API 리비전 이름 (예: `quali-admin-domap-00174-jm8`)
- 현재 트래픽 분포 확인:

```powershell
gcloud run services describe $SERVICE `
  --project=$PROJECT `
  --region=$REGION `
  --format="yaml(status.traffic,status.latestReadyRevisionName,metadata.name)"
```

2) **트래픽 50/50 전환**

```powershell
$OLD_REV = "quali-admin-domap-00173-vvq"  # 실제 값으로 교체
$NEW_REV = "quali-admin-domap-00174-jm8"  # 실제 값으로 교체

gcloud run services update-traffic $SERVICE `
  --project=$PROJECT `
  --region=$REGION `
  --to-revisions $OLD_REV=50,$NEW_REV=50
```

3) **health/status/report 스모크**

```powershell
$BASE_URL = "https://admin.standardai.co.kr"  # domap 운영 도메인
$ADMIN_TOKEN = "<관리자 토큰>"

# /health
Invoke-WebRequest "$BASE_URL/health" -Method GET

# /api/status
Invoke-WebRequest "$BASE_URL/api/status" -Method GET -Headers @{
  "Authorization" = "Bearer $ADMIN_TOKEN"
  "X-Admin-Token" = $ADMIN_TOKEN
}

# /api/report
Invoke-WebRequest "$BASE_URL/api/report" -Method POST -Headers @{
  "X-Admin-Token" = $ADMIN_TOKEN
} -Body "" | Out-Null
```

4) **/api/standards/reviews 스모크 (1회 + 10회)**

- 단일 호출 1회 + 반복 호출 10회 스모크는  
  **Reviews A1 Runbook** 의 PowerShell 코드를 그대로 사용합니다.

5) **롤백/재전환 전략**

- NEW_REV 관련 에러 발생 시
  - `update-traffic` 명령으로 OLD_REV 100%로 롤백
  - 로그·스택트레이스를 수집한 뒤, 테스트 브랜치에서 재현/수정 (브랜치 헌법 B 참조)

---

#### 2.4 /api/standards/reviews A1 Runbook & CI 스모크

**QualiJournal_Admin_Reviews_Smoke_Runbook_A1_CI_20251124** 기준  
/`api/standards/reviews` 스모크는 다음 PowerShell 코드로 정의되며,  
이 코드가 **운영 Runbook + CI의 SSOT**입니다.

1) **공통 설정 헤더**

```powershell
# === 공통 설정: domap BASE_URL + 헤더 ===

# 1) domap 서비스 URL
$BASE_URL = "https://quali-admin-domap-xxxxxxxxxx.xx.run.app"

# 2) 관리자 토큰
$ADMIN_TOKEN = "<ADMIN_TOKEN_여기에_붙여넣기>"

# 3) 공통 헤더
$headers = @{
  "X-Admin-Token" = $ADMIN_TOKEN
}

# ※ 필요 시 ID 토큰 병행 사용 가능
# $ID_TOKEN = "<ID_TOKEN_여기에_붙여넣기>"
# $headers["Authorization"] = "Bearer $ID_TOKEN"

##### 2.4.x 표준 리뷰 테스트 카드 시드 & 상태머신 검증 Runbook(A안)

이 Runbook은 표준 리뷰 상태머신(HOLD → REVIEWED → PUBLISHED, 2인 검수)을
로컬 및 domap(prod) 환경에서 반복 검증하기 위한 절차다.
테스트 카드는 `standard_id = "TEST-STD-1"` 기준으로 생성한다.

###### (0) 공통 변수 설정

```powershell
# domap 기준 예시
$BASE   = "https://admin.standardai.co.kr"
$TOKEN  = "<ADMIN_TOKEN 실제 값>"          # Secret Manager ADMIN_TOKEN latest
$STD_ID = "TEST-STD-1"

(1) 테스트 리뷰 카드 시드 (reset = true)
$bodyObj = @{
  standard_id = $STD_ID
  reset       = $true
}

Invoke-RestMethod `
  -Method Post `
  -Uri "$BASE/api/standards/reviews/test/init" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json) |
  ConvertTo-Json -Depth 5


기대:

ok: true

name: "standards_reviews_test_init"

data.review_task.standard_id == "TEST-STD-1"

status == "HOLD", approved_by == [], required_reviewers == 2

data.created == true, data.reset == true

(2) HOLD 상태 확인
Invoke-RestMethod `
  -Method Get `
  -Uri "$BASE/api/standards/reviews?status=HOLD" `
  -Headers @{ "X-Admin-Token" = $TOKEN } |
  ConvertTo-Json -Depth 5


기대:

count >= 1

대상 카드의 standard_id == "TEST-STD-1", status == "HOLD", approved_by == []

(3) 1차 승인(r1) – 여전히 HOLD 유지
$bodyObj = @{ reviewer_id = "r1" }

Invoke-RestMethod `
  -Method Post `
  -Uri "$BASE/api/standards/reviews/$STD_ID/approve" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json)


기대:

응답 data.review_task.approved_by 에 "r1" 포함

status == "HOLD" (required_reviewers=2 이므로 아직 REVIEWED 아님)

(4) 2차 승인(r2) – REVIEWED 전환
$bodyObj = @{ reviewer_id = "r2" }

Invoke-RestMethod `
  -Method Post `
  -Uri "$BASE/api/standards/reviews/$STD_ID/approve" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json)

Invoke-RestMethod `
  -Method Get `
  -Uri "$BASE/api/standards/reviews?status=REVIEWED" `
  -Headers @{ "X-Admin-Token" = $TOKEN } |
  ConvertTo-Json -Depth 5


기대:

count >= 1

대상 카드의 status == "REVIEWED"

approved_by 에 "r1", "r2" 두 명 포함

required_reviewers == 2

(5) 발행(publish) – PUBLISHED + PASS 승격
Invoke-RestMethod `
  -Method Post `
  -Uri "$BASE/api/standards/reviews/$STD_ID/publish" `
  -Headers @{ "X-Admin-Token" = $TOKEN; "Content-Length" = "0" }

Invoke-RestMethod `
  -Method Get `
  -Uri "$BASE/api/standards/reviews?status=PUBLISHED" `
  -Headers @{ "X-Admin-Token" = $TOKEN } |
  ConvertTo-Json -Depth 5


기대:

count >= 1

status == "PUBLISHED"

approved_by 에 "r1", "r2"

item.decision == "PASS" (발행 시 PASS 승격 규칙)

(6) 에러 케이스 상수화(선택)

다음 케이스는 상태머신/에러코드가 SSOT와 맞는지 확인하기 위한 선택 테스트다.

존재하지 않는 ID에 대한 승인/발행

$bodyObj = @{ reviewer_id = "rX" }

Invoke-RestMethod `
  -Method Post `
  -Uri "$BASE/api/standards/reviews/NO-SUCH-ID/approve" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json)
# 기대: HTTP 404, body.detail == "review task not found"


HOLD 상태에서 곧바로 publish 시도

Invoke-RestMethod `
  -Method Post `
  -Uri "$BASE/api/standards/reviews/$STD_ID/publish" `
  -Headers @{ "X-Admin-Token" = $TOKEN; "Content-Length" = "0" }
# 기대: HTTP 409, body.detail == "review task not reviewed"


이 Runbook은 L3 표준 리뷰 A1 Gate 에서 상태머신 검증을 수행할 때 사용하며,
CI의 /api/standards/reviews 스모크(1회 + 10회 GET)와 함께 운영 Gate PASS 여부를 판단하는 기준으로 삼는다.

이 블록만 C 헌법에 넣으면,

- domap 기준 **상태머신 검증 절차**가 문서화되고,   
- 지금까지 PowerShell로 실행한 내용과 1:1로 매칭돼서  
  이후 인수인계/다른 운영자도 그대로 따라 할 수 있어.

###### 2.4.y pytest 및 CI 연계 규칙

- 표준 리뷰 A1 상태머신은 **pytest 케이스** `tests/test_standards_reviews_a1_state_machine.py` 로도 동일하게 검증한다.
- GitHub Actions 워크플로 **`ci-test-admin.yml`** 의 `test-admin` 잡 안에서  
  `Run standards reviews A1 state-machine test` 스텝이 이 pytest를 실행한다.
- 운영자는 다음 두 가지만 확인하면 L3 A1 Gate 상태머신이 SSOT와 일치하는지 빠르게 판단할 수 있다.
  - 로컬 점검: `pytest -q tests/test_standards_reviews_a1_state_machine.py -vv`
  - CI 점검: PR 의 `Admin Tests (pytest only) / test-admin` 체크가 초록(PASS) 상태인지

---

## 3. 추가로 건드릴 필요 없는 부분

- **헌법 B(브랜치/테스트)** 쪽은 이번 변경으로 새 규칙이 생긴 건 아니라서  
  그대로 두어도 충분하다.  
- `/api/status`, `/api/report`, Cloud SQL Stop Policy, Quick Tools Gate 등은  
  이미 A/C 헌법에 잘 정리돼 있어서 **추가 조항 없이 지금 SSOT랑 맞는다.**  
```

2) **단일 스모크 (1회 호출)**

```powershell
Write-Host "== Smoke: GET /api/standards/reviews (single) =="

$resp = Invoke-WebRequest `
  -Uri "$BASE_URL/api/standards/reviews" `
  -Headers $headers `
  -Method GET

if ($resp.StatusCode -ne 200) {
  Write-Host "❌ StatusCode != 200 : $($resp.StatusCode)"
  exit 1
}

$body = $resp.Content | ConvertFrom-Json

if (-not $body.ok) {
  Write-Host "❌ ok=false"
  exit 1
}

if ($null -eq $body.count -or $null -eq $body.items) {
  Write-Host "❌ count 또는 items 없음"
  exit 1
}

Write-Host "✅ 단일 스모크 PASS (count=$($body.count))"
```

3) **반복 스모크 (10회 호출)**

```powershell
Write-Host "== Smoke: GET /api/standards/reviews (loop x10) =="

$successCount = 0

for ($i = 1; $i -le 10; $i++) {
  Write-Host (" - call #{0}" -f $i)
  try {
    $resp = Invoke-WebRequest `
      -Uri "$BASE_URL/api/standards/reviews" `
      -Headers $headers `
      -Method GET

    if ($resp.StatusCode -eq 200) {
      $successCount++
    } else {
      Write-Host "❌ StatusCode != 200 : $($resp.StatusCode)"
      exit 1
    }
  } catch {
    Write-Host "❌ 예외 발생: $_"
    exit 1
  }
}

if ($successCount -ne 10) {
  Write-Host "❌ 10회 연속 200이 아님"
  exit 1
}

Write-Host "✅ 10회 반복 스모크 PASS"
```

4) **운영 ↔ CI 동기화**

- Admin DoD v1.1에서는 **이 PowerShell 블록이 코드 SSOT**이며,  
  Traffic50 Runbook과 `ci-deploy-prod.yml` 의  
  `Smoke tests (/api/standards/reviews)` 스텝은 이 코드를 그대로 미러링해야 함.

---

#### 2.5 Quick Tools Gate – 운영 관점 스모크

Traffic50 Runbook에는 Quick Tools B-mode 스모크가 포함됩니다. (요약)

- 핵심 아이디어:
  - `/api/tools/approve_top` 호출 후
    - **정상 케이스**: rc=0, ok=true
    - **B-mode 케이스**: rc=127, ok=true, sync_log.ok=true
  - `/api/tools/repair` 호출 후에도 같은 기준 적용
- 운영자는 A1 Runbook 체크리스트에
  - Quick Tools 호출 결과와 rc 값,
  - sync_log.ok, count 값 등을 기록해두어야 합니다.

※ 실제 PowerShell/cURL 스크립트는 Admin DoD 및 Traffic50 Runbook 원문을 참조해  
레포 내 `tools_quick_smoke.ps1` 등으로 유지하는 것을 권장합니다.

---

#### 2.6 A1 운영 세션 워크플로우 연계 규칙

이 Git/브랜치 헌법은 QualiJournal 레포의 브랜치·PR·테스트 규칙을 정의하고,  
운영·배포·장애 대응 세션에서 사용하는 **A1 운영 세션 워크플로우**와 함께 적용한다.

1) 세션 시작 시 브랜치 전략

- 운영자는 새로운 작업 세션을 시작하기 전에 **A1 이슈 카드**에  
  - 다루는 Gate(L0~L5)  
  - 예상 변경 범위(코드/설정/문서)  
  - 사용할 브랜치 유형(feature/fix/test/docs 등)을 먼저 적는다.
- 실제 작업 브랜치는 원칙적으로  
  - `main-signed-ssot` 최신 상태에서  
  - `feature/...`, `fix/...`, `feature/test-...`, `docs/...` 형태로 분기한다.

2) Git 실행 루프와 Gate 연동

- A1 워크플로우 4단계(실행 루프)에서 Git 작업은 다음 순서를 따른다.  
  1. `main-signed-ssot` 동기화 → 새 작업 브랜치 생성  
  2. 작은 단위의 변경 + 로컬 테스트/pytest  
  3. PR 생성 후 **필수 체크(테스트 + SSOT Check + Deploy)** 통과 여부 확인  
  4. 필요 시 수정 → 재실행 → 모든 Gate PASS 후 main에 머지
- PR 설명에는 이 이슈가  
  - 어떤 A1 이슈 카드 번호/제목과 연결되는지,  
  - 어떤 Gate(L0~L5)에 영향을 주는지 간단히 적는다.

3) 테스트 브랜치와 위험 동작

- `feature/test-*` 브랜치는  
  - A1 이슈 카드에서 “실험/재현/검증 목적”으로 명시된 경우에만 사용한다.  
  - 한 실험 = 한 브랜치 = 한 파일 = 한 목적 원칙을 따른다.
- 모든 브랜치에서 다음 동작은 금지하거나, **A1 카드에 사전 기록 후에만 예외적으로 사용**한다.  
  - `git reset --hard ...`, `git clean -fdx`  
  - 대량 포맷팅/자동 정리 툴로 인한 광범위 변경  
  - 다른 브랜치의 파일을 통째로 복사·덮어쓰기
- 이상 징후(브랜치 보호 체크 사라짐, 이해 안 되는 충돌 등)를 발견하면  
  - 추가 Git 명령을 중단하고,  
  - 상태 스냅샷(스크린샷/`git status`)을 남긴 뒤 A1 이슈 카드와 인수인계 문서에 기록한다.

4) 인수인계와 “정상 PR 1사이클” 재현

- A1 워크플로우 6단계(인수인계)에서 운영자는  
  - 사용한 브랜치 이름,  
  - 주요 커밋/PR 번호,  
  - 통과한 CI 체크 목록을 인수인계 md에 남긴다.
- Git/브랜치 헌법의 “정상 PR 1사이클” 예시는  
  - 새 작업 세션에서도 그대로 재현 가능한지  
  - A1 이슈 카드와 함께 주기적으로 점검한다.


#### 2.7 SSOT Check FAIL 시 운영 대응 Runbook

SSOT_Check_Guide 기준 FAIL 시 운영 절차 요약:

1. PR 화면에서 **SSOT Check / guard → Details** 클릭
2. 로그의 `::error` 메시지 확인
3. 오류 유형에 따른 대응
   - SSOT 문서 관련:
     - PR에 `ssot-change` 라벨 추가
     - 필요한 경우 SSOT 문서 본문을 업데이트하고, 변경 이유를 명시
   - `ci-deploy-prod.yml` 구조 관련:
     - Admin DoD 문서의 12장(파이프라인)과 비교하여  
       env 키, job 수, smoke step 이름, image 출처 등을 수정
4. 수정 후 다시 push → 워크플로 자동 재실행
5. 여전히 FAIL이면 SRE/오너와 함께 원인 재분석 후,  
   SSOT Check 규칙 자체를 업데이트할지 검토

운영자는 **배포 전 마지막 게이트**로 SSOT Check를 활용한다고 이해하면 됩니다.

---

#### 2.8 Admin Runbook FINAL & 마무리 가이드 연동

기존 Admin Runbook과 마무리 가이드는 다음 항목을 운영 루틴에 포함:

- 도메인 매핑 상태 점검 및 인증서 지연 시 재생성
- Workload Identity Federation(WIF) 설정 확인
- `/health`, `/api/status`, `/api/report` 스모크 및 주요 장애 패턴 대응
- 스케줄러(`/api/report` 일일 보고서 생성) 설정 및 실패 시 재시도

이 헌법 C에서는 **표준 리뷰/Quick Tools/Cloud SQL/SSOT Check** 중심으로 정리했고,  
기타 도메인·WIF·Export/Backup 관련 운영은 기존 Runbook을 그대로 따릅니다.

---

### 3. 체크리스트

운영자가 배포/트래픽 전환/스모크를 수행할 때 사용할 수 있는 체크리스트입니다.

| 항목 | 설명 | 예시 |
| --- | --- | --- |
| Cloud SQL Stop 일일 스모크 | 아침마다 3줄 스모크를 실행해 SQL/Cloud Run/HTML 상태를 기록했는지 | gcloud list 결과 `quali-pg STOPPED/NEVER`, domap 100%, HTML 파일 정상 다운로드 로그 캡처 |
| Cloud SQL 플래그 파일 | `infra/FLAGS/CLOUDSQL_STOPPED.yaml` 값이 현재 정책과 일치하는지 | intent=stopped, until=YYYY-MM-DD가 실제 운영 계획과 맞는지 |
| 배포 전 SQL 가드레일 | CI에서 STOP 상태일 때 배포가 차단되는지, 배포 전 SQL 상태를 RUNNABLE로 맞췄는지 | 배포 직전 `gcloud sql instances describe` 로 state 확인, STOP이면 정책에 따라 작업 연기 |
| Traffic 50/50 전환 기록 | OLD_REV/NEW_REV 이름과 전환 시각·담당자를 기록했는지 | Runbook 체크리스트에 `OLD_REV=..., NEW_REV=..., 50/50 전환 시각` 메모 |
| health/status/report 스모크 | 트래픽 전환 후 `/health`, `/api/status`, `/api/report` 3종 스모크를 실행했는지 | 3개 모두 200 코드, `/api/status` 의 지표가 정상 범위인지 |
| `/api/standards/reviews` A1 스모크 | 단일 호출 1회 + 10회 반복 스모크를 모두 실행했고, 200/ok/count/items 기준을 만족했는지 | PowerShell 스크립트 로그에 “단일 스모크 PASS, 10회 반복 스모크 PASS” 메시지가 찍혔는지 |
| Quick Tools Gate 스모크 | approve_top/repair 호출 결과 rc/ok/sync_log가 Gate 기준을 충족하는지 | B-mode 케이스에서 rc=127이면서 sync_log.ok=true 인지 |
| SSOT Check / guard 상태 | 배포 직전 PR에서 SSOT Check가 PASS인지, FAIL일 경우 원인을 해결했는지 | FAIL 로그를 기반으로 ssot-change 라벨 추가 또는 YAML 구조 수정 후 재실행한 기록 |
| Admin Runbook 항목 점검 | 도메인·WIF·스케줄러·백업 등 기존 Runbook의 Top 항목을 완료했는지 | `/health 405`, `/api/status 401`, 도메인 Ready 지연 등 주요 장애 처리 플레이북을 적용했는지 |

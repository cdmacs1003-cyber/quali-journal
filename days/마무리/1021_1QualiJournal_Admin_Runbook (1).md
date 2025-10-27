# 📦 QualiJournal Admin — 업무 인수인계 보고서 (PowerShell 전용 Runbook)

> **대상**: `cdmacs1003-cyber/quali-journal`  
> **브랜치**: 작업 브랜치 `fix-authui-signed2` (서명 커밋) → `main` 머지 대상  
> **도구**: Windows PowerShell 5.1(기본) + Git + GitHub CLI(gh)  
> **핵심 원칙**: “서명 커밋(Verified) + 필수 체크 이름 일치 + 통과 → 머지”

---

## 0) TL;DR (지금 상태 & 다음 할 일)

- ✅ **SSH 서명키** 생성/등록 완료 → Git 전역 서명 설정(`gpg.format=ssh`, `user.signingkey="공개키.pub"`, `commit.gpgsign=true`)
- ✅ **서명 커밋으로 재작성**: `fix-authui-signed2`에 `-S` 체리픽 완료 → 원격 푸시 OK
- ✅ **PR 생성**(예: `#39`) 및 **보호 규칙 Required checks**를 **실제 check‑runs 이름(성공 2개)**으로 갱신  
  - Required = `Deploy to Cloud Run` / `deploy-preview`  
  - 실패 체크 **`deploy`**는 Required에서 제거됨
- ⛳ **남은 액션(머지 차단 해소)**  
  - PR이 **main 최신 상태** 요구 시 → `rebase/merge`로 최신화 후 푸시  
  - **리뷰/대화 해소** 규칙이 있으면 승인/Resolve (또는 보호 규칙에서 해제)  
  - 체크가 전부 초록이면 **일반 머지**(`--auto`는 리포 정책상 비허용)

---

## 1) 환경/사전 조건

- Windows PowerShell 5.1(기본)  
- Git, GitHub CLI(gh) 설치  
- 레포 권한: 브랜치 보호 규칙 수정/병합 가능 권한  
- GitHub 계정 **Settings → SSH and GPG keys → New SSH key** 에 **Key type = Signing key** 로 공개키 등록됨

---

## 2) 표준 작업 흐름 (1회 설정 + 반복 운영)

### A. 서명용 SSH 키 생성 & Git 전역 서명 설정 (1회)

```powershell
# 경로
$KeyDir     = Join-Path $env:USERPROFILE ".ssh"
$KeyPrivate = Join-Path $KeyDir "id_ed25519_signing"
$KeyPublic  = "$KeyPrivate.pub"
New-Item -ItemType Directory -Force -Path $KeyDir | Out-Null

# ssh-agent
if ((Get-Service ssh-agent).Status -ne 'Running') {
  Set-Service -Name ssh-agent -StartupType Automatic
  Start-Service ssh-agent
}

# 키 생성(프롬프트 나오면 Enter 두 번 → 빈 패스프레이즈)
ssh-keygen -t ed25519 -f "$KeyPrivate" -C "signing-key"
ssh-add "$KeyPrivate"

# Git 전역 서명 설정
git config --global gpg.format ssh
git config --global user.signingkey "$KeyPublic"
git config --global commit.gpgsign true

# 공개키 복사 → GitHub에 Signing key로 등록
Get-Content "$KeyPublic" | Set-Clipboard
```

### B. 서명 커밋 브랜치 준비 (이미 완료됨)

```powershell
git fetch origin
git switch -C fix-authui-signed2 origin/main
git rev-list --reverse origin/main..origin/fix-authui-signed | % { git cherry-pick -S $_ }
git push -u origin fix-authui-signed2 --force-with-lease
```

### C. PR 생성 → Required 체크 자동 반영 → 체크 대기 → 머지

> PowerShell 5.1 호환 ( `||` 대신 `$LASTEXITCODE` 사용 )

```powershell
$owner="cdmacs1003-cyber"; $repo="quali-journal"; $head="fix-authui-signed2"

# gh 로그인
gh auth status *> $null; if ($LASTEXITCODE -ne 0) { gh auth login --hostname github.com --device }

# PR 번호 확보(없으면 생성)
$pr = gh pr list --repo "$owner/$repo" --head $head --json number --jq '.[0].number'
if (-not $pr) {
  gh pr create --repo "$owner/$repo" --base main --head $head `
    --title "fix(auth-ui): signed commits & KPI refresh" `
    --body  "서명 커밋으로 재작성 및 KPI/리프레시 안정화"
  $pr = gh pr list --repo "$owner/$repo" --head $head --json number --jq '.[0].number'
}

# SHA & check-runs 가져오기 (필요 시 5초 간격 폴링)
$sha = gh pr view $pr --repo "$owner/$repo" --json headRefOid --jq .headRefOid

function Get-CheckRuns {
  $raw = gh api "repos/$owner/$repo/commits/$sha/check-runs" 2>$null
  if (-not $raw) { return @() }
  ($raw | ConvertFrom-Json).check_runs | ForEach-Object {
    [PSCustomObject]@{ name=$_.name; status=$_.status; conclusion=$_.conclusion; app_id=$_.app.id }
  }
}
$cr=@(); for($i=0;$i -lt 30;$i++){ $cr=Get-CheckRuns; if($cr){break}; Start-Sleep 5 }

# 성공 체크만 Required에 반영(실패/스킵 제거)
$success = $cr | Where-Object { $_.status -eq 'completed' -and $_.conclusion -eq 'success' } |
           Sort-Object name -Unique
$checks=@(); foreach($c in $success){ $checks += @{ context=$c.name; app_id=$c.app_id } }

$payload=@{
  required_status_checks=@{ strict=$true; checks=$checks }
  enforce_admins=$false; required_pull_request_reviews=$null; restrictions=$null
} | ConvertTo-Json -Depth 5

$payload | gh api -X PUT "repos/$owner/$repo/branches/main/protection" `
  -H "Accept: application/vnd.github+json" --input -

# 체크 대기(성공/실패 판단)
function Wait-Checks {
  while($true){
    $runs=Get-CheckRuns
    $pending = $runs | Where-Object { $_.status -ne 'completed' }
    $failed  = $runs | Where-Object { $_.status -eq 'completed' -and $_.conclusion -in @('failure','cancelled','timed_out','action_required') }
    if($failed){ $failed | ft name,conclusion -AutoSize; return $false }
    if(-not $pending -and ($runs | ?{$_.conclusion -eq 'success'}).Count -ge $runs.Count){ return $true }
    Start-Sleep 5
  }
}
$ok = Wait-Checks
if(-not $ok){ Write-Host "체크 실패/미완료. 위 표 참조."; return }

# up-to-date 정책 충족(필요 시 최신화) → 머지
git fetch origin
git switch $head
git rebase origin/main 2>$null; if($LASTEXITCODE -ne 0){ git rebase --abort; git merge origin/main }
git push --force-with-lease

gh pr merge $pr --repo "$owner/$repo" --merge --delete-branch
```

---

## 3) 스모크 테스트 & 캐시 무시 새로고침

> PowerShell의 `curl`은 `Invoke-WebRequest`의 별칭이라 **`-s`** 같은 리눅스 옵션이 동작하지 않습니다.

```powershell
# 헬스
Invoke-WebRequest -Uri "https://admin.standardai.co.kr/health" -UseBasicParsing

# 상태 (토큰 필요 시)
$TOKEN="32776f034dc84d4ba71613e76feec991"
Invoke-WebRequest -Uri "https://admin.standardai.co.kr/api/status?date=2025-10-21&keyword=IPC-A-610" `
  -Headers @{ Authorization = "Bearer $TOKEN" } -UseBasicParsing
```

브라우저: F12 → **Network ▸ Disable cache** 체크 →  
`https://admin.standardai.co.kr/?v=20251021` → **Ctrl+F5**

---

## 4) 실패/에러 대응 가이드 (현장 체크리스트)

- **contexts: []** → check-runs가 아직 생성 전. 5~10초 폴링 후 재시도.  
- **422 “invalid payload checks/contexts”** → 구형 `contexts` 대신 **신규 `checks=[{context,app_id}]`** 사용.  
- **401(REST 호출)** → `$env:GITHUB_TOKEN` 누락. `gh api`로 대체(gh 로그인 토큰 재사용).  
- **merge “base branch policy prohibits the merge”**  
  1) **Up‑to‑date(strict)** 미충족 → `rebase/merge`로 최신화 후 푸시  
  2) **Required reviews / conversation resolution** 정책 활성 → 승인/Resolve or 보호 규칙 조정  
- **Windows PowerShell에서 curl 에러** → `Invoke-WebRequest` 사용 또는 `curl.exe` 실행.  
- **check-runs 중 `deploy`만 failure** → Required에서 제외(성공 체크만 Required 유지) 후 머지.  
- **PR이 draft 또는 리뷰 필요** →  
  `gh pr view $pr --repo "$owner/$repo" --json mergeStateStatus,reviewDecision,isDraft --jq .` 로 상태 확인, 규칙 처리.

---

## 5) 운영 항목(권장)

- 보호 규칙(브랜치)  
  - **Require signed commits**: ON  
  - **Require status checks**: `Deploy to Cloud Run`, `deploy-preview`  
  - **Strict**: ON (필요 시 off)  
  - (조직 정책에 따라) 리뷰/대화 해소 규칙 최소화 또는 운영 절차 반영
- PR 워크플로  
  1. `fix-…-signed*` 브랜치에서 작업  
  2. PR 생성 → check‑runs 이름 자동 반영 → 체크 감시  
  3. up‑to‑date 충족 → 머지
- 배포 후 스모크 + 캐시 무시 새로고침 루틴 상시 적용

---

## 6) 다음 액션(담당자 체크박스)

- [ ] (필수) **PR 체크 “전부 초록” 확인** — 실패 체크 있으면 제거/수정  
- [ ] (필요) **리뷰/대화** 규칙 해소 또는 보호 규칙 조정  
- [ ] **머지 & 브랜치 삭제**  
- [ ] **/health, /api/status** 스모크 통과  
- [ ] **UI 캐시 무시 새로고침**(Ctrl+F5)로 동작 확인

---

## 7) 부록 — 단축 명령 모음

```powershell
# PR 번호
gh pr list --repo cdmacs1003-cyber/quali-journal --head fix-authui-signed2 --json number --jq '.[0].number'

# 마지막 SHA
gh pr view <PR> --repo cdmacs1003-cyber/quali-journal --json headRefOid --jq .headRefOid

# check-runs 요약
gh api repos/cdmacs1003-cyber/quali-journal/commits/<SHA>/check-runs `
  --jq '.check_runs[] | {name,status,conclusion,app_id:.app.id}'

# 보호 규칙 조회/갱신(신규 checks 규격)
gh api repos/cdmacs1003-cyber/quali-journal/branches/main/protection
# PUT 본문 예시: { required_status_checks: { strict:true, checks:[{context,app_id}...] } ... }
```

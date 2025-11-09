param(
  [string]$RepoURL = "https://github.com/cdmacs1003-cyber/quali-journal.git",
  [string]$Branch  = "",
  [string]$Commit  = "fix: admin static & WIF"
)
function Fail($m){Write-Host $m -ForegroundColor Red; exit 1}
# 레포 확인
& git rev-parse --is-inside-work-tree *> $null; if ($LASTEXITCODE -ne 0){ Fail "여기는 깃 레포가 아닙니다. 레포 루트에서 실행하세요." }
# gh 인증
& gh --version *> $null; if ($LASTEXITCODE -ne 0){ Fail "GitHub CLI(gh) 필요" }
& gh auth status *> $null; if ($LASTEXITCODE -ne 0){ Fail "먼저: gh auth login --web --scopes repo,workflow" }
& gh auth setup-git *> $null
# origin 보장
$remotes = (& git remote) -join "`n"; if ($remotes -notmatch "(?m)^origin$"){ & git remote add origin $RepoURL }
# fetch
& git fetch origin --prune
# 브랜치 결정
$current = (& git rev-parse --abbrev-ref HEAD).Trim()
if ([string]::IsNullOrWhiteSpace($Branch)){
  if ($current -eq "main" -or $current -eq "master"){ $Branch = "fix/wif-$([DateTime]::Now.ToString('yyyyMMdd-HHmmss'))"; & git checkout -b $Branch }
  else { $Branch = $current }
} else {
  $exists = (& git branch --list $Branch); if ([string]::IsNullOrWhiteSpace($exists)){ & git checkout -b $Branch } else { & git checkout $Branch }
}
# 변경 있으면 커밋
$st = & git status --porcelain; if ($st){ & git add -A; & git commit -m $Commit }
# 업스트림 있으면 리베이스
& git rev-parse --abbrev-ref --symbolic-full-name "@{u}" *> $null
if ($LASTEXITCODE -eq 0){ & git pull --rebase --autostash; if ($LASTEXITCODE -ne 0){ Fail "리베이스 실패. 충돌 해결 후 재실행." } }
# 푸시
& git push -u origin $Branch; if ($LASTEXITCODE -ne 0){ Fail "푸시 실패. 권한/보호 브랜치 확인." }
# PR 드래프트 생성(이미 있으면 경고만)
& gh pr create -B main -H $Branch -t $Commit -b "자동 생성 PR" -d *> $null
if ($LASTEXITCODE -eq 0){ Write-Host "Draft PR 생성 완료." -ForegroundColor Green } else { Write-Host "PR 생성 생략(이미 열려있을 수 있음)" -ForegroundColor Yellow }

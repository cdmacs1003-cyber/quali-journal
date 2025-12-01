# tools/smoke_stdreviews_L3B_clean.ps1
# L3 B-canary 전용, 가장 단순한 리비전 스모크 버전
# - /api/standards/reviews/test/init 호출
# - x-cloud-run-revision 헤더를 읽어서 ExpectedRevision 과 비교

param(
    [Parameter(Mandatory = $true)]
    [string] $BaseUrl,

    [Parameter(Mandatory = $true)]
    [string] $ExpectedRevision,

    [Parameter(Mandatory = $true)]
    [string] $AdminToken
)

$ErrorActionPreference = "Stop"

Write-Host "=== L3 B-canary clean smoke ==="
Write-Host "BaseUrl          = $BaseUrl"
Write-Host "ExpectedRevision = $ExpectedRevision"

if (-not $BaseUrl) {
    Write-Error "BaseUrl 이 비어 있습니다."
    exit 1
}

if ($BaseUrl.EndsWith("/")) {
    $BaseUrl = $BaseUrl.TrimEnd("/")
}

if (-not $AdminToken) {
    Write-Error "AdminToken 이 비어 있습니다."
    exit 1
}

$headers = @{
    "X-Admin-Token" = $AdminToken
}

# 테스트용 카드 하나를 초기화
$bodyObj = @{
    standard_id = "TEST-STD-1"
    reset       = $true
}

try {
    $url = "$BaseUrl/api/standards/reviews/test/init"
    Write-Host "POST $url"

    $response = Invoke-WebRequest `
        -Method Post `
        -Uri $url `
        -Headers $headers `
        -ContentType "application/json" `
        -Body ($bodyObj | ConvertTo-Json -Depth 5) `
        -ErrorAction Stop
}
catch {
    Write-Error "test/init 호출 실패: $($_.Exception.Message)"
    exit 1
}

Write-Host "Response status code = $($response.StatusCode)"
Write-Host "Response headers:"
foreach ($h in $response.Headers.GetEnumerator()) {
    Write-Host ("  {0}: {1}" -f $h.Name, ($h.Value -join ','))
}

# Cloud Run 기본 리비전 헤더(표준 이름) 사용
$revHeader = $response.Headers["x-cloud-run-revision"]

if (-not $revHeader) {
    Write-Error "x-cloud-run-revision 헤더가 응답에 없습니다. (Cloud Run 리비전 정보를 받지 못했습니다)"
    exit 1
}

Write-Host "Revision from header = $revHeader"

if ($revHeader -ne $ExpectedRevision) {
    Write-Error "리비전 불일치: Expected=$ExpectedRevision, Actual=$revHeader"
    exit 1
}

Write-Host "리비전 일치: ExpectedRevision OK"


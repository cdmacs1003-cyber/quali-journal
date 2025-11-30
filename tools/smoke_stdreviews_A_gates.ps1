param(
    [string]$BaseUrl          = "https://admin.standardai.co.kr",
    [string]$AdminToken       = $env:ADMIN_TOKEN,
    # B안: REV_NEW 헤더 기반 스모크용 (옵션)
    [string]$ExpectedRevision = $env:EXPECTED_REVISION
)
    $IsDotSourced = $MyInvocation.InvocationName -eq '.'


if (-not $IsDotSourced) {
    if (-not $AdminToken) {
        Write-Error "ADMIN_TOKEN 값을 -AdminToken 인자나 환경변수 ADMIN_TOKEN 으로 전달해야 합니다."
        exit 1
    }

    $BaseUrl = $BaseUrl.TrimEnd('/')
}


# A1/A2/A3 Gate용 테스트 전용 STD ID 매트릭스
# - TEST-STD-1: A1 Gate (기존)
# - TEST-STD-2: A2 Gate (테스트 전용)
# - TEST-STD-3: A3 Gate (테스트 전용)
$StdMatrix = @(
    @{ Id = "TEST-STD-1"; Gate = "A1" },
    @{ Id = "TEST-STD-2"; Gate = "A2" },
    @{ Id = "TEST-STD-3"; Gate = "A3" }
)

function Test-StdReviewRevisionSample {
    param(
        [string]$BaseUrl,
        [string]$AdminToken,
        [string]$ExpectedRevision,
        [int]$MaxAttempts = 100,
        [int]$MinMatches  = 5
    )

    # L3_MAX_ATTEMPTS / L3_MIN_MATCHES 환경변수로 SSOT 상수 덮어쓰기
    if ($env:L3_MAX_ATTEMPTS) {
        try {
            $MaxAttempts = [int]$env:L3_MAX_ATTEMPTS
        } catch {
            Write-Warning "L3_MAX_ATTEMPTS 값을 정수로 변환하지 못했습니다. 기본값 $MaxAttempts 를 사용합니다."
        }
    }
    if ($env:L3_MIN_MATCHES) {
        try {
            $MinMatches = [int]$env:L3_MIN_MATCHES
        } catch {
            Write-Warning "L3_MIN_MATCHES 값을 정수로 변환하지 못했습니다. 기본값 $MinMatches 를 사용합니다."
        }
    }

    if (-not $ExpectedRevision) {
        Write-Host ""
        Write-Host "ExpectedRevision 비어있음 → REV_NEW 헤더 스모크는 건너뜁니다." -ForegroundColor Yellow
        return $true
    }

    Write-Host ""
    Write-Host "=== REV_NEW 헤더 기반 L3 스모크 (B안) 시작 ===" -ForegroundColor Yellow
    Write-Host "  - ExpectedRevision: $ExpectedRevision"
    Write-Host "  - MaxAttempts     : $MaxAttempts"
    Write-Host "  - MinMatches      : $MinMatches"

    $headers = @{
        "X-Admin-Token" = $AdminToken
    }

    $matches = 0
    for ($i = 1; $i -le $MaxAttempts -and $matches -lt $MinMatches; $i++) {
        try {
            $resp = Invoke-WebRequest `
                -Method Get `
                -Uri "$BaseUrl/api/standards/reviews?status=HOLD" `
                -Headers $headers `
                -ErrorAction Stop
        } catch {
            Write-Warning "[$i/$MaxAttempts] REV_NEW 헤더 스모크 요청 실패: $_"
            continue
        }

        $revHeader = $resp.Headers["x-cloud-run-revision"]

        if ($revHeader -and $revHeader -eq $ExpectedRevision -and $resp.StatusCode -eq 200) {
            $matches++
            Write-Host "  -> [$i/$MaxAttempts] REV_NEW 매치 ($matches/$MinMatches): $revHeader" -ForegroundColor Green
        } else {
            Write-Host "  -> [$i/$MaxAttempts] 다른 리비전 응답: '$revHeader'" -ForegroundColor DarkGray
        }
    }

    if ($matches -lt $MinMatches) {
        Write-Error "REV_NEW 헤더 스모크 실패: Expected='$ExpectedRevision', Matches=$matches / $MaxAttempts"
        return $false
    }

    Write-Host ""
    Write-Host "REV_NEW 헤더 스모크 PASS: Expected='$ExpectedRevision', Matches=$matches / $MaxAttempts" -ForegroundColor Green
    return $true
}


function Invoke-StdReviewGate {
    param(
        [string]$StdId,
        [string]$Gate
    )

    Write-Host ""
    Write-Host "=== [StdReviews $Gate] $StdId 상태머신 스모크 ===" -ForegroundColor Cyan

    $headers = @{
        "X-Admin-Token" = $AdminToken
        "Content-Type"  = "application/json"
    }

    # 1) 테스트 카드 시드 생성 (reset=true → HOLD 보장)
    $body = @{ standard_id = $StdId; reset = $true } | ConvertTo-Json -Depth 3
    try {
        $resp = Invoke-RestMethod -Method Post `
            -Uri "$BaseUrl/api/standards/reviews/test/init" `
            -Headers $headers -Body $body -ErrorAction Stop
    } catch {
        Write-Error "[$Gate/$StdId] init 실패: $_"
        throw
    }

    $data = $resp.data
    if (-not $data) { $data = $resp }
    $task = $data.review_task
    if (-not $task) { $task = $data }

    if ($task.standard_id -ne $StdId -or $task.status -ne "HOLD") {
        throw "[$Gate/$StdId] init 결과가 HOLD/$StdId 가 아님 (status=$($task.status))"
    }
    if (($task.approved_by | Measure-Object).Count -ne 0) {
        throw "[$Gate/$StdId] init 시 approved_by 가 비어있지 않음"
    }
    if ($task.required_reviewers -ne 2) {
        throw "[$Gate/$StdId] required_reviewers 가 2가 아님"
    }

    # 2) 1차 승인(r1) – 여전히 HOLD 유지
    $body = @{ reviewer_id = "r1" } | ConvertTo-Json -Depth 3
    try {
        $resp = Invoke-RestMethod -Method Post `
            -Uri "$BaseUrl/api/standards/reviews/$StdId/approve" `
            -Headers $headers -Body $body -ErrorAction Stop
    } catch {
        Write-Error "[$Gate/$StdId] 1차 승인(r1) 실패: $_"
        throw
    }

    $data = $resp.data
    if (-not $data) { $data = $resp }
    $task = $data.review_task
    if (-not $task) { $task = $data }

    if ($task.status -ne "HOLD" -or -not ($task.approved_by -contains "r1")) {
        throw "[$Gate/$StdId] r1 승인 후 상태가 HOLD 이거나 approved_by 에 r1 이 없음"
    }

    # 3) 2차 승인(r2) – REVIEWED 전환 확인
    $body = @{ reviewer_id = "r2" } | ConvertTo-Json -Depth 3
    try {
        $null = Invoke-RestMethod -Method Post `
            -Uri "$BaseUrl/api/standards/reviews/$StdId/approve" `
            -Headers $headers -Body $body -ErrorAction Stop
    } catch {
        Write-Error "[$Gate/$StdId] 2차 승인(r2) 실패: $_"
        throw
    }

    try {
        $resp = Invoke-RestMethod -Method Get `
            -Uri "$BaseUrl/api/standards/reviews?status=REVIEWED" `
            -Headers @{ "X-Admin-Token" = $AdminToken } -ErrorAction Stop
    } catch {
        Write-Error "[$Gate/$StdId] REVIEWED 리스트 조회 실패: $_"
        throw
    }

    $data  = $resp.data
    if (-not $data) { $data = $resp }
    $items = $data.items
    if (-not $items) { $items = $data.reviews }

    $reviewed = $items | Where-Object { $_.standard_id -eq $StdId }
    if (-not $reviewed) {
        throw "[$Gate/$StdId] REVIEWED 리스트에 테스트 STD 가 없음"
    }

    $task = $reviewed[0]
    $approved = @($task.approved_by)
    if ($task.status -ne "REVIEWED" -or
        -not ($approved -contains "r1") -or
        -not ($approved -contains "r2")) {
        throw "[$Gate/$StdId] REVIEWED 상태/승인자 검증 실패"
    }

    # 4) publish – PUBLISHED + PASS 확인
    $publishHeaders = @{
        "X-Admin-Token"  = $AdminToken
        "Content-Length" = "0"
    }
    try {
        $null = Invoke-RestMethod -Method Post `
            -Uri "$BaseUrl/api/standards/reviews/$StdId/publish" `
            -Headers $publishHeaders -ErrorAction Stop
    } catch {
        Write-Error "[$Gate/$StdId] publish 실패: $_"
        throw
    }

    try {
        $resp = Invoke-RestMethod -Method Get `
            -Uri "$BaseUrl/api/standards/reviews?status=PUBLISHED" `
            -Headers @{ "X-Admin-Token" = $AdminToken } -ErrorAction Stop
    } catch {
        Write-Error "[$Gate/$StdId] PUBLISHED 리스트 조회 실패: $_"
        throw
    }

    $data  = $resp.data
    if (-not $data) { $data = $resp }
    $items = $data.items
    if (-not $items) { $items = $data.reviews }

    $published = $items | Where-Object { $_.standard_id -eq $StdId }
    if (-not $published) {
        throw "[$Gate/$StdId] PUBLISHED 리스트에 테스트 STD 가 없음"
    }

    $task = $published[0]
    if ($task.status -ne "PUBLISHED" -or $task.decision -ne "PASS") {
        throw "[$Gate/$StdId] 최종 상태가 PUBLISHED/PASS 가 아님 (status=$($task.status), decision=$($task.decision))"
    }

    Write-Host "OK  [$Gate/$StdId] HOLD → REVIEWED → PUBLISHED(PASS) 스모크 통과" -ForegroundColor Green
}

if (-not $IsDotSourced) {
    $overallOk = $true

    # B안: REV_NEW 헤더 기반 L3 스모크 (옵션)
    if ($ExpectedRevision) {
        $revOk = Test-StdReviewRevisionSample `
            -BaseUrl $BaseUrl `
            -AdminToken $AdminToken `
            -ExpectedRevision $ExpectedRevision

        if (-not $revOk) {
            $overallOk = $false
        }
    }

    foreach ($std in $StdMatrix) {
        try {
            Invoke-StdReviewGate -StdId $std.Id -Gate $std.Gate
        } catch {
            $overallOk = $false
            Write-Error $_
        }
    }

    if (-not $overallOk) {
        Write-Error "일부 Gate 스모크가 실패했습니다."
        exit 1
    } else {
        Write-Host ""
        Write-Host "=== StdReviews A1/A2/A3 L3 스모크 전체 통과 ===" -ForegroundColor Green
    }
}


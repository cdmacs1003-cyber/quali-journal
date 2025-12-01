# tools/smoke_stdreviews_A_gates.ps1
# L3/L5 표준 리뷰 게이트 공용 스모크 스크립트
# - Test-StdReviewRevisionSample 함수만 정의
# - GitHub Actions에서는 dot-source (. script.ps1) 로 불러서 사용
function Use-AdminTokenFromSecret {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $ProjectId
    )

    # 🟡확인 필요: ADMIN_TOKEN 은 GitHub Secrets → env.ADMIN_TOKEN 으로 주입됨.
    # L3 B-canary SSOT 설계에서는 별도 gcloud 호출 없이 이 값을 그대로 사용한다.
    if (-not $env:ADMIN_TOKEN) {
        throw "ADMIN_TOKEN environment variable is empty. Check GitHub Secrets / workflow env."
    }

    return $env:ADMIN_TOKEN
}



$ErrorActionPreference = "Stop"

# L3 B안 REV_NEW 헤더 스모크 함수
function Test-StdReviewRevisionSample {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string] $BaseUrl,

        [Parameter(Mandatory = $true)]
        [string] $ExpectedRevision,

        [Parameter(Mandatory = $true)]
        [string] $AdminToken
    )

    Write-Host "=== L3 stdreviews REV_NEW smoke 시작 ==="
    Write-Host "BaseUrl          = $BaseUrl"
    Write-Host "ExpectedRevision = $ExpectedRevision"

    if (-not $BaseUrl) {
        Write-Error "BaseUrl 가 비어 있습니다."
        return $false
    }

    if (-not $ExpectedRevision) {
        Write-Error "ExpectedRevision 이 비어 있습니다."
        return $false
    }

    if (-not $AdminToken) {
        Write-Error "AdminToken(ADMIN_TOKEN) 이 비어 있습니다."
        return $false
    }

    # BaseUrl 끝의 '/' 정리
    if ($BaseUrl.EndsWith('/')) {
        $BaseUrl = $BaseUrl.TrimEnd('/')
    }

    # 공통 헤더 (ADMIN_TOKEN)
    $headers = @{
        "X-Admin-Token" = $AdminToken
    }

    # A1 Gate Runbook과 맞추기 위해 테스트 카드 시드 엔드포인트 사용
    # 표준 리뷰 상태머신 테스트용 카드: TEST-STD-1
    $bodyObj = @{
        standard_id = "TEST-STD-1"
        reset       = $true
    }

    try {
        Write-Host "POST $BaseUrl/api/standards/reviews/test/init"
        $response = Invoke-WebRequest `
            -Method Post `
            -Uri "$BaseUrl/api/standards/reviews/test/init" `
            -Headers $headers `
            -ContentType "application/json" `
            -Body ($bodyObj | ConvertTo-Json -Depth 5) `
            -ErrorAction Stop
        
        # ==== 여기부터 응답 헤더 덤프 추가 ====
        Write-Host "Response status code = $($response.StatusCode)"
        Write-Host "Response headers:"
        foreach ($h in $response.Headers.GetEnumerator()) {
            Write-Host ("  {0}: {1}" -f $h.Name, ($h.Value -join ','))
        }
    }
    catch {
        Write-Error "표준 리뷰 test/init 호출 실패: $($_.Exception.Message)"
        return $false
    }

    # 🟡확인 필요: 서버에서 실제로 사용하는 헤더 이름
    #   - 현재 가정: 응답 헤더에 'REV_NEW' 라는 커스텀 헤더로
    #     Cloud Run 리비전 이름이 들어온다.
    # 1차 시도: REV_NEW 커스텀 헤더 (서버에서 향후 설정 예정)
    $revHeader = $response.Headers["REV_NEW"]

    # 2차 시도: Cloud Run 기본 리비전 헤더(x-cloud-run-revision) 사용 🟡헤더 이름 실제 값 확인 필요
    if (-not $revHeader) {
        $revHeader = $response.Headers["x-cloud-run-revision"]
        if ($revHeader) {
            Write-Host "x-cloud-run-revision 헤더에서 리비전 값을 찾았습니다: $revHeader"
        }
    }

    if (-not $revHeader) {
        Write-Error "REV_NEW / x-cloud-run-revision 헤더가 응답에 없습니다. (서버에서 리비전 헤더를 설정하고 있는지 확인 필요)"
        return $false
    }

    Write-Host "L3 stdreviews 리비전 헤더 = $revHeader"


    if ($revHeader -ne $ExpectedRevision) {
        Write-Error "REV_NEW 불일치: Expected=$ExpectedRevision, Actual=$revHeader"
        return $false
    }

    Write-Host "REV_NEW header OK: $revHeader"
    Write-Host "=== L3 stdreviews REV_NEW smoke 성공 ==="
    return $true
}

# dot-source 여부 플래그 (헌법 5.2 설계 반영)
$IsDotSourced = $MyInvocation.InvocationName -eq '.'

if (-not $IsDotSourced) {
    # 직접 실행 모드(옵션): 환경 변수나 인자를 활용해 1회 스모크 실행
    $envBase   = $env:BASE_URL
    $envRev    = $env:EXPECTED_REVISION
    $envToken  = $env:ADMIN_TOKEN

    if ($envBase -and $envRev -and $envToken) {
        $ok = Test-StdReviewRevisionSample `
            -BaseUrl          $envBase `
            -ExpectedRevision $envRev `
            -AdminToken       $envToken

        if (-not $ok) {
            exit 1
        }
    }
}

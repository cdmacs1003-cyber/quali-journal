param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [string]$SourceCommit = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    if (-not $SourceCommit) {
        $SourceCommit = (git rev-parse HEAD).Trim()
    }
    if ($SourceCommit -notmatch "^[0-9a-f]{40}$") {
        throw "SourceCommit must be one full lowercase Git commit hash."
    }
    if ((git rev-parse HEAD).Trim() -ne $SourceCommit) {
        throw "SourceCommit must equal the checked-out HEAD."
    }

    git diff --quiet --
    if ($LASTEXITCODE -ne 0) {
        throw "Tracked worktree must be clean before artifact creation."
    }
    git diff --cached --quiet --
    if ($LASTEXITCODE -ne 0) {
        throw "Index must be clean before artifact creation."
    }

    $SourceBranch = (git rev-parse --abbrev-ref HEAD).Trim()
    $SourceDateEpoch = (git show -s --format=%ct $SourceCommit).Trim()
    $OutputParent = Split-Path -Parent $OutputPath
    if (-not $OutputParent) {
        $OutputParent = (Get-Location).Path
        $OutputPath = Join-Path $OutputParent $OutputPath
    }
    New-Item -ItemType Directory -Force -Path $OutputParent | Out-Null
    $ResolvedParent = (Resolve-Path $OutputParent).Path
    $ResolvedOutput = Join-Path $ResolvedParent (Split-Path -Leaf $OutputPath)
    if (Test-Path -LiteralPath $ResolvedOutput) {
        throw "OutputPath already exists; use a new task-owned path."
    }
    $DockerDestination = $ResolvedOutput.Replace("\", "/")
    $Tag = "qlib-skillup-runtime:r469a-$($SourceCommit.Substring(0, 12))"

    docker buildx build `
        --no-cache `
        --provenance=false `
        --platform linux/amd64 `
        -f deploy/qlib-skillup-runtime/Dockerfile `
        --build-arg SOURCE_REPOSITORY=quali-journal-track-a-clean-standalone `
        --build-arg SOURCE_BRANCH=$SourceBranch `
        --build-arg SOURCE_COMMIT=$SourceCommit `
        --build-arg SOURCE_DATE_EPOCH=$SourceDateEpoch `
        --output "type=oci,dest=$DockerDestination,rewrite-timestamp=true,name=$Tag" `
        .
    if ($LASTEXITCODE -ne 0) {
        throw "QLIB runtime OCI artifact build failed."
    }

    $Artifact = Get-Item -LiteralPath $ResolvedOutput
    $Sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ResolvedOutput).Hash.ToLowerInvariant()
    [pscustomobject]@{
        artifact_path = $Artifact.FullName
        artifact_bytes = $Artifact.Length
        artifact_sha256 = $Sha256
        source_branch = $SourceBranch
        source_commit = $SourceCommit
        source_date_epoch = [long]$SourceDateEpoch
        target_service = "qlib-skillup-runtime"
        target_mode = "authenticated_limited_field_beta"
        image_tag = $Tag
        deployment_executed = $false
        registry_write = $false
    } | ConvertTo-Json -Compress
}
finally {
    Pop-Location
}

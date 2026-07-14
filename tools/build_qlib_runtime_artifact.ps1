param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [string]$SourceCommit = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CanonicalBuilder = "desktop-linux"
$ExpectedDockerDesktopVersion = "4.50.0.209931"
$ExpectedDockerVersion = "28.5.1"
$ExpectedBuildxVersion = "v0.29.1-desktop.1"
$ExpectedBuildKitVersion = "v0.25.1"

function Assert-CanonicalDockerToolchain {
    $DockerDesktopPath = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $DockerDesktopPath)) {
        throw "Canonical Docker Desktop executable is missing."
    }
    $DockerDesktopVersion = (Get-Item -LiteralPath $DockerDesktopPath).VersionInfo.ProductVersion
    if ($DockerDesktopVersion -ne $ExpectedDockerDesktopVersion) {
        throw "Docker Desktop version mismatch: expected $ExpectedDockerDesktopVersion."
    }

    $DockerVersionLine = (docker version --format '{{.Client.Version}}|{{.Server.Version}}').Trim()
    if ($LASTEXITCODE -ne 0 -or $DockerVersionLine -ne "$ExpectedDockerVersion|$ExpectedDockerVersion") {
        throw "Docker client/server version mismatch: expected $ExpectedDockerVersion."
    }

    $BuildxText = (docker buildx version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $BuildxText -notmatch [regex]::Escape($ExpectedBuildxVersion)) {
        throw "Docker buildx version mismatch: expected $ExpectedBuildxVersion."
    }

    $BuilderText = docker buildx inspect $CanonicalBuilder --bootstrap | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Canonical builder $CanonicalBuilder is unavailable."
    }
    if ($BuilderText -notmatch '(?m)^Driver:\s+docker\s*$') {
        throw "Canonical builder driver must be docker."
    }
    if ($BuilderText -notmatch "(?m)^BuildKit version:\s+$([regex]::Escape($ExpectedBuildKitVersion))\s*$") {
        throw "BuildKit version mismatch: expected $ExpectedBuildKitVersion."
    }
    if ($BuilderText -notmatch '(?m)^Platforms:\s+.*\blinux/amd64\b') {
        throw "Canonical builder must support linux/amd64."
    }

    [pscustomobject]@{
        docker_desktop = $DockerDesktopVersion
        docker_client = $ExpectedDockerVersion
        docker_server = $ExpectedDockerVersion
        buildx = $ExpectedBuildxVersion
        buildkit = $ExpectedBuildKitVersion
        builder = $CanonicalBuilder
        driver = "docker"
        platform = "linux/amd64"
    }
}

Push-Location $RepoRoot
try {
    $Toolchain = Assert-CanonicalDockerToolchain
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
    $Tag = "qlib-skillup-runtime:r471-$($SourceCommit.Substring(0, 12))"

    docker buildx build `
        --builder $CanonicalBuilder `
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
        toolchain = $Toolchain
        task_id = "R9ZNW-471"
        deployment_executed = $false
        registry_write = $false
    } | ConvertTo-Json -Compress
}
finally {
    Pop-Location
}

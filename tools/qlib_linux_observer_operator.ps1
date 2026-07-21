[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepositoryRoot = "",

    [Parameter()]
    [switch]$SmokeAbsencePath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}

function Write-OperatorResult {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("PASS", "HOLD", "FAIL")]
        [string]$Status,

        [Parameter(Mandatory = $true)]
        [ValidatePattern("^[A-Z0-9_]+$")]
        [string]$Cause,

        [Parameter(Mandatory = $true)]
        [ValidatePattern("^[A-Z0-9_]+$")]
        [string]$SafeAction,

        [Parameter(Mandatory = $true)]
        [ValidatePattern("^[A-Z0-9_]+$")]
        [string]$NextAction,

        [Parameter(Mandatory = $true)]
        [ValidatePattern("^[A-Z0-9_]+$")]
        [string]$EvidenceScope
    )

    Write-Output "$Status | cause=$Cause | safe_action=$SafeAction | next_action=$NextAction | evidence_scope=$EvidenceScope"
}

if ($SmokeAbsencePath) {
    Write-OperatorResult `
        -Status "HOLD" `
        -Cause "WSL2_NOT_AVAILABLE" `
        -SafeAction "NO_INSTALL_ATTEMPTED" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "ABSENCE_PATH_SMOKE"
    exit 0
}

$wslCommand = Get-Command -Name "wsl.exe" -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    Write-OperatorResult `
        -Status "HOLD" `
        -Cause "WSL2_NOT_AVAILABLE" `
        -SafeAction "NO_INSTALL_ATTEMPTED" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 2
}

$distributionNames = @(& $wslCommand.Source --list --quiet 2>$null) |
    ForEach-Object { ([string]$_).Replace([char]0, "").Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
$distributionListExit = $LASTEXITCODE
$selectedDistribution = $distributionNames |
    Where-Object { $_ -notmatch "^(?i:docker-desktop(?:-data)?)$" } |
    Select-Object -First 1
if (($distributionListExit -ne 0) -or [string]::IsNullOrWhiteSpace([string]$selectedDistribution)) {
    Write-OperatorResult `
        -Status "HOLD" `
        -Cause "WSL2_DISTRIBUTION_NOT_AVAILABLE" `
        -SafeAction "NO_INSTALL_ATTEMPTED" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 2
}

$kernelIdentityLines = @(
    & $wslCommand.Source --distribution $selectedDistribution --exec uname -r 2>$null
)
$kernelIdentityExit = $LASTEXITCODE
$kernelIdentity = if ($kernelIdentityLines.Count -eq 1) {
    ([string]$kernelIdentityLines[0]).Replace([char]0, "").Trim()
} else {
    ""
}
if (($kernelIdentityExit -ne 0) -or
    ($kernelIdentity -notmatch "(?i)(?:microsoft-standard-wsl2|wsl2)")) {
    Write-OperatorResult `
        -Status "HOLD" `
        -Cause "WSL2_DISTRIBUTION_NOT_AVAILABLE" `
        -SafeAction "NO_INSTALL_ATTEMPTED" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 2
}

try {
    $resolvedRoot = (Resolve-Path -LiteralPath $RepositoryRoot -ErrorAction Stop).Path
} catch {
    Write-OperatorResult `
        -Status "FAIL" `
        -Cause "REPOSITORY_ROOT_NOT_AVAILABLE" `
        -SafeAction "NO_LINUX_PROCESS_STARTED" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 3
}

$linuxRootLines = @(
    & $wslCommand.Source --distribution $selectedDistribution --exec wslpath -a $resolvedRoot 2>$null
)
if (($LASTEXITCODE -ne 0) -or ($linuxRootLines.Count -ne 1) -or
    [string]::IsNullOrWhiteSpace([string]$linuxRootLines[0])) {
    Write-OperatorResult `
        -Status "HOLD" `
        -Cause "WSL2_REPOSITORY_PATH_UNAVAILABLE" `
        -SafeAction "NO_ACCEPTANCE_RUN_STARTED" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 2
}

$linuxRoot = ([string]$linuxRootLines[0]).Trim()
$syntheticId = [Guid]::NewGuid().ToString("N")
$evidenceDirectory = "/tmp/r9znw-488d42-operator-$syntheticId"
$acceptanceArguments = @(
    "--distribution", $selectedDistribution,
    "--cd", $linuxRoot,
    "--exec", "python3",
    "tools/qlib_linux_process_supervisor.py",
    "acceptance",
    "--output", $evidenceDirectory,
    "--deterministic-repeats", "3",
    "--actual-repeats", "10",
    "--stress-seeds", "100"
)

& $wslCommand.Source @acceptanceArguments 1>$null 2>$null
$acceptanceExit = $LASTEXITCODE
# A local WSL run intentionally lacks GitHub runner metadata, so a structurally
# complete 310-case bundle closes as HOLD (exit 2) until the local-only verifier
# proves the campaign and the authoritative GitHub workflow runs.
if (($acceptanceExit -ne 0) -and ($acceptanceExit -ne 2)) {
    Write-OperatorResult `
        -Status "FAIL" `
        -Cause "WSL_FALLBACK_ACCEPTANCE_FAILED" `
        -SafeAction "NO_RETRY_NO_INSTALL" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 4
}

$verificationArguments = @(
    "--distribution", $selectedDistribution,
    "--cd", $linuxRoot,
    "--exec", "python3",
    "tools/qlib_linux_process_supervisor.py",
    "verify-artifact",
    "--input", $evidenceDirectory,
    "--allow-local-non-github"
)

& $wslCommand.Source @verificationArguments 1>$null 2>$null
$verificationExit = $LASTEXITCODE

if ($verificationExit -ne 0) {
    Write-OperatorResult `
        -Status "FAIL" `
        -Cause "WSL_FALLBACK_EVIDENCE_NOT_VERIFIED" `
        -SafeAction "PRESERVE_SANITIZED_FAILED_BUNDLE_NO_RETRY" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 5
}

$cleanupArguments = @(
    "--distribution", $selectedDistribution,
    "--cd", $linuxRoot,
    "--exec", "python3",
    "tools/qlib_linux_process_supervisor.py",
    "cleanup-artifact",
    "--input", $evidenceDirectory
)
& $wslCommand.Source @cleanupArguments 1>$null 2>$null
$cleanupExit = $LASTEXITCODE

if ($cleanupExit -ne 0) {
    Write-OperatorResult `
        -Status "FAIL" `
        -Cause "WSL_FALLBACK_EVIDENCE_CLEANUP_FAILED" `
        -SafeAction "NO_RETRY_NO_INSTALL" `
        -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
        -EvidenceScope "LOCAL_WSL_FALLBACK"
    exit 6
}

Write-OperatorResult `
    -Status "HOLD" `
    -Cause "LOCAL_WSL_ACCEPTANCE_VERIFIED_NOT_GITHUB" `
    -SafeAction "NO_INSTALL_NO_EXTERNAL_MUTATION" `
    -NextAction "RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW" `
    -EvidenceScope "LOCAL_WSL_FALLBACK_NOT_GITHUB"
exit 0

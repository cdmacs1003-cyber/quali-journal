# tools/gate_contract_check.ps1  (admin 폴더 기준으로 실행됨)
# Gate: Contract Check (Stage 2)
# - workflow 문서가 contract 문서를 참조하는지
# - contract 문서에 핵심 계약 패턴(/api/logs args kwargs 200 422)이 있는지
# - PASS/FAIL을 CI에서 고정

[CmdletBinding()]
param(
  [string]$WorkflowPath = ".\docs\workflows\Trinity_Dev_Workflow_2025-12-13_threshold90.md",
  [string]$ContractPath = ".\docs\contracts\Contract_Catalog_QualiJournal_Admin_2025-12-13_v3.md",
  [string]$ContractRefString = "docs/contracts/Contract_Catalog_QualiJournal_Admin_2025-12-13_v3.md",
  [string[]]$RequiredContractPatterns = @("/api/logs", "args", "kwargs", "200", "422")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail([string]$msg) { Write-Error $msg; exit 1 }
function Pass([string]$msg) { Write-Host  $msg; exit 0 }

# 1) 파일 존재 Gate
if (-not (Test-Path -LiteralPath $WorkflowPath)) { Fail "FAIL: missing workflow file: $WorkflowPath" }
if (-not (Test-Path -LiteralPath $ContractPath))  { Fail "FAIL: missing contract file:  $ContractPath" }

# 2) Workflow  Contract 링크 무결성 Gate
$refOk = Select-String -Path $WorkflowPath -Pattern ([regex]::Escape($ContractRefString)) -Quiet
if (-not $refOk) { Fail "FAIL: workflow does not reference contract path: $ContractRefString" }

# 3) Contract 핵심 패턴 포함 Gate
foreach ($p in $RequiredContractPatterns) {
  $ok = Select-String -Path $ContractPath -Pattern $p -Quiet
  if (-not $ok) { Fail "FAIL: contract missing pattern: $p" }
}

Pass "PASS: Contract Gate OK"

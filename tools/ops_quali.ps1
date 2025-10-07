param(
  [string]$Repo = "$env:USERPROFILE\Desktop\퀄리저널",
  [switch]$Net
)

Set-Location -Path $Repo
$ts   = (Get-Date).ToString("yyyyMMdd_HHmm")
$rep  = Join-Path $Repo "archive\reports"
$new  = Join-Path $rep  "ops_summary_$ts.txt"
if (-not (Test-Path $rep)) { New-Item -ItemType Directory -Path $rep | Out-Null }

# 1) 건강검진
$hcCmd = "py .\data\run_all.py --repo `"$Repo`""
if ($Net) { $hcCmd += " --net" }
$hcOut = (Invoke-Expression $hcCmd) 2>&1 | Out-String

# 2) 링크 상태 점검(인터넷 가능할 때)
$linkOut = ""
if ($Net) {
  try { $linkOut = (py .\tools\verify_sources.py) 2>&1 | Out-String }
  catch { $linkOut = "[verify_sources] 에러: $($_.Exception.Message)" }
}

# 3) 무결성 점검
$intOut = ""
try { $intOut = (py .\tools\validate_selection_integrity.py) 2>&1 | Out-String }
catch { $intOut = "[integrity] 에러: $($_.Exception.Message)" }

# 4) 요약 저장
@"
=== OPS SUMMARY $ts ===

[HealthCheck]
$hcOut

[LinkCheck]
$linkOut

[Integrity]
$intOut
"@ | Set-Content -Encoding UTF8 $new

Write-Host "완료! 요약  $new"

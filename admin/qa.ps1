# ========== Quali QA 원클릭 ==========
chcp 65001 > $null
[Console]::InputEncoding  = New-Object System.Text.UTF8Encoding
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding

$URL   = "http://127.0.0.1:8010"
$TOKEN = $env:ADMIN_TOKEN
if (-not $TOKEN -or $TOKEN.Trim().Length -eq 0) {
  Write-Host "❗ 먼저 $env:ADMIN_TOKEN 를 설정하세요." -ForegroundColor Yellow
  return
}
$H = @{ Authorization = "Bearer $TOKEN" }

Write-Host "`n[1] Reason Tags" -ForegroundColor Cyan
# UTF-8로 강제 디코드해서 출력
$raw  = Invoke-WebRequest "$URL/api/reason-tags" -Headers $H
$ms   = New-Object System.IO.MemoryStream
$raw.RawContentStream.CopyTo($ms)
$json = [System.Text.Encoding]::UTF8.GetString($ms.ToArray())
$json | ConvertFrom-Json | ConvertTo-Json -Depth 3


Write-Host "`n[2] Ready KPI" -ForegroundColor Cyan
Invoke-RestMethod "$URL/api/ready/status" -Headers $H | Format-List

Write-Host "`n[3] Snapshot (monthly)" -ForegroundColor Cyan
$snap = Invoke-RestMethod "$URL/api/archive/snapshot-monthly" -Method POST -Headers $H
$snap | Format-List

Write-Host "`n[4] Weekly Diff" -ForegroundColor Cyan
$diff = Invoke-RestMethod "$URL/api/archive/diff-weekly" -Method POST -Headers $H
$diff | Format-List

Write-Host "`n[5] Verify files (hash)" -ForegroundColor Cyan
$lastSnap = Get-ChildItem ".\archive\snapshots\monthly\*.json" | Sort-Object LastWriteTime -Desc | Select -First 1
$hSnap    = if ($lastSnap) { Get-FileHash $lastSnap.FullName -Algorithm SHA256 | Select -ExpandProperty Hash } else { "" }
$lastDiff = Get-ChildItem ".\archive\diffs\weekly\*.json" | Sort-Object LastWriteTime -Desc | Select -First 1
$hDiff    = if ($lastDiff) { Get-FileHash $lastDiff.FullName -Algorithm SHA256 | Select -ExpandProperty Hash } else { "" }
[PSCustomObject]@{ Snapshot=$lastSnap?.Name; Hash=$hSnap; Diff=$lastDiff?.Name; Hash2=$hDiff }

Write-Host "`n[6] ETag 200→304" -ForegroundColor Cyan
$r1 = Invoke-WebRequest "$URL/api/ready/items" -Headers $H
$etag = $r1.Headers.ETag; $H['If-None-Match'] = $etag
try   { $r2 = Invoke-WebRequest "$URL/api/ready/items" -Headers $H -ErrorAction Stop;  "HTTP 200" }
catch { $resp = $_.Exception.Response; "HTTP " + [int]$resp.StatusCode }
# ====================================
param(
  [Parameter(Mandatory=$true)][string]$URL,
  [Parameter(Mandatory=$true)][string]$TOKEN,
  [ValidateSet("status","set-gate","report","enrich-all","enrich-sel","export-md","export-csv")]
  [string]$Action="status",
  [int]$Value=15,                                 # set-gate에서 게이트 값
  [string]$Date=(Get-Date -Format "yyyy-MM-dd"),  # report/enrich 용
  [string]$Key="IPC-A-610",
  [switch]$Private                                # 비공개 Cloud Run이면 -Private 추가
)

function Get-Headers {
  if ($Private) {
    $idt = (gcloud auth print-identity-token)
    return @{ Authorization = "Bearer $idt"; "X-Admin-Token" = $TOKEN }
  } else {
    return @{ Authorization = "Bearer $TOKEN" }
  }
}

$H = Get-Headers

switch ($Action) {
  "status"      { irm "$URL/api/status" -Headers $H; break }
  "set-gate"    {
      $body = @{ gate_required = $Value } | ConvertTo-Json -Compress
      irm "$URL/api/config/gate_required" -Method PATCH -Headers $H -ContentType "application/json" -Body $body
      irm "$URL/api/status" -Headers $H
      break
  }
  "report"      {
      $body = @{ date = $Date; keyword = $Key } | ConvertTo-Json -Compress
      $r = irm "$URL/api/report" -Method POST -Headers $H -ContentType "application/json" -Body $body
      $r; if ($r.path) { Start-Process "$URL/$($r.path)" }
      break
  }
  "enrich-all"  {
      $body = @{ date = $Date; keyword = $Key } | ConvertTo-Json -Compress
      $k = irm "$URL/api/enrich/keyword" -Method POST -Headers $H -ContentType "application/json" -Body $body
      $k; if ($k.path_all) { Start-Process "$URL/$($k.path_all)" }
      break
  }
  "enrich-sel"  {
      $body = @{ date = $Date; keyword = $Key } | ConvertTo-Json -Compress
      $s = irm "$URL/api/enrich/selection" -Method POST -Headers $H -ContentType "application/json" -Body $body
      $s; if ($s.path_selected) { Start-Process "$URL/$($s.path_selected)" }
      break
  }
  "export-md"   { curl.exe "$URL/api/export/md"  -H "Authorization: $($H.Authorization)" -H "X-Admin-Token: $TOKEN" -o final.md;  Start-Process .\final.md; break }
  "export-csv"  { curl.exe "$URL/api/export/csv" -H "Authorization: $($H.Authorization)" -H "X-Admin-Token: $TOKEN" -o final.csv; Start-Process .\final.csv; break }
}

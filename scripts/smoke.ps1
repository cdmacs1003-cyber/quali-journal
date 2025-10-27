param(
  [string]$Service    = "quali-journal",
  [string]$Region     = "asia-northeast3",
  [string]$AdminToken # 필수: 운영토큰
)

$u = gcloud run services describe $Service --region $Region --format "value(status.url)"
$r = gcloud run services describe $Service --region $Region --format "value(status.traffic)"
Write-Host ("URL: {0}  Traffic: {1}" -f $u, $r)

# 1차: Authorization 헤더로 검사
$res = curl.exe -s -i -H "Authorization: Bearer $AdminToken" "$u/api/status"
if ($res -match "200 OK") { "STATUS ✅ OK (Authorization)" ; return }

# 2차: X-Admin-Token 헤더로 재시도
$res2 = curl.exe -s -i -H "X-Admin-Token: $AdminToken" "$u/api/status"
if ($res2 -match "200 OK") { "STATUS ✅ OK (X-Admin-Token)" ; return }

"STATUS ⚠️ Check: $($res.Split("`n")[0]) / $($res2.Split("`n")[0])"

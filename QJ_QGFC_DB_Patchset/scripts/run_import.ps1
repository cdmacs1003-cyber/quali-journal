# scripts/run_import.ps1  (PowerShell 순정 문법)
param(
  [string]$Work="selected_keyword_articles.json",
  [string]$Final="selected_articles.json",
  [string]$Etype="keyword",
  [string]$Keyword=$null,
  [string]$Edate=(Get-Date -Format "yyyy-MM-dd")
)

# 기본값: 환경변수 없으면 SQLite 사용
if (-not $env:QJ_DB_URL -or [string]::IsNullOrWhiteSpace($env:QJ_DB_URL)) {
  $env:QJ_DB_URL = "sqlite:///./qj.sqlite3"
}

python ./app/import_from_json.py `
  --work $Work `
  --final $Final `
  --etype $Etype `
  --keyword $Keyword `
  --edate $Edate

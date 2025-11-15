# scripts/run_migration.ps1
param(
  [string]$Conn=""
)
if (-not $Conn) {
  Write-Host "예) .\scripts\run_migration.ps1 -Conn `"postgres://USER:PASS@HOST:5432/qj`"" -ForegroundColor Yellow
  exit 1
}
$env:PGPASSWORD = ($Conn -split "[:/@]")[2]
# 순서대로 실행
psql "$Conn" -f "./db/migrations/0001_init.sql"
psql "$Conn" -f "./db/migrations/0002_indexes.sql"

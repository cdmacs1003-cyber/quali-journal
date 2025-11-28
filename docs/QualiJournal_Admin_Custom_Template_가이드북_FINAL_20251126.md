# QualiJournal Admin — 조직 내 실제 값이 반영된 Cloud Run/운영 템플릿 (v1.0)

> 이 문서는 **QualiJournal 관리자 시스템(prod 환경)**에 대해, 프로젝트/서비스/리전/버킷 등 *실제 값*을 모두 채운 실행 템플릿입니다.  
> 운영/개발 팀이 **Windows PowerShell + gcloud** 기준으로 바로 복붙하여 사용할 수 있도록 구성되었습니다.

---

## 1. SSOT(단일 출처 값 요약)

### 1.1 GCP / Cloud Run

- **프로젝트 ID**
  - `quali-journal-prod`
- **Cloud Run 리전**
  - 운영(소스) 서비스: `asia-northeast3`  _(서울)_
  - 도메인 매핑용 서비스: `asia-northeast1`  _(도쿄)_
- **Cloud Run 서비스 이름**
  - 소스 빌드용 서비스: `quali-journal-admin`
  - 도메인 연결 라이브 서비스: `quali-admin-domap`
- **커스텀 도메인**
  - 서비스 도메인: `https://admin.standardai.co.kr`
  - DNS: `CNAME admin.standardai.co.kr → ghs.googlehosted.com`

### 1.2 시크릿 / 환경 변수

- **Secret Manager**
  - 관리자 토큰: Secret 이름 `ADMIN_TOKEN`, 사용 버전 `latest`
  - 실제 토큰 문자열은 *Secret Manager에서만 조회*하며, 문서에는 기록하지 않습니다.
- **주요 환경 변수**
  - `ADMIN_TOKEN` : Secret Manager에서 주입
  - `API_KEY` : (선택) 외부 요약/번역 API 키
  - `DB_URL` : (선택) DB 연결 문자열
  - `GCS_BUCKET` : `qualijournal-archive`

### 1.3 Cloud Scheduler (일일 보고서)

- **잡 이름**: `daily-report`
- **리전**: `asia-northeast1`
- **스케줄**: 매일 09:00 (Asia/Seoul 기준)
- **요청 URL**: `https://admin.standardai.co.kr/api/report`
- **인증 방식**: HTTP 헤더 `X-Admin-Token: <ADMIN_TOKEN 현재값>`

---

## 2. PowerShell 공통 초기 설정

```powershell
# 공통 SSOT 변수
$PROJECT   = "quali-journal-prod"
$REG_SEO   = "asia-northeast3"    # 서울 (소스 서비스 리전)
$REG_TYO   = "asia-northeast1"    # 도쿄 (도메인 매핑 리전)

$SRC       = "quali-journal-admin"  # 소스 빌드용 Cloud Run 서비스
$LIVE      = "quali-admin-domap"    # 도메인 연결 라이브 서비스
$DOMAIN    = "admin.standardai.co.kr"
$BUCKET    = "qualijournal-archive"

# 관리자 토큰은 Secret Manager에서만 조회 (값은 문서에 기재하지 않음)
$TOKEN = (gcloud secrets versions access latest `
  --secret=ADMIN_TOKEN `
  --project $PROJECT | Out-String).Trim()
```

---

## 3. Cloud Run 배포 템플릿

### 3.1 소스 코드 기반 배포 (서울, `quali-journal-admin`)

> 로컬 `admin` 디렉터리 기준. FastAPI + 정적 HTML이 하나의 컨테이너로 빌드됩니다.

```powershell
cd "C:\path	o\quali-journaldmin"

gcloud run deploy $SRC `
  --source . `
  --project $PROJECT `
  --region  $REG_SEO `
  --platform managed `
  --allow-unauthenticated `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" `
  --set-env-vars "PYTHONUTF8=1,PYTHONIOENCODING=utf-8,GCS_BUCKET=$BUCKET"
```

- 배포 완료 후, `status.url` 값은 **서울 리전용 관리 URL**입니다.
- 이 서비스의 이미지를 **도메인용 서비스(quali-admin-domap)** 에 재사용합니다.

### 3.2 도메인 연결용 서비스에 동일 이미지 배포 (`quali-admin-domap`)

```powershell
# 1) 소스 서비스에서 현재 사용 중인 이미지 URI 추출
$IMG = gcloud run services describe $SRC `
  --region  $REG_SEO `
  --project $PROJECT `
  --format  "value(spec.template.spec.containers[0].image)"

$IMG  # 확인용 출력

# 2) 도메인 연결용 서비스에 같은 이미지 배포 (도쿄 리전)
gcloud run deploy $LIVE `
  --image   $IMG `
  --region  $REG_TYO `
  --project $PROJECT `
  --platform managed `
  --allow-unauthenticated `
  --set-secrets "ADMIN_TOKEN=ADMIN_TOKEN:latest" `
  --set-env-vars "PYTHONUTF8=1,PYTHONIOENCODING=utf-8,GCS_BUCKET=$BUCKET"
```

### 3.3 최신 리비전에 100% 트래픽 전환

```powershell
gcloud run services update-traffic $LIVE `
  --to-latest `
  --region  $REG_TYO `
  --project $PROJECT
```

- 기대 결과: `status.traffic` 값이 `latestRevision: True, percent: 100` 인 항목 하나만 남습니다.

---

## 4. 커스텀 도메인 매핑 템플릿

> 이 단계는 이미 구성되어 있을 수 있습니다. 재구성이 필요할 때만 사용합니다.

### 4.1 도메인 매핑 생성

```powershell
gcloud beta run domain-mappings create `
  --service $LIVE `
  --domain  $DOMAIN `
  --region  $REG_TYO `
  --project $PROJECT
```

### 4.2 DNS (가비아) 설정 개요

- 타입: `CNAME`
- 이름: `admin`
- 값: `ghs.googlehosted.com`

> DNS 전파 후, Cloud Run 도메인 매핑 상태의 `Ready=True`, 인증서 상태 `CertificateReady=True` 인지 확인합니다.

---

## 5. Cloud Scheduler — 일일 보고서 자동화 템플릿

### 5.1 HTTP 잡 생성 (X-Admin-Token 헤더 방식)

```powershell
$LOCATION = "asia-northeast1"

# 최초 생성
gcloud scheduler jobs create http daily-report `
  --project  $PROJECT `
  --location $LOCATION `
  --schedule "0 9 * * *" `
  --time-zone "Asia/Seoul" `
  --http-method POST `
  --uri  "https://$DOMAIN/api/report" `
  --headers "X-Admin-Token=$TOKEN,Content-Length=0"
```

### 5.2 잡 설정 변경 시 (토큰 변경 등)

```powershell
gcloud scheduler jobs update http daily-report `
  --project  $PROJECT `
  --location $LOCATION `
  --uri      "https://$DOMAIN/api/report" `
  --headers  "X-Admin-Token=$TOKEN,Content-Length=0"
```

### 5.3 수동 실행 / 상태 확인

```powershell
gcloud scheduler jobs describe daily-report `
  --project  $PROJECT `
  --location $LOCATION `
  --format "value(state,lastAttemptTime,lastScheduleTime)"

gcloud scheduler jobs run daily-report `
  --project  $PROJECT `
  --location $LOCATION
```

---

## 6. GCS 백업/아카이브 정책 템플릿

### 6.1 버전 관리 및 보존 정책

```powershell
# 객체 버전 관리 활성화
gsutil versioning set on gs://$BUCKET

# 보존 기간 365일 설정 (예: 1년)
gsutil retention set 365d gs://$BUCKET

# 설정 확인
gsutil versioning get gs://$BUCKET
gsutil retention  get gs://$BUCKET
```

### 6.2 (선택) 월 1회 복구 리허설 권장 절차

1. 특정 일자의 MD/CSV 파일 1개를 선택해 로컬로 다운로드
2. 로컬 환경에서 동일 경로로 복원 및 파일 무결성 확인
3. 문제가 없다면 “복구 리허설 완료”로 기록

---

## 7. 일일 스모크 테스트 3줄 템플릿

> 운영자가 *하루 1회* 서비스 상태를 확인할 때 사용하는 최소 스크립트입니다.

```powershell
$URL = gcloud run services describe $LIVE `
  --region  $REG_TYO `
  --project $PROJECT `
  --format "value(status.url)"

$TOKEN = (gcloud secrets versions access latest `
  --secret=ADMIN_TOKEN `
  --project $PROJECT | Out-String).Trim()

Write-Host "HEALTH :" (curl.exe -s -o NUL -w "%{http_code}" "$URL/health")
Write-Host "STATUS :" (curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/status")
Write-Host "REPORT :" (curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/api/report")
```

- **기대값**
  - HEALTH: `200`
  - STATUS: `200` (토큰 O), `401` (토큰 X)
  - REPORT: `200` (토큰 O)

---

### L3 표준 리뷰 A1 Gate 스모크 템플릿

> 표준 리뷰 큐와 2인 검수 상태머신이 정상인지, 운영자가 수동으로 확인할 때 사용하는 스크립트입니다.  
> 기본 테스트 카드 ID는 `TEST-STD-1` 입니다.

```powershell
# 표준 리뷰 A1 Gate 상태머신 점검 스크립트
# 전제: 위에서 설정한 $URL, $TOKEN 값을 그대로 사용한다.
$STD_ID = "TEST-STD-1"

# 1) 테스트 카드 시드 생성/초기화 (HOLD 상태 보장)
$bodyObj = @{ standard_id = $STD_ID; reset = $true }
Invoke-RestMethod -Method Post -Uri "$URL/api/standards/reviews/test/init" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json)

# 2) HOLD 리스트 확인
Write-Host "HOLD 리스트 확인..."
Invoke-RestMethod -Method Get -Uri "$URL/api/standards/reviews?status=HOLD" `
  -Headers @{ "X-Admin-Token" = $TOKEN } |
  ConvertTo-Json -Depth 5

# 3) r1 / r2 승인 (2인 검수)
$bodyObj = @{ reviewer_id = "r1" }
Invoke-RestMethod -Method Post -Uri "$URL/api/standards/reviews/$STD_ID/approve" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json)

$bodyObj = @{ reviewer_id = "r2" }
Invoke-RestMethod -Method Post -Uri "$URL/api/standards/reviews/$STD_ID/approve" `
  -Headers @{ "X-Admin-Token" = $TOKEN } `
  -ContentType "application/json" `
  -Body ($bodyObj | ConvertTo-Json)

# 4) 발행(publish) + PUBLISHED/PASS 확인
Invoke-RestMethod -Method Post -Uri "$URL/api/standards/reviews/$STD_ID/publish" `
  -Headers @{ "X-Admin-Token" = $TOKEN; "Content-Length" = "0" }

Write-Host "PUBLISHED 리스트 확인..."
Invoke-RestMethod -Method Get -Uri "$URL/api/standards/reviews?status=PUBLISHED" `
  -Headers @{ "X-Admin-Token" = $TOKEN } |
  ConvertTo-Json -Depth 5



## 8. Definition of Done 체크리스트 (이 템플릿 기준)

아래 항목을 모두 충족하면 “운영 가능한 상태”로 간주합니다.

- [ ] `quali-journal-prod` 프로젝트에 `quali-journal-admin`, `quali-admin-domap` 서비스가 존재한다.
- [ ] 두 서비스 모두 `ADMIN_TOKEN`(Secret), `GCS_BUCKET=qualijournal-archive` 환경 변수를 갖는다.
- [ ] `quali-admin-domap` 최신 리비전에 트래픽 100%가 할당되어 있다.
- [ ] `admin.standardai.co.kr` → `quali-admin-domap` 도메인 매핑이 `Ready=True` 상태다.
- [ ] Cloud Scheduler `daily-report` 잡이 `asia-northeast1` 리전에 존재하고, 09:00 Asia/Seoul에 `/api/report`를 호출한다.
- [ ] `gs://qualijournal-archive` 버킷에 버전 관리 및 365일 보존 정책이 적용되어 있다.
- [ ] 일일 스모크 3줄 실행 시 HEALTH/STATUS/REPORT 의 HTTP 코드가 기대값을 만족한다.

---

> 이 파일은 “조직 내 실제 값(프로젝트 ID, 서비스명, 버킷명 등)”이 반영된 **최종 실행 템플릿**입니다.  
> 신규 운영자/개발자는 이 문서의 SSOT 값을 기준으로, 배포·도메인·스케줄러·백업을 재현할 수 있습니다.

# QualiJournal 관리자 시스템 – 개발자/운영자 실행 체크리스트

## ✅ 완료된 작업 요약
- 관리자 패널 UI 개선 및 주요 API(`/api/report`, `/api/enrich/*`, `/api/export/*`) 완성  
- Cloud Run 서비스 배포 및 환경변수(ADMIN_TOKEN 등) 설정 완료  
- KPI 자동 갱신 및 임계값 슬라이더 UI 구현  
- JWT 기반 인증 적용 및 환경 변수 Secret Manager 연동  
- Cloud Run 배포 절차 표준화 및 스모크 테스트 수행 완료【417†source】【418†source】  

---

## 🚧 남은 주요 과제
1. **WIF(Workload Identity Federation) 기반 CI/CD 인증 안정화**  
2. **커스텀 도메인 연결 (Gabia DNS → Cloud Run)**  
3. **Cloud Scheduler를 통한 일일 보고서 자동화**  
4. **GCS 백업 자동화 및 버전 관리 정책 적용**  
5. **UI 테마/스타일 QA 및 접근성 개선**  
6. **Cloud Run Invoker 권한 최소화(보안 하드닝)**【419†source】【421†source】  

---

## 🧭 실행 가이드 (Runbook)

### 1️⃣ WIF 기반 인증 안정화
**사전 점검**
- [ ] GitHub 워크플로우에 `permissions: id-token: write` 설정 포함  
- [ ] GCP 서비스계정에 `roles/iam.workloadIdentityUser` 및 `roles/run.admin` 권한 부여  
- [ ] Workload Identity Provider 상태(issuerUri, attributeCondition) 확인  
- [ ] audience 불일치/invalid_target 에러 로그 점검【419†source】  

**명령 예시**
```bash
gcloud iam service-accounts add-iam-policy-binding <SA_EMAIL>   --role roles/iam.workloadIdentityUser   --member "principalSet://iam.googleapis.com/projects/<NUM>/locations/global/workloadIdentityPools/<POOL_ID>/attribute.repository/<ORG>/<REPO>"
```

**검증**
- GitHub Actions 재실행 → “Authentication successful” 로그 확인  
- Cloud Run 배포 성공 및 `/health` 200 OK 응답 확인  

---

### 2️⃣ 커스텀 도메인 연결
**단계별 절차**
1. 도메인 인증 확인  
   ```bash
   gcloud domains verify standardai.co.kr
   ```
2. Cloud Run 매핑 생성  
   ```bash
   gcloud beta run domain-mappings create --service <SERVICE_NAME>      --domain standardai.co.kr --region asia-northeast3
   ```
3. 가비아 DNS에 CNAME 등록 → `ghs.googlehosted.com`【418†source】  

**검증**
- `https://standardai.co.kr` 접속 및 SSL 인증서 활성화 확인  

---

### 3️⃣ Cloud Scheduler 자동화
**Cloud Run 엔드포인트**: `POST /api/report`  
**OIDC 방식 호출 예시**
```bash
gcloud scheduler jobs create http daily-report   --schedule="0 9 * * *"   --http-method=POST   --uri="https://<SERVICE_URL>/api/report"   --oidc-service-account-email="scheduler-invoker@<PROJECT_ID>.iam.gserviceaccount.com"   --oidc-token-audience="https://<SERVICE_URL>"
```
- Scheduler와 Run은 동일 리전에 설정(asia-northeast3)  
- 첫 실행은 `gcloud scheduler jobs run daily-report`로 수동 검증【418†source】  

---

### 4️⃣ GCS 백업 자동화
**설정 명령**
```bash
gsutil versioning set on gs://<bucket>
gsutil retention set 365d gs://<bucket>
```
- Cloud Scheduler 또는 Cloud Function을 활용한 주기적 업로드  
- IAM 최소권한: `roles/storage.objectAdmin`【418†source】  

---

### 5️⃣ UI 테마·접근성 QA
**체크리스트**
- [ ] 라이트/다크 모드 대비 및 색상 일관성 확인  
- [ ] 반응형(모바일·태블릿) 테스트  
- [ ] 툴팁/로딩 피드백 및 단축키 작동 여부 검증【418†source】  

---

### 6️⃣ Cloud Run 보안 하드닝
**명령 예시**
```bash
gcloud run services update <SERVICE_NAME>   --no-allow-unauthenticated --region asia-northeast3
```
- Invoker 최소 권한: Scheduler SA 등 지정【421†source】  

---

## 📅 운영자 일일 점검표 (Ops Checklist)
| 구분 | 점검 항목 | 확인 |
|------|------------|------|
| 서비스 상태 | `/health` 200 OK, KPI 갱신 확인 | [ ] |
| 스케줄러 | `/api/report` 실행 로그 및 GCS 백업 확인 | [ ] |
| 보안 | Cloud Run Invoker 정책, ADMIN_TOKEN 교체 주기 | [ ] |
| UI | 슬라이더, 테마, 반응형 QA 결과 반영 | [ ] |

---

## ✅ Definition of Done
| 항목 | 완료 조건 |
|------|------------|
| WIF 인증 | main/PR 브랜치 모두 3회 연속 성공, invalid_target 미발생 |
| 도메인 연결 | 커스텀 도메인 HTTPS 연결 및 인증서 Active |
| 스케줄러 | 자동 보고서 7일 연속 성공, 실패율 <1% |
| 백업 | GCS 버킷 Versioning 및 Retention 정책 적용 |
| UI | 다크모드·단축키 등 QA 통과, 문서 스크린샷 반영 |

---

**참고 문서:**  
- `1016_1QualiJournal 관리자 시스템 인수인계 보고서.pdf`【417†source】  
- `1017_1QualiJournal Cloud Run 배포 및 운영 가이드.pdf`【418†source】  
- `1018_2WIF 기반 Cloud Run 인증 문제 평가 및 권장 방안.pdf`【419†source】  
- `1018_4Google Cloud WIF 인증 이슈 해결 가이드북.pdf`【421†source】  


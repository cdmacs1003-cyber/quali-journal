# QualiJournal 표준·기술 자료 큐레이션 + 관리자 모드(Admin API) 통합 헌법 v2
(통합 기준일: 2025-11-19, 헌법 버전: 2.0)

이 문서는 QualiJournal 표준·기술 자료 큐레이션과 관리자 모드(Admin)의
스펙·DoD·운영 원칙을 하나로 묶은 **단일 진실(SSOT) 헌법**입니다.

통합 대상 원문은 다음 세 계열입니다.

1. `nSSOT_QualiJournal_Standards_Admin_Constitution_20251119_v2.md`  
   - 표준·기술 자료 큐레이션 SSOT + Admin(domap) 표준 스코어링/릴리스 DoD 통합본
2. `SSOT_Admin_DoD_AdminAPI_v1.md`  
   - QualiJournal 관리자 모드 SSOT (Admin DoD v1.1)
3. `QualiJournal_Admin_DoD_AdminAPI_20251116.md`  
   - 관리자 모드 DoD – Admin API 응답 스키마 마감 보고서

본 v2 헌법의 구성은 다음과 같습니다.

- **0장. 통합 메타 & 공통 축(표준·Admin·도메인/domap) 고정**
  - 표준 큐레이션 4축 스코어, PASS/HOLD 경계, 2인 검수 워크플로우,
    Admin API 응답 스키마, Cloud Run domap Release DoD 사이의 관계를 상위 레벨에서 고정합니다.
- **I장. 표준·기술 자료 큐레이션 + Admin(domap) 통합 헌법 v1 (원문 그대로 포함)**  
  - 표준 큐레이션 SSOT와 표준 스코어링/검수/domap 스모크에 대한 기존 헌법 원문입니다.
- **II장. QualiJournal 관리자 모드 SSOT (Admin DoD v1.1) 원문**  
  - 관리자 모드 전체 DoD·Cloud SQL Stop Policy·DevOps 품질 게이트 SSOT 원문입니다.
- **III장. QualiJournal 관리자 모드 DoD – Admin API 응답 스키마 마감 보고서 원문**  
  - Admin API 엔드포인트/응답 스키마/HTTP 코드 정렬에 대한 DoD 마감 보고서입니다.

아래 0장은 세 문서를 연결하는 메타 요약이며,
I~III장은 **원문을 수정 없이 그대로** 포함합니다.
(기존 문단·표·코드블록은 내용 불변 원칙을 따른다.)

---

## 0장. 통합 메타 & 공통 축 고정

### 0-1. 역할 분담(표준 큐레이션 vs Admin vs domap)

- **표준·기술 자료 큐레이션 SSOT**는
  - 표준/기술 문서의 수집·전처리·4축 스코어링·PASS/HOLD/BAD 결정·발행/아카이브 전 과정을 정의합니다.
  - 특히 `score_regular/applic/evid/trust` 4축(각 0~5점)과 `score_total(0~20)`,
    `score_total ≥ 14 → PASS` 규칙을 “표준 품질 판단의 헌법”으로 고정합니다.
- **관리자 모드(Admin SSOT + Admin API DoD)**는
  - KPI 대시보드(`/api/status`), 기사 목록/발행(`/api/items*`),
    보고서/Enrich(`/api/report`, `/api/enrich/*`), 로그/백업(`/api/logs*`, `/api/backup/status`),
    작업 플로우(`/api/tasks/*`), 게이트 설정(`/api/config/gate_required`) 등
    QualiJournal 운영자가 직접 사용하는 관리 기능 전체의 DoD를 정의합니다.
  - 응답 JSON·HTTP 코드·보안/IAM·Cloud SQL Stop Policy·DevOps 품질 게이트까지 포함한
    “관리자 모드 전체 운영 헌법”입니다.
- **domap(`quali-admin-domap`, admin.standardai.co.kr)**는
  - Cloud Run 상에서 도메인에 매핑된 **라이브 관리자 서비스**로,
    dev/SSOT 서비스에서 검증된 이미지를 가져와 배포하는 “운영 프록시” 역할을 합니다.
  - 표준 큐레이션 A안에서는 `/health`, `/api/status`, `/api/report`,
    `/api/standards/score-test` 4개 엔드포인트를 기준으로 Release DoD를 판단합니다.

### 0-2. 표준 품질 판단(4축 스코어)과 Admin API 스키마의 연결

- 표준 큐레이션 SSOT에서 정의한 4축 스코어는 다음과 같이 Admin API에 매핑됩니다.
  - 입력 모델: `StandardItem` 및 관련 Pydantic 모델
  - 핵심 엔드포인트: `POST /api/standards/score-test`
  - 응답은 Admin API DoD에서 정의한 공통 스키마
    `HTTP 200 + { ok: true, data: { ... } }` 패턴을 따르며,
    여기서 `data` 필드 안에 4축 점수·총점·decision(PASS/HOLD 등)이 포함됩니다.
- **결정(Decision) vs 검수 워크플로우(Status)** 는 다음 두 층으로 분리됩니다.
  - Decision 레벨: `decision ∈ { PASS, HOLD, BAD }`
  - ReviewTask 워크플로우 레벨: `status ∈ { HOLD, REVIEWED, PUBLISHED }`
- 흐름 요약:
  1. `/api/standards/score-test` 호출 → 4축 스코어 계산
  2. `score_total ≥ 14` → `decision=PASS`, ReviewTask 없음
  3. `score_total < 14` → `decision=HOLD`, ReviewTask 생성(`status=HOLD`, `required_reviewers=2`)
  4. `/api/standards/reviews/{id}/approve` 를 서로 다른 reviewer_id로 2회 호출 → `status=REVIEWED`
  5. `/api/standards/reviews/{id}/publish` → `status=PUBLISHED` (정책에 따라 decision PASS 승격)

### 0-3. 응답 JSON·HTTP 코드·보안 SSOT

- 성공 응답 기본 규칙(관리자 모드 공통):
  - HTTP 200 + `{ "ok": true, "data": { ...엔드포인트별 payload... } }`
  - 예외: SSE 스트림(`/api/tasks/{job_id}/stream`), 파일 다운로드(`/api/export/*`, `/api/logs/{name}/download`, `/api/archive/*`)
- 실패 응답 기본 규칙:
  - HTTP 4xx/5xx + `{ "ok": false, "error": "...", "error_code": "E.ADMIN.XXX", "detail"?:{...} }`
- 보안/IAM:
  - Cloud Run 레벨: private 서비스 + ID 토큰 기반 Invoker 보호
  - App 레벨: `ADMIN_TOKEN`/`API_TOKEN` 기반 `X-Admin-Token` 또는 `Authorization: Bearer <token>` 검사
  - 표준·큐레이션 관련 Admin API(`/api/standards/*`) 또한 동일한 토큰 정책을 따른다.

### 0-4. Cloud SQL Stop Policy와 표준 큐레이션 저장소 전략

- Cloud SQL 인스턴스는 기본 STOP 상태(`state=STOPPED`, `activationPolicy=NEVER`)를 유지하는 것이 SSOT입니다.
- 표준 큐레이션 A안의 2인 승인 큐(`standard_reviews.json`)는
  - `admin/logs/standard_reviews.json` 파일 기반 JSON 저장소를 사용하며,
  - Cloud SQL STOP 상태에서도 검수/승인 워크플로우가 동작하도록 설계되어 있습니다.
- 향후 부하/동시성이 증가하면 DB 테이블로의 마이그레이션 B안을 검토하되,
  - 필드 구조는 Curation SSOT + Admin DoD 스키마를 그대로 따르는 것을 기본 원칙으로 합니다.

### 0-5. DevOps 품질 게이트와 Release DoD 통합 관점

- Admin DoD v1.1에서 정의한 **DevOps 품질 게이트**:
  - `Admin Tests (pytest only)` 워크플로의 성공
  - main 브랜치 보호(서명 커밋, 선형 히스토리, 필수 status check)
- 표준 큐레이션 A안에서 정의한 **domap Release DoD**:
  - `GET /health` (ID 토큰) = 200
  - `GET /api/status` (ID + X-Admin-Token) = 200
  - `POST /api/report` (ID + X-Admin-Token) = 200
  - `POST /api/standards/score-test` (ID + X-Admin-Token) = 200
- 통합 헌법 v2에서의 Release 관점:
  1. dev(SSOT) 환경에서 Admin Tests + 주요 API 스키마/동작이 모두 PASS
  2. 동일 커밋/이미지로 domap에 배포
  3. domap에서 위 4개 스모크(표준 큐레이션 Release DoD)가 PASS
  4. Cloud SQL STOP Policy를 위반하지 않는 상태임을 확인
  5. 위 1~4를 만족해야 “표준 큐레이션 + Admin 모드 통합 릴리스”로 본다.

### 0.6 Quick Tools Gate (Admin / domap)

1. 목적
   - Quick Tools(자동 승인/복구 툴)의 동작을 **코드 / CI / 운영** 전 구간에서 일관되게 보장하고,
     기사 승인·동기화 품질을 최소 기준 이상으로 유지한다.

2. 대상 범위
   - Admin API 중 다음 엔드포인트를 Quick Tools Gate 대상으로 정의한다.
     - `POST /api/tools/approve_top`  (상위 N개 기사 일괄 승인)
     - `POST /api/tools/repair`       (선택 파일/발행본 구조 복구)
   - 두 엔드포인트는 공통 헬퍼 함수 `_tools_ok(rc, sync_log, stderr)`를 통해
     최종 성공 여부를 판정한다.

3. 성공 판정 규칙(SSOT)
   - Quick Tools Gate의 **유일한 성공 판정 규칙**은 다음과 같다.
   - 기본 성공:
     - `rc == 0`
     - 응답 JSON의 `ok == true`
   - B-Mode 예외(Quick Tools B-Mode, “스크립트 없음 + 싱크 성공” 시 성공으로 간주):
     - `rc == 127`
     - `sync_log.ok == true`
     - `stderr` (또는 내부 로그)에 `"not found"` 문자열이 포함되어 있음  
       (예: `"approve_top20.py not found"`, `"repair_selection_files.py not found"`).
   - 위 두 조건 중 하나를 만족할 때만:
     - `_tools_ok(...)` 가 `True` 를 반환하고
     - API 응답 JSON에서 `ok == true` 로 기록한다.
   - 그 외 조합(`rc != 0/127`, `sync_log.ok == false`, `"not found"` 미포함 등)은 모두 실패로 간주하고
     `ok == false` 를 반환한다.

4. 응답 스키마(요약)
   - Quick Tools 관련 응답 JSON은 다음 필드를 최소 포함한다.
     - `ok: bool`          – 위 성공 판정 규칙 결과
     - `rc: int`           – 내부 파이썬 스크립트 종료 코드
     - `stdout: str`       – 표준 출력(요약 로그)
     - `stderr: str`       – 표준 오류(에러/예외 메시지 포함)
     - `synced: bool`      – `sync_log.ok` 를 boolean으로 축약한 값
     - `sync_log: object`  – 선택/발행본 싱크 결과 전체
     - (`approve_top` 전용) `top: int` – 승인 대상 개수 N
   - 이 스키마는 Admin API 응답 SSOT의 일반 규칙(성공 시 `ok=true`, 실패 시 `ok=false`)을 따른다.

5. CI / 브랜치 보호 연계(Quick Tools B-Mode CI Gate)
   - Quick Tools Gate의 동작은 **전용 테스트 파일**로 검증한다.
     - 파일: `admin/tests/test_tools_quick.py`
     - 시나리오: 기본 성공 + B-Mode 성공(approve_top / repair) + 여러 실패 조합까지 총 6개 케이스(T1~T6).
   - GitHub Actions 워크플로우 `.github/workflows/ci-test-admin.yml` 의
     `Admin Tests (pytest only) / test-admin` 잡 안에
     `Run quick tools tests (pytest only)` 스텝을 고정한다.
   - 위 Admin Tests 체크는 `main` 브랜치의 **Required check** 로 유지한다.
     - `test_tools_quick.py` 중 하나라도 실패하면 `main` 으로의 머지가 차단된다.

6. 운영·문서 연계
   - Quick Tools Gate의 운영 관리는 별도 Runbook
     `QualiJournal_Admin_Traffic50_Runbook_A1_v1_20251121.md` 및
     `QualiJournal_Admin_Runbook_FINAL_2025-10-19.md` 의
     스모크/장애 대응 절을 따른다.
   - 헌법(SSOT)은 **규칙 정의**에 집중하고,
     구체 명령어와 실행 순서는 Runbook에서 관리한다.
### 0.7 Ready / Quick Tools / CI Gate 메타 요약

- 이 헌법에서 **운영 상태를 판정하는 관문(게이트)** 는 크게 세 층으로 나뉜다.  
  1) 컨테이너·라우팅 레벨 헬스 체크(`/health`),  
  2) 관리자 상태·준비도 레벨(`/api/status`, `/api/ready`),  
  3) 기능·배포 레벨 Quick Tools Gate + CI Gate + domap Traffic50 Runbook 이다.

- **Ready 계층(/health · /api/status · /api/ready)** 은  
  domap 서비스가 “게이트를 통과할 준비가 되었는지”를 확인하는 역할을 맡는다.  
  - `/health` 는 Cloud Run 리비전과 라우팅이 살아 있는지 보는 최소 헬스 체크이다.  
  - `/api/status` 는 `total`, `ready_count`, `gate_required`, `ready_rate`, `gate_pass` 등 KPI를 한 번에 보여준다.  
  - `/api/ready` 는 위 지표와 내부 점검 결과를 바탕으로, 정해진 Ready 스키마(JSON 필드·타입)를 따라  
    현재 배포가 Gate PASS 상태인지 요약한다.

- **Quick Tools Gate** 는 Admin API 중  
  `POST /api/tools/approve_top`, `POST /api/tools/repair` 두 엔드포인트에 대해  
  내부 스크립트 종료 코드(`rc`)와 싱크 결과(`sync_log`)를 조합해 성공/실패를 단일 규칙으로 판정한다.  
  - 기본 성공: `rc == 0` 이고 응답 JSON의 `ok == true`.  
  - B-Mode 성공: `rc == 127` 이고 `sync_log.ok == true` 이며, `stderr` 또는 로그에 `"not found"` 가 포함된다.  
  - 위 둘이 아닌 조합은 모두 실패(`ok == false`)로 본다.

- **CI Gate(Quick Tools B-Mode CI Gate)** 는  
  `admin/tests/test_tools_quick.py` 를 실행해 Quick Tools Gate의 기본 성공·B-Mode 성공·대표 실패 조합을 모두 검증하고,  
  GitHub Actions `Admin Tests (pytest only)` 잡 안의  
  `Run quick tools tests (pytest only)` 스텝이 실패하면 `main` 브랜치 머지를 차단한다.

- **운영 Runbook / SSOT Check 연계**  
  - domap 배포 시에는 Traffic50 Runbook A1을 따라  
    `/health`, `/api/status`, `/api/report`, `/api/standards/reviews` 의 200/404 여부로 최종 Go/No-Go 를 결정한다.  
  - `.github/workflows/ssot-check.yml` 은  
    SSOT 문서와 `ci-deploy-prod.yml` 이 이 헌법과 DoD(특히 `/api/status`, `/api/report`, `/api/standards/reviews` 스모크)를  
    깨지 않도록 자동으로 검사한다.

#### 0.7-1 Ready / Quick Tools / CI Gate 관계 표

| 레벨 | 이름 | 주 역할 | 성공 기준(요약 SSOT) | 구현·검증 위치 |
|---|---|---|---|---|
| L1 – 헬스 | 컨테이너 헬스 체크 | Cloud Run 리비전·라우팅 생존 여부 확인 | `GET /health` → `HTTP 200` | `server_quali.py` 헬스 엔드포인트, domap Runbook 스모크 |
| L2 – 상태 | 상태 대시보드 | 기사 수·Ready 수·게이트 기준 등 KPI 확인 | `GET /api/status` → `HTTP 200` + 공통 스키마(`ok=true, data{ total, ready_count, gate_required, ready_rate, gate_pass }`) | Admin 헌법 v2 / Admin DoD, `test_admin_api_schema.py` |
| L2.5 – Ready | Ready API | 현재 배포가 Gate PASS 상태인지 요약 | `/api/ready` 가 사전 정의된 Ready 스키마를 만족하고 성공 시 `ok=true` | `server_quali.py` Ready 엔드포인트, `admin/tests/test_admin_api_schema.py` |
| L3 – 기능 게이트 | Quick Tools Gate | approve_top / repair 결과의 성공/실패 판정 | `rc==0 & ok==true` 또는 `rc==127 & sync_log.ok==true & "not found"` 포함 | `_tools_ok(rc, sync_log, stderr)` 구현, Quick Tools 섹션(0.6) |
| L4 – CI Gate | Quick Tools B-Mode CI Gate | Quick Tools Gate 규칙이 깨지면 main 머지를 차단 | `admin/tests/test_tools_quick.py` 전체 통과, `Run quick tools tests (pytest only)` 스텝 성공 | `.github/workflows/ci-test-admin.yml`, 브랜치 보호 설정 |
| L5 – 운영 게이트 | domap Traffic50 A1 | 새 domap 리비전의 실제 운영 Go/No-Go 판단 | `/health`, `/api/status`, `/api/report`, `/api/standards/reviews` 스모크와 `/api/standards/reviews` 10회 호출이 모두 200 | `QualiJournal_Admin_Traffic50_Runbook_A1_v1_QuickTools_2025-11-23.md` |


---

## I장. 표준·기술 자료 큐레이션 + Admin(domap) 통합 헌법 v1 (원문)

# QualiJournal 표준·기술 자료 큐레이션 + Admin(domap) 통합 헌법 v1
(통합 기준일: 2025-11-19, 헌법 버전: 1.0)

이 문서는 다음 네 가지 문서를 하나로 통합한 **단일 진실(SSOT) 헌법**입니다.

1. 표준·기술 자료 큐레이션 SSOT 원문: `SSOT_QualiJournal_Curation_Standards_v.md`
2. QualiJournal Admin 인수인계 (2025-11-18): `QualiJournal_Admin_Handover_20251118_.md`
3. QualiJournal Admin 인수인계 1차 (2025-11-19): `QualiJournal_Admin_Handover_20251119_1.md`
4. QualiJournal Admin 인수인계(domap 표준/스코어링, 2025-11-19): `QualiJournal_Admin_Handover_20251119_2_domap_standards.md`

아래 **0장 ‘통합 요약 및 SSOT 고정’**은 위 네 문서를 기반으로
2025-11-19 현재 상태를 요약한 메타 섹션이며,
I~III장은 각 문서의 **원문을 수정 없이 그대로 포함**합니다.

---

## 0장. 통합 요약 및 SSOT 고정

### 0-1. 현재 상태 한 줄 요약

- 표준·기술 자료 큐레이션 A안(4축 스코어 + PASS/HOLD + 2인 검수 큐)은
  SSOT·로컬·dev 기준으로 스펙/코드/인수인계가 정렬된 상태입니다.
- Cloud Run domap(`quali-admin-domap`)에서는
  `/health`, `/api/status`, `/api/report`, `/api/standards/score-test`
  네 엔드포인트가 Release DoD 기준 PASS이며,
  `/api/standards/reviews*` 3종(목록/approve/publish)만 HOLD(404 조사 대상)입니다.
- 이 헌법 문서는 **큐레이션 SSOT + Admin 인수인계 3장을 한 번에 참조**하기 위한
  최상위 기준 문서입니다.

### 0-2. 표준 키(standard_key) SSOT

이번 표준 큐레이션 A안 전체를 대표하는 표준 키는 다음과 같이 고정합니다.

```json
{
  "name": "QualiJournal-Admin-Standards-API",
  "rev": "v1",
  "date": "2025-11-19"
}
```

모든 추가 문서·코드·배포는 이 standard_key를 기준으로
같은 기능 세트를 가리키도록 유지합니다.

### 0-3. 4축 스코어 및 판정 규칙 SSOT

- 스코어 축: `score_regular`(규격성), `score_applic`(적용성),
  `score_evid`(근거성), `score_trust`(신뢰성)
- 총점: `score_total = 네 축 합산 (0~20)`
- 판정 경계값: `score_total ≥ 14 → PASS`, `< 14 → HOLD`
- BAD는 별도 정책(명시적 불량 플래그)으로 관리합니다.
- PASS/HOLD/BAD는 **문서 품질(Decision)** 을 나타내며,
  2인 검수 워크플로우의 상태(ReviewTask.status)와는 구분합니다.

### 0-4. 2인 검수 워크플로우 SSOT

- ReviewTask.status: `HOLD → REVIEWED → PUBLISHED`
- 승인 인원: `required_reviewers = 2`
- 기본 흐름:

  1. `/api/standards/score-test` 로 4축 스코어 계산
  2. `score_total < 14` 이면 2인 검수 큐(ReviewTask) 생성
  3. `/api/standards/reviews/<built-in function id>/approve` 를
     **서로 다른 reviewer_id로 2회 이상** 호출 → `REVIEWED`
  4. `/api/standards/reviews/<built-in function id>/publish` 로 `PUBLISHED` 승격

- publish 시 Decision(PASS/HOLD/BAD)을 어떻게 업데이트할지는
  1판에서는 보수적으로 유지하고, 추후 정책 확정 시 SSOT와 함께 수정합니다.

### 0-5. Cloud Run domap Release DoD SSOT

표준·기술 자료 큐레이션 A안의 **릴리스 기준(Release DoD)** 은
다음 네 엔드포인트가 ID 토큰 + X-Admin-Token 조합에서 모두 200을 반환하는지로 판단합니다.

- `GET /health`
- `GET /api/status`
- `POST /api/report`
- `POST /api/standards/score-test`

위 네 개가 PASS이면 **“표준 큐레이션 A안 기본 백엔드 릴리스 완료”** 상태로 보며,
리뷰 플로우(`/api/standards/reviews*`)는 별도 이슈(HOLD)로 관리합니다.

---

## I장. 표준·기술 자료 큐레이션 SSOT 원문 (SSOT_QualiJournal_Curation_Standards_v.md)

# SSOT_QualiJournal_Curation_Standards_v1.md  
_QualiJournal 표준·기술 자료 큐레이션 운영 헌법 (v1)_

---

## 0. 문서 개요

### 0.1 목적

본 문서는 QualiJournal에서 다루는 **표준·기술 자료(IPC/ECSS/NASA/J-STD/자동차 등)** 의  
수집·선별·평가·발행 전 과정을 일관되게 운영하기 위한 **단일 진실(SSOT, Single Source of Truth)** 을 정의한다.

이 헌법은 다음을 목표로 한다.

- 표준·기술 자료 큐레이션의 **재현 가능성** 확보  
- 사람(Human-in-the-loop, HiTL)이 개입하는 **검증 지점 명시**  
- DevOps/DoD 체계와 연계되는 **품질 게이트 기준** 수립  
- 향후 자동화·API·관리자 모드(Admin) 기능과의 **일관된 스키마** 제공  

---

### 0.2 적용 범위

본 헌법은 다음 범위에 적용된다.

- QualiJournal가 다루는 다음 카테고리의 자료:
  - 전자 어셈블리 / 솔더링 / 케이블·하네스 / 우주·항공 전자장비
  - 자동차 전장품 / 고신뢰 전기·전자 어셈블리
  - 관련 국제·국가·협회 표준 및 기술 보고서, 가이드라인
- “표준·기술 자료 큐레이션 파이프라인”:
  - 수집 → 전처리·중복 제거 → 스코어링 → PASS/HOLD 결정 → 발행·아카이브  

뉴스, 일반 기사, 커뮤니티 글 등 **비표준·비기술 자료 큐레이션 규칙**은 별도 문서에서 정의하며,  
본 헌법은 표준·기술 자료를 최우선 대상으로 한다.

---

### 0.3 상위 및 관련 문서

- `SSOT_Admin_DoD_AdminAPI_v1.md`  
  - 관리자 모드(Admin Mode) API 및 운영 DoD 헌법  
- `QualiJournal_Admin_DoD_AdminAPI_20251116.md`  
  - 관리자 모드 DoD 세부 정의서  
- `1115_AdminMode_DoD_Security_IAM_SSOT_20251117.md`  
  - 보안·IAM 관련 SSOT  
- `QualiJournal Cloud SQL Stop Policy .pdf`  
  - Cloud SQL STOP 정책 (DB 없는 상태에서도 동작 가능한 구조 요구)  

본 문서는 위 문서들과 **동등한 위상의 SSOT**로서,  
“표준·기술 자료 큐레이션”에 관한 의사결정 시 최상위 기준으로 활용한다.

---

## 1. 정의 및 용어

### 1.1 표준(Standards)

- 국제·국가·협회 등에서 발행한 공식 규격·표준 문서
  - 예:  
    - IPC-A-610, IPC/WHMA-A-620, J-STD-001  
    - ECSS-Q-ST-70-61, ECSS-Q-ST-70-08  
    - NASA-STD-8739 시리즈 등

### 1.2 기술 자료(Technical Materials)

- 위 표준을 해설·적용·확장하는 다음 성격의 문서:
  - 공식 해설서, 가이드라인, FAQ
  - 학회·저널 논문, 기술 보고서(Technical Report)
  - 벤더 애플리케이션 노트, 교육기관 강의 자료 등

### 1.3 소스 티어(Source Tier)

자료의 출처에 따른 우선순위 계층을 다음과 같이 정의한다.

- **Tier 1 – 공식(Official)**  
  - 정부, 국제/국가 표준기구, 규격 발행처(IPC, ECSS, NASA 등)  
- **Tier 2 – 학술/협회(Academic/Association)**  
  - 학회, 협회, 피어리뷰 저널, 공식 컨퍼런스 프로시딩 등  
- **Tier 3 – 기타(Vendor/Community)**  
  - 벤더 애플리케이션 노트, 기술 블로그, 교육기관 자료, 커뮤니티 정리본 등  

소스 티어는 점수화 및 최종 순위 결정 시 **가중치**로 사용된다.

### 1.4 표준 키(standard_key)

동일·유사 표준 문서를 식별하기 위해 다음 3요소를 묶어 `standard_key`로 정의한다.

- `name`: 규격명(예: `"IPC-A-610"`, `"ECSS-Q-ST-70-61"`)  
- `rev`: 개정 번호/레벨(예: `"J"`, `"Rev. 1"`)  
- `date`: 발행일(또는 최신 개정일, ISO 8601 형식: `"2023-10-01"` 등)  

동일 `standard_key`를 가진 문서는 **동일 표준 문서**로 간주하며,  
중복 제거·버전 관리의 기준으로 삼는다.

### 1.5 결정 상태(Decision Status)

큐레이션된 각 문서는 아래 상태 중 하나를 가진다.

- `PASS`  
  - 최소 스코어 기준을 충족하며, 자동 승인 또는 편집자 확인 후 발행 가능  
- `HOLD`  
  - 스코어는 있으나 기준 미달 또는 추가 검토 필요  
  - 반드시 사람(큐레이터/편집자)이 검수해야 하는 상태  
- `BAD` (선택)  
  - 명백한 오정보·스팸 등으로 추후 활용 가치가 없다고 판단한 경우  

운영 초기에는 `PASS` / `HOLD` 2상태를 기본으로 하되, 필요 시 `BAD` 도입을 고려한다.

### 1.6 HiTL (Human-in-the-loop)

- 기계(자동 스코어링·필터링)가 아닌 **사람이 최종적으로 개입·결정**하는 단계.
- 본 헌법에서는 특히 다음 두 지점을 필수 HiTL 지점으로 정의한다.
  1. **HOLD 문서 검수**  
  2. **최종 발행 직전 검토(에디터 코멘트 작성 포함)**  

---

## 2. 기능 DoD – 큐레이션 프로세스

### 2.1 전체 플로우

표준·기술 자료 큐레이션 프로세스는 다음 5단계로 정의한다.

1. **수집(Collect)**  
2. **전처리 및 중복/유사중복 제거(Preprocess & Dedup)**  
3. **스코어링(Scoring)**  
4. **결정(PASS/HOLD 결정 및 HiTL 큐 분리)**  
5. **발행 및 아카이브(Publish & Archive)**  

각 단계는 아래 세부 DoD를 만족해야만 “완료”로 간주한다.

---

### 2.2 수집(Collect)

**DoD-C1. 소스 티어 라벨링**

- 수집된 모든 문서에 대해 `source_tier ∈ {official, association, vendor}` 값을 부여한다.
- 소스 도메인/기관 정보를 기반으로 자동 판정하되, 모호한 경우 HiTL 검토 대상에 포함한다.

**DoD-C2. 메타데이터 확보**

- 가능하면 다음 필드를 채운다.
  - `title`, `url`, `publisher`, `published_at`, `language`
  - 표준 문서의 경우 `standard_key{name,rev,date}`  
- 필수 필드:
  - `title`, `url`, `source_tier`  

---

### 2.3 전처리 및 중복/유사중복 제거

**DoD-D1. 해시 및 식별자 생성**

- 각 문서에 대해 다음 값을 계산/저장한다.
  - `url_hash`: 정규화된 URL의 해시
  - `title_fingerprint`: 제목 기반 n-gram 또는 토큰 시그니처
  - `standard_key`: 가능한 경우 설정

**DoD-D2. 완전 중복 제거 규칙**

- 아래 조건 중 하나라도 만족하면 **완전 중복**으로 판단하고 대표 1건만 남긴다.
  - `url_hash`가 동일  
  - `standard_key`가 완전히 동일(`name`, `rev`, `date` 모두 동일)

**DoD-D3. 유사 중복 제거 규칙**

- 아래 조건을 만족하는 문서 그룹은 **유사 중복 그룹**으로 묶는다.
  - 제목 유사도(예: cosine) ≥ 0.85 이상으로 추정  
  - `standard_key.name` 이 동일하고, `rev` 또는 `date`가 일부만 일치 또는 비어 있음
- 유사 중복 그룹 내 대표 선정 기준:
  1. `source_tier`: official > association > vendor  
  2. 발행일(`published_at`)이 더 최근인 문서  
  3. 위 2가지가 동일할 경우, 사람이 검토하여 대표 지정(HiTL)  

대표로 선택되지 않은 문서는 “중복 그룹 구성원”으로 로그에만 남기고  
실제 큐레이션 리스트에는 나타나지 않도록 한다.

---

### 2.4 스코어링(4축 평가)

각 문서에 대해 다음 4축을 **0~5점**으로 평가하고 합산하여 `score_total(0~20)`을 산출한다.

1. `score_regular` – **규격성(Regularity)**  
   - 정식 발행본·개정 이력·버전·발행일 표기 여부  
   - 5점: 공식 표준 PDF, 개정 정보 명확  
   - 0~2점: 블로그·요약글, 출처·날짜 불명확

2. `score_applic` – **적용성(Applicability)**  
   - QualiJournal의 핵심 영역(전자어셈블리, 솔더링, 우주/자동차 품질 등)과의 직결성  
   - 5점: 실무 적용·교육에 바로 활용 가능한 수준  
   - 0~2점: 주변 개념, 간접 관련

3. `score_evid` – **근거성(Evidence)**  
   - 원문 링크, 인용의 정확도, 참고 문헌의 풍부함  
   - 5점: 인용·참고문헌이 충분하고 직접 접근 가능  
   - 0~2점: 출처 미비, 2차·3차 출처 추정

4. `score_trust` – **신뢰성(Trust)**  
   - 도메인·기관의 신뢰도 (Tier 1/2/3 반영)  
   - 5점: 공식 표준기구, 정부, 국제기구  
   - 3~4점: 학회·협회·피어리뷰 저널  
   - 0~2점: 개인 블로그, 출처 불명확 자료

**DoD-S1. PASS/HOLD 경계값**

- 기본 경계값은 다음과 같이 정의한다.
  - `score_total ≥ 14`: `PASS` (자동 승인 후보)  
  - `score_total < 14`: `HOLD` (사람 검수 필요)  

경계값은 버전 업데이트 시 조정 가능하나,  
변경 시에는 본 문서의 마이너/메이저 버전 업데이트가 필요하다.

---

### 2.5 결정(PASS/HOLD 및 HiTL 큐 분리)

**DoD-J1. 결정 로직**

각 문서에 대해 다음 규칙으로 `decision`을 부여한다.

1. `score_total ≥ 14`  
   - 기본값: `PASS`  
   - 단, 중복/유사 중복 그룹 내에서 낮은 우선순위를 가진 문서는  
     대표 문서를 제외하고 `HOLD` 또는 비선정 처리 가능  
2. `score_total < 14`  
   - `HOLD`  
   - HiTL 검수 큐에 자동 등록  

**DoD-J2. 검수(HOLD) 큐 관리**

- 모든 `HOLD` 문서는 다음 정보를 반드시 포함한다.
  - `decision = "HOLD"`  
  - `reason_short`: 짧은 검수 사유(예: “출처는 좋으나 최신 개정 반영 여부 불명확”)  
- 검수는 **최소 1인 이상**이 수행하며, 향후 필요 시 “2인 교차 검토”로 강화 가능하다.
- 검수 결과:
  - `PASS` 로 승격 → 발행 가능  
  - `HOLD` 유지 → 추후 재검토 대상  
  - (선택) `BAD` → 제외 및 로그만 유지  

---

### 2.6 발행 및 아카이브(Publish & Archive)

**DoD-P1. 발행 단위**

- 발행 최소 단위는 “키워드·주제별 큐레이션 묶음”으로 한다.
  - 예:  
    - “IPC-A-610J Class 3 외관 검사 기준”  
    - “ECSS-Q-ST-70-61C 솔더링 품질 요구사항 개요”  

**DoD-P2. 발행 조건**

- 한 발행 묶음 안에 포함되는 문서는:
  - `PASS` 상태 문서만 기본 포함  
  - 필요 시, `HOLD` 문서를 “참고용(검수 필요)” 태그로 별도 구역에 표시  
- 각 묶음에는 **편집자(에디터)의 한 줄 코멘트**를 포함한다.
  - 예: “실제 교육용 자료로는 1~3번 문서를 우선 추천, 4번은 연구·추가 참고용.”

**DoD-P3. 아카이브 및 복원**

- 발행 결과(큐레이션 리스트)는 **머신 리더블 형식(JSON/YAML)** 과  
  **사람이 읽기 쉬운 마크다운/HTML** 두 형태로 보존한다.
- Cloud SQL STOP Policy를 고려하여,
  - DB가 중지된 상태에서도 파일 기반(예: `archive/standards/YYYYMMDD_*.json`)으로  
    조회·복원이 가능해야 한다.

---

## 3. 비기능 DoD – 데이터 구조, 로그, 저작권, 용어

### 3.1 공통 데이터 스키마

큐레이션 결과는 아래 필드 세트를 기본으로 한다.

```yaml
id: <string>                # 내부 식별자 (예: URL 해시 기반)
title: <string>             # 문서 제목
url: <string>               # 원문 접근 URL
source_tier: official | association | vendor

standard_key:
  name: <string | null>     # 예: "IPC-A-610"
  rev: <string | null>      # 예: "J", "Rev.1"
  date: <string | null>     # ISO 8601, 예: "2023-10-01"

score_regular: 0-5
score_applic: 0-5
score_evid: 0-5
score_trust: 0-5
score_total: 0-20

decision: PASS | HOLD | BAD
reason_short: <string>      # 결정/검수 사유 요약 (한 줄)

meta:
  publisher: <string | null>
  published_at: <string | null>    # 발행일
  language: <string | null>        # 예: "en", "ko"
  tags:                            # 키워드 리스트
    - <string>

log:
  created_at: <string>      # 최초 수집 일시
  updated_at: <string>      # 마지막 업데이트 일시
  created_by: system | human
  updated_by: system | human

---

## II장. 인수인계 보고서 원문 모음

### II-A. QualiJournal Admin 인수인계 (2025-11-18, QualiJournal_Admin_Handover_20251118_.md)

---
service: QualiJournal Admin
task_name: 표준 큐레이션 4축 스코어러 & 2인 승인 워크플로우 A안 적용
start_date: 2025-11-18
current_stage: Test
stages: [Plan, Build, Test, Release, Operate]
progress:
  overall: 70
  development: 75
  testing: 65
  documentation: 80
---

# QualiJournal Admin 인수인계 보고서 (2025-11-18)

## 1. Executive Summary

- **Go/No-Go 판단**: **조건부 Go (로컬 기준)**  
- **핵심 사유 3개**
  1. **4축 스코어러(규격성/적용성/근거성/신뢰성) + PASS 하한선 14점 로직이 성공적으로 구현·로컬 검증됨.**
  2. **HOLD → REVIEWED → PUBLISHED로 이어지는 2인 승인 워크플로우 A안(JSON 파일 기반)이 엔드포인트 수준에서 완전히 동작함.**
  3. **보안 정책(ADMIN_TOKEN 기반 X-Admin-Token 헤더 검증)을 준수한 상태에서 새로운 API들이 정상 응답하는 것을 PowerShell 스모크 테스트로 확인함.**

> 단, Cloud Run 배포 및 관리자 UI 연동은 아직 미완료이므로, 운영 환경 기준으로는 **“Test 단계 완료, Release/Operate 대기”** 상태로 본다.

---

## 2. 메타 & 진행 현황

### 2.1 작업 메타

- **작업명(Task Name)**: 표준 큐레이션 4축 스코어러 & 2인 승인 워크플로우 A안 적용  
- **작업 시작일(Start Date)**: 2025-11-18  
- **현재 단계(Current Stage)**: **Test**  
- **작업 단계(Stage 목록)**: [Plan, Build, Test, Release, Operate]  
- **담당자/연락처(Owner/Contact)**: 닥터 윤 / (내부 채널 & 직접 협업)

### 2.2 단계표

| 단계 | 상태 | 메모 | 예정 종료 |
|---|---|---|---|
| Plan | ✅ 완료 | 4축 스코어 & 승인 워크플로우 A안(JSON) 구조 합의 | 2025-11-18 |
| Build | ✅ 완료 | `curation_scoring.py`, `standard_review_store.py`, Admin API 엔드포인트 구현 | 2025-11-18 |
| Test | 🔘 진행중 | 로컬 PowerShell 스모크 완료, UI/Cloud Run 연동 테스트는 미완 | 2025-11-20 (추정) |
| Release | ☐ 미시작 | Cloud Run 배포 및 domap 서비스에 반영 필요 | - |
| Operate | ☐ 미시작 | 관리자 UI 버튼/플로우 연결 후 실제 표준 큐레이션 운영 | - |

### 2.3 진척도 요약

진척도는 **기능/비기능/운영** DoD 상태와 실제 테스트 범위를 기준으로 추정했다.

| 지표 | % | 바(20칸) | 근거 |
|---|---:|---|---|
| Overall | 70 | ██████████████□□□□□□ | 기능 구현 75%, 테스트 65%, 문서화 80% 기준 평균(추정) |
| 개발 | 75 | ███████████████□□□□ | 4축 스코어러 + 승인 API 4종 + 저장소 유틸 완성 |
| 테스트 | 65 | █████████████□□□□□ | 로컬 API 스모크 & 승인 플로우 전 구간 검증, UI/배포 테스트 미실시 |
| 문서화 | 80 | ████████████████□□ | 본 인수인계 보고서 및 SSOT 반영 정도 기준 (추정) |

> 테스트/문서화 비율은 DoD 스코어보드가 아니기 때문에 **“추측:” 성격의 추정치**임.

---

## 3. DoD 상태 요약

### 3.1 기능 DoD

| 영역 | 상태 | 설명 |
|---|---|---|
| 4축 스코어러(표준·기술 자료) | 충족 | `curation_scoring.py`의 `StandardItem`, `compute_scores`, `score_and_maybe_enqueue_for_review` 구현 완료 및 PASS/HOLD/하한선 14점 로직 검증 |
| 2인 승인 워크플로우 A안 | 충족 | `ReviewTask` dataclass + JSON 기반 저장소(`standard_review_store.py`) + `/reviews`/`/approve`/`/publish` 3종 엔드포인트 구현 |
| Admin API 연동 | 충족 | `admin/server_quali.py` 상단 import, Pydantic 모델(`StandardScoreRequest`, `ReviewApproveReq`), `POST /api/standards/score-test` 추가 |
| UI 연동(표준 탭/버튼) | 미충족 | 아직 프론트엔드 버튼 및 워크플로우와 연결되지 않음 |
| Cloud Run 배포 반영 | 미충족 | 현재 로컬 서버 기준만 검증, domap 서비스 배포·테스트는 다음 단계 |

### 3.2 비기능 DoD

| 영역 | 상태 | 설명 |
|---|---|---|
| 보안/IAM | 부분충족 | ADMIN_TOKEN 기반 `X-Admin-Token` 헤더 검증 하에 새 API 동작 확인. 단, 역할 기반 세분화/로그인 연동은 아직 범위 밖 |
| 로그/추적 | 부분충족 | `logs/standard_reviews.json` 에 승인 큐가 남도록 구현. 다만 중앙 로그/메트릭 시스템과의 통합은 미진 |
| 데이터 스키마 일관성 | 부분충족 | `StandardItem` 구조와 Admin 응답 패턴(`_ok`) 일관성 유지. DB 테이블 연동은 미완료 |
| 성능/부하 | 미충족 | 파일 기반 JSON 저장소로 충분하지만, 부하·동시성 테스트는 미실시 |

### 3.3 운영·환경 DoD

| 영역 | 상태 | 설명 |
|---|---|---|
| 로컬 개발 환경 | 충족 | PowerShell + Python 로컬 실행 및 테스트 성공 |
| Cloud Run 환경 | 미충족 | 아직 새 엔드포인트로 배포·스모크 테스트를 수행하지 않음 |
| Cloud SQL STOP Policy 영향 | 부분충족 | 승인 큐는 파일(JSON) 기반이라 STOP 정책과 충돌 없음. 향후 DB 테이블 도입 시 정책 재검토 필요 |

---

## 4. 우선순위 백로그

### 4.1 Backlog 요약 표

| 우선순위 | 항목 | DoD 정의 | 검증 스텝 | 예상 리드타임 |
|---|---|---|---|---|
| Blocker | Cloud Run 배포 & domap 서비스에 새 API 반영 | `/api/standards/*` 엔드포인트가 prod/domap에서 200/401 정상 응답 | curl 또는 PowerShell로 401→토큰→200 흐름 확인 | 0.5~1일 |
| Blocker | 관리자 UI(표준 탭)에 리뷰/승인/발행 버튼 연결 | HOLD 문서가 UI에서 조회/승인/발행 가능 | `GET /reviews`, `POST /approve`, `POST /publish` 호출 로그 + UI 버튼 클릭 스모크 | 1~2일 |
| High | `StandardItem` ↔ DB 테이블 매핑(선택) | 표준 문서 메타/점수/상태가 DB에서도 조회·갱신 가능 | sqlite 또는 Cloud SQL에서 SELECT/INSERT/UPDATE 스모크 | 1~2일 |
| High | 표준 큐레이션 SSOT와 실제 응답 필드 맵핑 재검토 | `SSOT_QualiJournal_Curation_Standards_v1.md` 의 필드와 API 응답 필드 일치 | SSOT vs `score-test` 응답 diff 확인 | 0.5일 |
| Normal | 리뷰 큐 파일 회전/백업 정책 | `standard_reviews.json` 용량/기록 보존 정책 수립 | 일정 기간/건수 초과 시 백업 파일 생성 스크립트 검증 | 0.5일 |

---

## 5. 결정사항 & 근거

### 5.1 설계/정책 결정

1. **4축 스코어러 구조**
   - 축: `규격성(score_regular)`, `적용성(score_applic)`, `근거성(score_evid)`, `신뢰성(score_trust)`
   - 총점: `score_total = 4축 합산`
   - 하한선: `PASS_THRESHOLD = 14`

2. **결정 상태(Decision & Workflow)**  
   - `decision`: `PASS` 또는 `HOLD`  
   - `workflow_status`: `HOLD` → `REVIEWED` → `PUBLISHED`

3. **HOLD 조건 & 2인 검수**
   - `score_total < 14` 인 경우 **자동 HOLD + ReviewTask 큐 등록**
   - `required_reviewers = 2`
   - 서로 다른 `reviewer_id` 두 명이 승인 시 `status = REVIEWED`

4. **승인 워크플로우 A안 저장소 선택**
   - A안: **JSON 파일 저장소(`admin/logs/standard_reviews.json`)**
   - 이유: Cloud SQL STOP 정책과 무관하게 표준 승인 큐 운영 가능, 초기 구현 비용 최소화

### 5.2 코어 엔드포인트 목록

- `POST /api/standards/score-test`  
  - 표준 문서 1건에 대해 4축 스코어 계산 + HOLD 시 리뷰 큐에 등록  
- `GET /api/standards/reviews`  
  - JSON 저장소에 있는 ReviewTask 목록 조회  
- `POST /api/standards/reviews/{standard_id}/approve`  
  - 특정 표준에 대한 승인(검수자 1인) 처리  
- `POST /api/standards/reviews/{standard_id}/publish`  
  - REVIEWED 상태의 표준을 PUBLISHED 로 승격

> 모든 엔드포인트는 `X-Admin-Token: <ADMIN_TOKEN>` 헤더를 요구함.

---

## 6. 테스트 결과

### 6.1 스코어러 PASS 케이스 (IPC-A-610J)

#### 요청 예시 (PowerShell)

```powershell
$body = @{
  id = "ipc-a-610j-official-2023"
  title = "IPC-A-610J: Acceptability of Electronic Assemblies"
  url = "https://www.ipc.org/standard/ipc-a-610j"
  source_tier = "official"
  standard_name = "IPC-A-610"
  standard_rev = "J"
  standard_date = "2023-10-01"
  meta_publisher = "IPC"
  meta_published_at = "2023-10-01"
  meta_language = "en"
  tags = @("IPC-A-610","Class 3","전자어셈블리")
  target_keywords = @("IPC-A-610","Class 3","전자어셈블리")
} | ConvertTo-Json -Depth 5

$token = "***ADMIN_TOKEN***"

Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/score-test" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $body
```

#### 주요 응답 요약

- `ok: True`
- `name: "standards_score_test"`  
- `pass_threshold: 14`
- `item.score_total: 20`
- `item.decision: "PASS"`  
- `item.workflow_status: "REVIEWED"`  
- `review_task: null` (큐 등록 없음)

### 6.2 스코어러 HOLD + 리뷰 큐 케이스 (dummy-vendor-001)

#### 요청 예시

```powershell
$body = @{
  id = "dummy-vendor-001"
  title = "Some blog about soldering"
  url = "https://example.com/blog-soldering"
  source_tier = "vendor"
  standard_name = ""
  standard_rev = ""
  standard_date = ""
  meta_publisher = ""
  meta_published_at = ""
  meta_language = "en"
  tags = @("blog","random")
  target_keywords = @("soldering")
} | ConvertTo-Json -Depth 5

$token = "***ADMIN_TOKEN***"

Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/score-test" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $body
```

#### 주요 응답 요약

- `score_regular=1`, `score_applic=3`, `score_evid=3`, `score_trust=3` → `score_total=10`
- `decision: "HOLD"`  
- `workflow_status: "HOLD"`  
- `review_task.standard_id: "dummy-vendor-001"`  
- `review_task.status: "HOLD"`  
- `review_task.required_reviewers: 2`  
- `review_task.approved_by: []`

### 6.3 승인 플로우 테스트

1. **리뷰 큐 조회**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews" `
  -Headers @{ "X-Admin-Token" = $token }
```

- `items[0].standard_id = "dummy-vendor-001"`  
- `status = "HOLD"`, `approved_by = []`

2. **1차 승인(editor_a)**

```powershell
$approveBody = @{ reviewer_id = "editor_a" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews/dummy-vendor-001/approve" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $approveBody
```

- 응답: `status = "HOLD"`, `approved_by = ["editor_a"]`

3. **2차 승인(editor_b) → REVIEWED**

```powershell
$approveBody2 = @{ reviewer_id = "editor_b" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews/dummy-vendor-001/approve" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $approveBody2
```

- 응답: `status = "REVIEWED"`, `approved_by = ["editor_a","editor_b"]`

4. **발행(PUBLISHED)**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews/dummy-vendor-001/publish" `
  -Method POST `
  -Headers @{ "X-Admin-Token" = $token }
```

- 응답: `status = "PUBLISHED"`

---

## 7. 운영 스냅샷

> 이번 세션은 **로컬 개발 환경** 기준이며, Cloud Run/도메인(domap) 환경은 다음 단계에서 다룰 예정.

- **서버 실행**:  
  - 명령: `python server_quali.py`  
  - 바인딩: `http://0.0.0.0:8080`  
- **보안**:  
  - `.env` 의 `ADMIN_TOKEN` 값 기반으로 `X-Admin-Token` 헤더 검증  
  - 민감값은 보고서 상에서 `***ADMIN_TOKEN***` 로 마스킹
- **로그/파일**:
  - 승인 큐 파일: `admin/logs/standard_reviews.json`  
  - IPC-A-610J PASS/HOLD 테스트 등은 서버 로그와 PowerShell 출력으로 확인
- **Cloud SQL STOP 상태**:  
  - 이번 워크플로우는 DB 의존 없으며, STOP 정책에 영향 없음(파일 기반).  

---

## 8. 리스크 & 완화책

| 리스크 | 내용 | 발생 조건 | 완화책 |
|---|---|---|---|
| 파일 기반 저장소의 동시성 문제 | 다중 인스턴스/동시 요청 시 JSON 파일 충돌 위험 | Cloud Run에서 동시 인스턴스가 표준 승인 API를 자주 호출할 때 | 초기에는 승인 요청 빈도가 낮다고 가정하고 사용, 추후 DB 테이블 기반으로 마이그레이션 |
| 리뷰 큐 용량 증가 | `standard_reviews.json` 파일이 계속 커질 수 있음 | 장기간 운영 시 승인 이력 누적 | 주기적 백업/회전 전략 수립(예: 월 단위 스냅샷 후 파일 리셋) |
| SSOT와 실제 응답 필드 불일치 | Curation SSOT, Admin DoD 문서와 실제 API 응답이 달라질 위험 | 향후 스키마 변경 시 문서 업데이트 누락 | SSOT 문서와 API 스키마 변경 시 항상 PR 리뷰 체크리스트에 포함 |
| Cloud Run 배포 시 미스매치 | 로컬 테스트 코드는 OK이나 prod 환경 설정 누락 가능 | env/토큰/네트워크 설정 차이 | `deploy-admin` 파이프라인에서 /health + `/api/standards/score-test` 스모크 추가 |

---

## 9. 다음 단계(Owners/Due)

| No | 항목 | Owner | Due(예상) | 완료 정의(DoD) |
|---|---|---|---|---|
| 1 | Cloud Run domap 서비스에 새 API 배포 | 닥터 윤 + DevOps | 2025-11-20 | `/api/standards/score-test`/`/reviews`/`/approve`/`/publish` 가 domap에서 200/401 정상 응답 |
| 2 | 관리자 UI(표준 탭)에 리뷰/승인/발행 버튼 연결 | FE/풀스택 | 2025-11-25 | HOLD 문서가 UI에서 조회·승인·발행까지 대화형으로 동작 |
| 3 | Curation SSOT ↔ API 스키마 정합성 점검 | PM/아키텍트 | 2025-11-22 | `SSOT_QualiJournal_Curation_Standards_v1.md` 와 실제 응답 필드 매핑표 작성 |
| 4 | 리뷰 큐 백업/회전 스크립트 초안 작성 | 백엔드 | 2025-11-27 | 일정 조건(건수/용량) 초과 시 백업 파일 생성 + 원본 축소 스크립트 동작 |
| 5 | Cloud Run 배포 테스트 케이스 정의 | QA | 2025-11-23 | `/health` + `/api/standards/*` 스모크 시나리오 목록 작성 및 문서화 |

---

## 10. 🔖 Carry-Over / Next Task

다음 채팅에서 바로 이어서 착수하면 좋은 **Top 3 과제**는 아래와 같다.

| 순위 | Task | DoD 기준 | 테스트 방법 |
|---|---|---|---|
| 1 | Cloud Run domap 에 새 API 배포 및 스모크 | `/api/standards/*` 4개 엔드포인트가 401→토큰→200 플로우 정상 작동 | PowerShell/curl 로 실제 도메인 호출, 로그에서 응답 코드·latency 확인 |
| 2 | 관리자 UI 표준 탭과 승인 워크플로우 연결 | HOLD 문서 카드에 “리뷰/승인/발행” 버튼이 나타나고, 클릭 시 각각 API 호출 및 상태 갱신 | 로컬 프론트엔드에서 버튼 클릭 → 백엔드 로그 및 JSON 응답 확인 |
| 3 | SSOT_Curation v1 + Admin DoD 간 스키마 정렬 | Curation SSOT 필드와 `StandardScoreRequest`/응답 구조가 표로 정리되어 불일치 없는 상태 | SSOT 문서와 코드(스키마/엔드포인트)를 나란히 비교하는 리뷰 세션 수행 |

---

### 다음 채팅 시작용 추천 프롬프트

```text
오늘은 QualiJournal Admin 표준 큐레이션 워크플로우를 Cloud Run domap 환경에 배포하고, 
/health + /api/standards/* 스모크 테스트를 통해 Release 단계 DoD를 충족하는 것부터 같이 보자.
```


### II-B. QualiJournal Admin 인수인계 (2025-11-19 1차, QualiJournal_Admin_Handover_20251119_1.md)

---
service: QualiJournal Admin
task_name: 표준 큐레이션 4축 스코어러 & 2인 승인 워크플로우 A안 – SSOT 정렬 세션
date: 2025-11-19
stage: Test
stages: [Plan, Build, Test, Release, Operate]
---

# QualiJournal Admin 인수인계 요약 (2025-11-19)

## 1. 오늘 다룬 이슈 요약

- **이슈 제목**  
  - 표준 큐레이션 4축 스코어러 & 2인 승인 워크플로우 A안에 대해,  
    `SSOT_QualiJournal_Curation_Standards_v1`와 `QualiJournal_Admin_Handover_20251118_.md` 기준으로  
    **스코어링 규칙·결정 로직·검수 상태머신·Admin API 스키마·리뷰 큐 운영 룰**을 1:1로 정렬.

- **목표**  
  1. Curation SSOT v1에서 정의한 4축 스코어(규격성/적용성/근거성/신뢰성)와 PASS/HOLD(≥14점) 규칙을,  
     Admin의 `StandardItem` + `score-test` 구현과 완전히 일치시킬 것.  
  2. SSOT의 `PASS/HOLD/BAD` 개념과 Admin의 `HOLD → REVIEWED → PUBLISHED` 2인 승인 워크플로우를  
     하나의 상태머신으로 정리할 것.  
  3. `/api/standards/score-test`, `/reviews`, `/approve`, `/publish` 4개 엔드포인트의  
     요청/응답 스키마를 SSOT 스키마와 매핑된 “기준 스펙”으로 정리할 것.  
  4. `admin/logs/standard_reviews.json` 파일 기반 A안에 대해,  
     Cloud SQL STOP Policy와 충돌하지 않는 운영 룰(백업/회전/동시성 가정)을 명문화할 것.

- **오늘 실제로 한 일 (성공)**  
  1. **4축 스코어 & PASS/HOLD 규칙 정렬**  
     - SSOT: 4축(0~5점) 합산으로 `score_total(0~20)` 산출, `score_total ≥ 14 → PASS`, `<14 → HOLD`.  
     - Admin: `score_regular/applic/evid/trust`, `score_total = 합산`, `PASS_THRESHOLD=14`,  
       `decision: PASS/HOLD` 로 구현되어 있음을 상호 확인.  
  2. **Decision vs ReviewTask 상태머신 통합**  
     - Decision: `PASS/HOLD/BAD` (문서 품질 판단).  
     - ReviewTask.status: `HOLD → REVIEWED → PUBLISHED` (2인 승인 워크플로우).  
     - 흐름: `score-test`로 HOLD 발생 시 ReviewTask 생성 → 2인 `approve`로 REVIEWED → `publish`로 PUBLISHED,  
       발행 시 PASS 승격을 권장하는 정책 초안까지 합의.  
  3. **4개 Admin API 스키마 정리**  
     - 공통 모델: `StandardItem`, `ReviewTask`, 공통 응답 래퍼(`ok`, `name`, `pass_threshold`, `item`, `items`, `review_task`).  
     - 각 엔드포인트별 요청/응답 JSON 예제를 정리해, SSOT 필드와 매핑이 가능한 상태로 정리.  
  4. **`standard_reviews.json` 운영 룰 초안 확정**  
     - 위치: `admin/logs/standard_reviews.json`.  
     - 역할: HOLD 문서의 검수 진행 상황(2인 승인 상태)을 담는 임시 큐.  
     - 운영 원칙: 초기에는 파일 기반 + 저QPS 전제, 월 단위(또는 용량/건수 기준) 스냅샷 후 회전,  
       Cloud SQL STOP 상태에서도 승인 큐가 동작하도록 유지.

- **남은 부분 / 추후 확인 필요(🟡)**  
  - `publish` 시점에 `decision`을 반드시 `PASS` 로 승격할지,  
    혹은 `published_decision` 같은 별도 필드로 들고 갈지 **정책 확정 필요(🟡)**.  
  - `reason_short`, `log.*` 필드를 코드 레벨에서 **필수 컬럼으로 강제할지**,  
    일부 엔드포인트에서만 노출할지에 대한 구현 디테일은 추후 코드/테스트에서 확인 필요(🟡).  
  - Cloud Run domap 배포 및 관리자 UI(표준 탭) 연동은 여전히 **Release/Operate 단계 TODO**.

---

## 2. SSOT 업데이트 (오늘 세션 기준 정리본)

### 2.1 4축 스코어링 & PASS/HOLD 규칙

- 4축 이름과 의미는 SSOT와 Admin 양쪽에서 다음으로 고정한다.  
  - `score_regular` – 규격성(정식 발행본, 개정 이력, 발행일 등)  
  - `score_applic` – 적용성(Quali 핵심 도메인과의 직접 관련성)  
  - `score_evid` – 근거성(원문 링크, 인용, 참고문헌)  
  - `score_trust` – 신뢰성(소스 티어, 기관 신뢰도)  
- 총점과 경계값:  
  - `score_total = score_regular + score_applic + score_evid + score_trust` (0~20점).  
  - 기본 경계값: `score_total ≥ 14 → PASS`, `<14 → HOLD`.  
  - 이 값(`14`)을 바꾸려면 **Curation SSOT 버전 업데이트 + Admin 인수인계 동시 수정**이 필요하다.

### 2.2 Decision vs ReviewTask 상태머신

- **Decision 레벨(문서 품질 판단)**  
  - `decision ∈ {PASS, HOLD, BAD}`  
  - HOLD 문서는 모두 `reason_short` 를 필수로 갖는다.  
- **ReviewTask 레벨(검수 워크플로우 상태)**  
  - `status ∈ {HOLD, REVIEWED, PUBLISHED}`  
  - `required_reviewers = 2`, `approved_by = [reviewer_id…]`.  
- **통합 흐름**  
  1. `score-test` 호출 → `score_total ≥ 14` 이면 `decision=PASS`, ReviewTask 없음.  
  2. `score_total < 14` 이면 `decision=HOLD`, ReviewTask 생성(`status=HOLD`).  
  3. 서로 다른 reviewer 두 명의 `/approve` 호출로 `status=REVIEWED`.  
  4. `/publish` 호출로 `status=PUBLISHED`, 이때 `decision` 을 PASS로 승격하는 것을 권장(정책 옵션).  

### 2.3 Admin API 스키마 SSOT

- 공통 모델
  - `StandardItem`:  
    - `id`, `title`, `url`, `source_tier`, `standard_key{name,rev,date}`,  
      `score_regular/applic/evid/trust`, `score_total`, `decision`,  
      `meta{publisher,published_at,language,tags[]}`, `log{created_at,updated_at,created_by,updated_by}`.  
  - `ReviewTask`:  
    - `standard_id`, `decision`, `status(HOLD/REVIEWED/PUBLISHED)`,  
      `required_reviewers`, `approved_by[]`, `reason_short`, `log{…}`.  
- 응답 래퍼 패턴(standards 계열 공통):  
  - `ok: bool`  
  - `name: string` (엔드포인트 식별자)  
  - `pass_threshold?: int` (`score-test` 전용, 기본 14)  
  - `item?: StandardItem` / `items?: ReviewTask[]`  
  - `review_task?: ReviewTask`  
- 엔드포인트별 역할  
  - `POST /api/standards/score-test` : 4축 스코어 계산 + HOLD 시 ReviewTask 생성.  
  - `GET  /api/standards/reviews` : ReviewTask 목록 조회.  
  - `POST /api/standards/reviews/{standard_id}/approve` : reviewer 1인의 승인.  
  - `POST /api/standards/reviews/{standard_id}/publish` : REVIEWED → PUBLISHED 승격.  
  - 모든 엔드포인트는 `X-Admin-Token: <ADMIN_TOKEN>` 필수.

### 2.4 `standard_reviews.json` 파일 운영 룰(A안)

- 위치: `admin/logs/standard_reviews.json`.  
- 역할: HOLD 문서의 검수 진행 상황을 담는 **승인 큐(임시 저장소)**.  
- Cloud SQL STOP Policy와의 관계:  
  - 승인 큐는 DB에 의존하지 않고 파일만으로 동작 → STOP 상태와 충돌 없음.  
  - 향후 DB 테이블로 옮길 때도 필드 구조는 그대로 가져간다.  
- 백업/회전(rotate) 전략(초안):  
  - 기준: 월 단위 또는 건수/용량 기준으로 회전.  
  - 방법:  
    - 회전 시 현 파일을 `standard_reviews_YYYYMM.json` 으로 복사(백업).  
    - `status="PUBLISHED"` 및 오래된 `REVIEWED` 는 백업으로 옮기고,  
      아직 미해결인 `HOLD` 만 새 파일에 남긴다.  
- 동시성 가정:  
  - 초기에는 승인 API QPS가 낮고, 운영자가 제한된 시간대에 수동 승인한다는 전제를 둔다.  
  - 대량 승인/자동화가 필요해지면 **DB 테이블로의 마이그레이션**을 B안으로 준비한다.

---

## 3. 복붙용 명령/코드 모음 (로컬 스모크 기준)

> 전제: `python server_quali.py` 로 로컬에서 8080 포트에 서버가 떠 있고,  
> `.env` 의 `ADMIN_TOKEN` 값이 `***ADMIN_TOKEN***` 로 치환되어 있다고 가정.

### 3.1 PASS 케이스 – IPC-A-610J 스코어 테스트

```powershell
$body = @{
  id               = "ipc-a-610j-official-2023"
  title            = "IPC-A-610J: Acceptability of Electronic Assemblies"
  url              = "https://www.ipc.org/standard/ipc-a-610j"
  source_tier      = "official"
  standard_name    = "IPC-A-610"
  standard_rev     = "J"
  standard_date    = "2023-10-01"
  meta_publisher   = "IPC"
  meta_published_at= "2023-10-01"
  meta_language    = "en"
  tags             = @("IPC-A-610","Class 3","전자어셈블리")
  target_keywords  = @("IPC-A-610","Class 3","전자어셈블리")
} | ConvertTo-Json -Depth 5

$token = "***ADMIN_TOKEN***"

Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/score-test" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $body
```

### 3.2 HOLD 케이스 – 벤더 블로그 dummy-vendor-001

```powershell
$body = @{
  id               = "dummy-vendor-001"
  title            = "Some blog about soldering"
  url              = "https://example.com/blog-soldering"
  source_tier      = "vendor"
  standard_name    = ""
  standard_rev     = ""
  standard_date    = ""
  meta_publisher   = ""
  meta_published_at= ""
  meta_language    = "en"
  tags             = @("blog","random")
  target_keywords  = @("soldering")
} | ConvertTo-Json -Depth 5

$token = "***ADMIN_TOKEN***"

Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/score-test" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $body
```

### 3.3 리뷰 큐 조회 → 승인 → 발행 플로우

```powershell
$token = "***ADMIN_TOKEN***"

# 1) 리뷰 큐 조회
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews" `
  -Headers @{ "X-Admin-Token" = $token }

# 2) 1차 승인 (editor_a)
$approveBody1 = @{ reviewer_id = "editor_a" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews/dummy-vendor-001/approve" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $approveBody1

# 3) 2차 승인 (editor_b)
$approveBody2 = @{ reviewer_id = "editor_b" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews/dummy-vendor-001/approve" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-Admin-Token" = $token } -Body $approveBody2

# 4) 발행(PUBLISH)
Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/standards/reviews/dummy-vendor-001/publish" `
  -Method POST `
  -Headers @{ "X-Admin-Token" = $token }
```

---

## 4. 다음 세션 TODO

1. Cloud Run domap 에 `/api/standards/*` 4개 엔드포인트 배포 및 스모크.
2. 관리자 UI(표준 탭)의 HOLD 카드에 리뷰/승인/발행 버튼 연결.
3. Curation SSOT v1 ↔ Admin Pydantic/응답 스키마 최종 매핑표 작성.
4. `standard_reviews.json` 회전/백업 스크립트 초안 작성.
5. 배포 파이프라인용 `/health` + `/api/standards/*` 테스트 케이스 정의.

---

## 5. 다음 채팅 시작용 한 줄 프롬프트

```text
오늘은 QualiJournal Admin 표준 큐레이션 워크플로우를 Cloud Run domap 환경에 배포하고, /health랑 /api/standards/* 4개 엔드포인트 스모크 테스트로 Release 단계 DoD부터 같이 맞추자.
```


### II-C. QualiJournal Admin 인수인계 (2025-11-19 domap 표준/스코어링, QualiJournal_Admin_Handover_20251119_2_domap_standards.md)

# QualiJournal Admin – domap 표준 큐레이션 워크플로우 인수인계 보고서  
(Release DoD 관점 / Cloud Run domap 환경 기준)

---

## 0. 메타 정보

- **작업명(Task Name)**: QualiJournal Admin – domap 표준 큐레이션 워크플로우 Release DoD 점검  
- **작업 시작일(Start Date)**: 2025-11-19 (Asia/Seoul)  
- **작업 단계(Stage 목록)**: [Plan, Build, Test, Release, Operate]  
- **현재 단계(Current Stage)**: **Release** (표준 큐레이션 핵심 API 스모크 완료, 리뷰 플로우 일부 HOLD)  
- **작업 완성률(주관적)**:  
  - Release 핵심(health/status/report/score-test): **100%**  
  - Review Flow(reviews/approve/publish): **~40%** (스펙·코드 정합, domap 배포/엔드포인트 미정렬)  

---

## 1. 오늘 세션 작업 요약

1. **목표 설정**  
   - Cloud Run `quali-admin-domap` 서비스에서  
     - `/health`  
     - `/api/status`  
     - `/api/report`  
     - `/api/standards/score-test`  
     4개 엔드포인트를 기준으로 **표준 큐레이션 워크플로우 Release DoD(배포 확인용 스모크)** 를 맞추는 것이 1차 목표.  
   - 추가로 `/api/standards/reviews` 및 `approve/publish` 리뷰 플로우의 상태를 확인해 **PASS/HOLD를 분류**하는 것이 2차 목표.

2. **핵심 결론**  
   - **Cloud Run 관문(ID 토큰)** 과 **Admin 앱 관문(X-Admin-Token)** 을 모두 통과하는 방식으로 스모크를 수행했고, 결과는 다음과 같다.
   - `quali-admin-domap` 기준  
     - `/health`: **200 OK (ID 토큰만)** → Cloud Run 보호 상태에서 서비스 정상 동작 확인.  
     - `/api/status`: **200 OK (ID + X-Admin-Token)** → Admin 상태 점검 엔드포인트 정상.  
     - `/api/report`: **200 OK (ID + X-Admin-Token)** → 일일 리포트 API 정상, Cloud Scheduler 연동 전제 충족.  
     - `/api/standards/score-test`: **200 OK (ID + X-Admin-Token)** → 표준 큐레이션 4축 스코어 계산/큐 등록 로직 정상.  
     - `/api/standards/reviews`: **404 Not Found** → domap에 리뷰 목록 라우트가 아직 포함되지 않은 버전으로 배포된 상태로 추정.  
   - 따라서 **표준 큐레이션 핵심(health/status/report/score-test)은 Release DoD 기준 PASS**, 리뷰 플로우는 **HOLD(추가 배포/정렬 필요)** 로 분류.

3. **인증 구조 재정리(두 개의 문 비유)**  
   - **바깥 문: Cloud Run ID 토큰**  
     - `Authorization: Bearer <ID_TOKEN>` (출처: `gcloud auth print-identity-token`)  
     - 이 문을 통과해야 `/health` 에 도달할 수 있음.  
   - **안쪽 문: Admin 토큰(X-Admin-Token)**  
     - `X-Admin-Token: <ADMIN_TOKEN>` (출처: Secret Manager `ADMIN_TOKEN:latest`)  
     - 이 문을 추가로 통과해야 `/api/status`, `/api/report`, `/api/standards/*` 등 Admin API 접근 가능.  
   - 오늘 스모크는 **두 문을 모두 올바른 열쇠로 열고**, 각 엔드포인트가 기대한 HTTP 코드를 반환하는지 확인하는 과정이었다.

---

## 2. 시스템 구조 · 인증 흐름 정리

### 2.1 Cloud Run 서비스 구조

- **dev(SSOT)**  
  - 서비스: `quali-journal-admin-dev`  
  - 리전: `asia-northeast3` (서울)  
  - env:  
    - `PYTHONUNBUFFERED=1`  
    - `PYTHONUTF8=1`  
    - `ADMIN_TOKEN` → Secret `ADMIN_TOKEN:latest`  
  - `COMMIT_SHA` env는 **아직 없음** (배포 추적 보조 기능은 domap 쪽이 더 풍부).  

- **domap(도메인 프록시)**  
  - 서비스: `quali-admin-domap`  
  - 리전: `asia-northeast1` (도쿄)  
  - 주요 env:  
    - `PYTHONUNBUFFERED=1`  
    - `PYTHONUTF8=1`  
    - `ADMIN_TOKEN` → Secret `ADMIN_TOKEN:latest`  
    - `COMMIT_SHA="<커밋 해시>"` (배포 추적용)  
    - `ALLOWED_ORIGINS=https://admin.standardai.co.kr`  
  - domap는 **도메인 매핑된 실사용 관리자 UI/API 엔드포인트**로, 오늘 스모크의 주 무대.

### 2.2 서버(App) 인증 로직 (`server_quali.py` 기준)

- 공통 의존성: `authorize()`
  - env에서 **`ADMIN_TOKEN`, `API_TOKEN`** 후보를 읽음.  
  - 둘 다 비어 있으면 **open mode (인증 없이 통과)**.  
  - 값이 있으면 아래 조건 중 하나라도 맞아야 통과:  
    1. `X-Admin-Token: <ADMIN_TOKEN 또는 API_TOKEN>`  
    2. `Authorization: Bearer <ADMIN_TOKEN 또는 API_TOKEN>`  
  - 둘 다 아니면 **HTTP 401 Unauthorized** 반환.  
- 라우트 의존성
  - `/health`: **인증 없이** 항상 200 반환 (단, Cloud Run 레벨 보호는 별도).  
  - `/api/status`, `/api/report`, `/api/standards/*`: 전부 `Depends(authorize)` → Admin 토큰 필수.

### 2.3 Cloud Run/IAM 보호층

- 서비스 자체는 **“Require authentication”** 상태로,  
  - ID 토큰 없이 접근하면 Cloud Run 프록시에서 **403 Forbidden** 반환.  
- 오늘 스모크는 모두  
  - `Authorization: Bearer <ID_TOKEN>` (Cloud Run 문)  
  - `X-Admin-Token: <ADMIN_TOKEN>` (App 문)  
  조합으로 수행.

---

## 3. 엔드포인트별 Release DoD 결과 표

### 3.1 요약표 (4축 스코어 + PASS/HOLD)

`standard_key = { name: "QualiJournal-Admin-Standards-API", rev: "v1", date: "2025-11-19" }`

> 4축 스코어 정의  
> - **규격성(score_regular)**: SSOT 스펙·라우트 정의와의 정합성  
> - **적용성(score_applic)**: domap 환경에서 실제 운영에 바로 활용 가능한지  
> - **근거성(score_evid)**: 코드, 문서, 로그 등 근거 수준  
> - **신뢰성(score_trust)**: HTTP 코드 안정성, 인증/보안 구조 신뢰도  
> - 총점: `score_total = regular + applic + evid + trust (0~20)`  
> - **판정**: `score_total ≥ 14 → PASS`, 그 외는 HOLD

| Rank | Endpoint                               | Auth 방식                     | score_regular | score_applic | score_evid | score_trust | score_total | Decision | 비고 |
|------|----------------------------------------|------------------------------|---------------|--------------|------------|-------------|-------------|----------|------|
| 1    | `GET /health`                          | ID 토큰만                    | 4             | 5            | 4          | 4           | **17**      | **PASS** | Cloud Run 보호 상태에서 200 확인, 배포 헬스 기준 충족 |
| 2    | `GET /api/status`                      | ID + X-Admin-Token           | 4             | 5            | 4          | 4           | **17**      | **PASS** | Admin 상태 점검 API, domap에서도 200 |
| 3    | `POST /api/report`                     | ID + X-Admin-Token           | 4             | 5            | 4          | 4           | **17**      | **PASS** | 일일 리포트 API, Cloud Scheduler 연동 가능 상태 |
| 4    | `POST /api/standards/score-test`       | ID + X-Admin-Token           | 5             | 5            | 5          | 4           | **19**      | **PASS** | 표준 4축 스코어 계산 + HOLD 시 ReviewTask 생성까지 정상 |
| 5    | `GET /api/standards/reviews`           | ID + X-Admin-Token           | 3             | 1            | 3          | 2           | **9**       | **HOLD** | domap에서 404 → 라우트 미배포/경로 차이 의심 |
| 6    | `POST /api/standards/reviews/{id}/approve` | ID + X-Admin-Token       | 3             | 1            | 2          | 1           | **7**       | **HOLD** | dev 코드·SSOT 상 존재, domap은 `/reviews` 404라 미검증 |
| 7    | `POST /api/standards/reviews/{id}/publish` | ID + X-Admin-Token       | 3             | 1            | 2          | 1           | **7**       | **HOLD** | approve와 동일 – ReviewFlow 전체가 domap에선 미배포 추정 |

---

## 4. 검수 큐(HOLD 목록) 및 향후 액션

### 4.1 HOLD-1: `/api/standards/reviews` 404 이슈

- **현상**
  - domap에서 `GET /api/standards/reviews` 호출 시 **HTTP 404 Not Found**.  
  - 이는 “서버는 살아 있지만, 이 URL에 해당하는 리소스를 찾지 못했다”는 의미 → **엔드포인트 미배포/경로 불일치** 쪽 가능성 높음.
- **가능 원인**
  1. domap 서비스가 **리뷰 플로우가 포함되기 이전 커밋**을 사용 중인 경우  
  2. dev와 domap이 서로 다른 브랜치/이미지로 배포된 경우  
  3. 실제 FastAPI 라우트 경로가 `/api/standards/review`(단수) 혹은 `/api/v1/standards/reviews` 등으로 차이가 있는 경우
- **검수/수정 제안**
  - dev 기준 `server_quali.py` 라우트 정의(현재 SSOT)와 domap 배포 이미지의 버전을 대응시키기.  
  - 필요 시, domap 배포에 사용되는 GitHub Actions 워크플로우에서 **dev와 동일 커밋/이미지**를 참조하도록 조정.  
  - domap 상에서 `/docs`(Swagger) 혹은 `/openapi.json` 을 확인하여 **실제 라우트 경로**를 비교.

### 4.2 HOLD-2: Review Approve/Publish 플로우 미검증

- **현상**
  - `/api/standards/reviews` 가 404라서, 테스트용 `standard_id`를 얻지 못해 approve/publish 스모크를 수행하지 못함.  
- **dev 기준 기대 동작**
  - HOLD 문서 → ReviewTask 생성(2인 승인 필요)  
  - `approve` 1회: `approved_by`에 리뷰어1 추가, status 유지 또는 PARTIAL 상태  
  - `approve` 2회: `approved_by`에 리뷰어2 추가, status → `REVIEWED`  
  - `publish`: `REVIEWED` 상태에서만 `PUBLISHED` 승격 + 최종 decision PASS 확정  
- **domap에서의 검증 순서(향후)**  
  1. `/api/standards/score-test` 로 HOLD 문서를 하나 넣어 리뷰 큐 생성.  
  2. `/api/standards/reviews` 200 + `count ≥ 1` 확인.  
  3. 선택된 `standard_id` 에 대해 `approve` ×2 → `publish` → 상태/HTTP 코드 확인.

### 4.3 HOLD-3: dev env COMMIT_SHA 미설정 (배포 추적 메타)

- **현상**
  - domap env에는 `COMMIT_SHA` 가 있으나, dev env에는 없음.  
- **영향**
  - 기능에는 영향 없음.  
  - “어느 커밋이 dev/도메인에 배포되었는지” 를 Cloud Run 설정만으로 바로 비교하기 어렵다.  
- **개선 아이디어(장기)**  
  - dev/배포 워크플로우에서 `COMMIT_SHA=$GITHUB_SHA` 같은 형태로 env를 추가해,  
    - `gcloud run services describe ...` 한 번으로 배포 커밋을 추적 가능하게 만들 것.

---

## 5. 스모크 재현 절차 (PowerShell 스크립트)

### 5.1 domap 핵심 4종 엔드포인트 스모크

```powershell
# ===== 공통 설정 =====
$PROJECT = "quali-journal-prod"
$REGION  = "asia-northeast1"
$SERVICE = "quali-admin-domap"

# 1) Cloud Run 서비스 URL
$URL = gcloud run services describe $SERVICE `
  --region $REGION --project $PROJECT `
  --format "value(status.url)"

Write-Host "Cloud Run URL = $URL" -ForegroundColor Cyan

# 2) Cloud Run ID 토큰 (계정 신분증)
$ID_TOKEN = gcloud auth print-identity-token
Write-Host "ID_TOKEN length = $($ID_TOKEN.Length)" -ForegroundColor DarkGray

# 3) ADMIN_TOKEN (Secret Manager)
$ADMIN_TOKEN = (gcloud secrets versions access latest `
  --secret=ADMIN_TOKEN `
  --project $PROJECT | Out-String).Trim()
Write-Host "ADMIN_TOKEN length = $($ADMIN_TOKEN.Length)" -ForegroundColor DarkGray

# [H1] /health (ID 토큰만)
Write-Host "`n[H1] /health (ID 토큰만)" -ForegroundColor Yellow
curl.exe -s -o NUL -w "%{http_code} health-id`n" `
  -H "Authorization: Bearer $ID_TOKEN" `
  "$URL/health"

# [H2] /api/status (ID + X-Admin-Token)
Write-Host "`n[H2] /api/status (ID + X-Admin-Token)" -ForegroundColor Yellow
curl.exe -s -o NUL -w "%{http_code} status-auth`n" `
  -H "Authorization: Bearer $ID_TOKEN" `
  -H "X-Admin-Token: $ADMIN_TOKEN" `
  "$URL/api/status"

# [H3] /api/report (ID + X-Admin-Token, POST)
Write-Host "`n[H3] /api/report (ID + X-Admin-Token, POST)" -ForegroundColor Yellow
curl.exe -s -o NUL -w "%{http_code} report-auth`n" `
  -H "Authorization: Bearer $ID_TOKEN" `
  -H "X-Admin-Token: $ADMIN_TOKEN" `
  -X POST "$URL/api/report" `
  -H "Content-Length: 0"

# [H4] /api/standards/score-test (ID + X-Admin-Token, POST)
Write-Host "`n[H4] /api/standards/score-test (ID + X-Admin-Token, POST)" -ForegroundColor Yellow
curl.exe -s -o NUL -w "%{http_code} score-test-auth`n" `
  -H "Authorization: Bearer $ID_TOKEN" `
  -H "X-Admin-Token: $ADMIN_TOKEN" `
  -X POST "$URL/api/standards/score-test" `
  -H "Content-Type: application/json" `
  --data (@"
{
  "title": "Cloud Run domap 표준 큐레이션 테스트",
  "url": "https://admin.standardai.co.kr/",
  "source_tier": "official",
  "standard_name": "QualiJournal-Admin-Standards-API",
  "standard_rev": "v1",
  "standard_date": "2025-11-19",
  "meta_publisher": "QualiJournal Admin",
  "meta_published_at": "2025-11-19",
  "meta_language": "ko",
  "tags": ["cloud-run","standards","admin","domap"],
  "target_keywords": ["Cloud Run","standards","admin","domap"]
}
"@)
```

### 5.2 리뷰 플로우 스모크(향후 재사용용 템플릿)

```powershell
# 전제: 위에서 $URL, $ID_TOKEN, $ADMIN_TOKEN 이미 설정됨

Write-Host "`n[R1] /api/standards/reviews (GET, ID + X-Admin-Token)" -ForegroundColor Yellow

$REVIEWS_JSON = curl.exe -s `
  -H "Authorization: Bearer $ID_TOKEN" `
  -H "X-Admin-Token: $ADMIN_TOKEN" `
  "$URL/api/standards/reviews"

$code = curl.exe -s -o NUL -w "%{http_code}" `
  -H "Authorization: Bearer $ID_TOKEN" `
  -H "X-Admin-Token: $ADMIN_TOKEN" `
  "$URL/api/standards/reviews"

Write-Host "HTTP status = $code (reviews)" -ForegroundColor Green

$REVIEWS_JSON | Set-Content -Path "reviews.json" -Encoding UTF8
Write-Host ">> reviews.json 파일에 검수 큐 내용 저장" -ForegroundColor Cyan

$REVIEWS_OBJ = $REVIEWS_JSON | ConvertFrom-Json
Write-Host ("count = " + $REVIEWS_OBJ.count)
$REVIEWS_OBJ.items | Select-Object -First 5 standard_id, status, required_reviewers, approved_by

if ($REVIEWS_OBJ.count -eq 0) {
    Write-Host "`n[R2] 검수 큐가 비어 있어서 approve/publish 스모크는 건너뜁니다." -ForegroundColor Yellow
}
else {
    $STANDARD_ID = $REVIEWS_OBJ.items[0].standard_id
    Write-Host "`n[R2] 테스트에 사용할 standard_id = $STANDARD_ID" -ForegroundColor Yellow

    # approve/publish 호출 템플릿 (필요 시 주석 해제)
    # curl.exe ... "$URL/api/standards/reviews/$STANDARD_ID/approve"
    # curl.exe ... "$URL/api/standards/reviews/$STANDARD_ID/publish"
}
```

---

## 6. 정리 · 결론

1. **표준 큐레이션 핵심 4종(health/status/report/score-test)** 은  
   - Cloud Run domap 환경에서 ID 토큰 + ADMIN_TOKEN 조합으로 **정상 PASS**.  
   - Release 단계 DoD 상, “표준 큐레이션 워크플로우 A안”의 핵심 백엔드는 실서비스 기준으로 합격 상태.

2. **리뷰 플로우(reviews/approve/publish)** 는  
   - dev/SSOT 기준으로는 구현·스펙이 정렬되어 있으나,  
   - domap에서 `/reviews` 가 404라 **“아직 domap에 탑재되지 않은 기능”** 으로 보는 것이 안전.  
   - 차후 배포/커밋 기준을 정리한 뒤, 동일 커밋을 domap에 재배포하고 다시 스모크할 필요가 있음.

3. **배포 추적 메타(COMMIT_SHA)**  
   - domap에는 존재, dev에는 없음 → 기능에는 영향 없으나,  
   - 장기적으로 dev에도 `COMMIT_SHA` 를 넣어 두면 배포/롤백 추적이 쉬워진다.

---

## 7. 다음 세션을 위한 추천 한 줄 프롬프트

> **“찌빠, 오늘 정리한 인수인계 보고서 기준으로, domap에서 `/api/standards/reviews` 404가 나는 원인을 dev·배포 파이프라인·서버 라우트 관점에서 한 번에 정리해 보자. 1단계는 dev와 domap의 server_quali.py 라우트 정의 diff 비교부터 시작하자.”**


---

## II장. QualiJournal 관리자 모드 SSOT (Admin DoD v1.1) 원문


# QualiJournal 관리자 모드 SSOT (Admin DoD v1.1)

> **Single Source of Truth (SSOT)**  
> 이 문서는 QualiJournal 관리자 모드(Admin)의 개발·테스트·배포·운영을 위한 단일 기준(Single Source of Truth)이다.  
> 관리자 모드 관련 모든 논의·결정·작업은 아래 내용을 기본 전제로 한다.

---

## 0. 문서 개요

### 0.1 목적

- 관리자 모드의 **개발 완료 기준(DoD, Definition of Done)** 을 한 곳에 모아 정리한다.
- 기능, 비기능, 운영/환경, DevOps, Cloud SQL 정책, PR 체크리스트를 **하나의 기준선**으로 통합한다.
- “관리자 모드 인크리먼트가 Done인가?”를 판단할 때 이 문서를 가장 먼저 참고한다.

### 0.2 Ground Truth 문서

이 SSOT는 다음 다섯 문서를 종합·요약한 것이다.

1. **「1115 관리자 모드 개발 완료 기준 정의서」**  fileciteturn1file0  
2. **「DoD(Definition of Done) 충족 상태 점검 요약서」**  fileciteturn1file1  
3. **「QualiJournal Cloud SQL Stop Policy」**  fileciteturn1file2  
4. **「QualiJournal_DoD_런북_인수인계_v1.1__2025-11-15.md」**  fileciteturn1file3  
5. **「QualiJournal_PR_Template_DevOps_Checklist_v1_2025-11-15_1.md」**  fileciteturn1file4  

원문이 변경될 경우, 이 SSOT를 함께 업데이트해야 한다.

---

## 1. 기능 DoD 체크리스트 (요약)

관리자 모드는 다음 기능이 “정의서 기준”으로 동작해야 **기능 DoD를 충족**한 것으로 본다. fileciteturn1file0  

### 1.1 KPI 및 상태 대시보드 (/api/status)

- **설명**: 전체 기사 수, 승인/대기 수, 게이트 기준치, 게이트 통과 여부 등을 한눈에 보여주는 관리용 대시보드.
- **완료 기준**
  - 관리자 UI 대시보드와 `/api/status` 응답 값이 서로 일관된다.
  - 최소 필드: `total`, `ready_count`, `gate_required`, `ready_rate`, `gate_pass` 등.
- **검증 방법**
  - 테스트 데이터(예: 기사 100건 중 80건 승인)를 준비한다.
  - `/api/status` 응답에서 위 필드 값이 기대값과 일치하는지 확인한다.
  - UI 대시보드 숫자와 API 응답을 교차 검증한다.

### 1.2 비동기 플로우 실행 및 로그 확인 (/api/flow/*, /api/tasks/flow)

- **설명**: 뉴스 수집, 일일 플로우, 키워드 플로우 등 백그라운드 작업을 수동으로 실행하고, 결과 로그를 조회하는 기능. fileciteturn1file0turn1file1  
- **완료 기준**
  - 운영자가 일일/키워드 플로우를 버튼 또는 API로 실행할 수 있다.
  - 플로우 실행 시, 대응되는 로그 파일이 `/api/logs` 목록에 새로 생긴다.
- **검증 방법**
  - `/api/flow/daily` 또는 `/api/tasks/flow?type=daily` 를 호출한다.
  - 작업 완료 후 `/api/logs` 를 호출해 신규 로그가 생성되었는지 확인한다.
  - 필요 시 `/api/tasks/{job_id}/stream` SSE로 진행 로그를 확인한다.

### 1.3 기사 목록 확인 및 승인/거절/코멘트

- **설명**: 수집된 기사 목록을 확인하고, 승인/거절 처리 및 편집자 코멘트를 저장하는 기능. fileciteturn1file0  
- **완료 기준**
  - 대기 중(ready 전) 기사 목록을 UI와 API(`/api/items`, `/api/community`)에서 조회할 수 있다.
  - 각 기사에 대해 승인/거절이 가능하며, 승인 시 상태가 “준비(ready)”로 변경된다.
  - 코멘트 입력 시 데이터에 저장된다.
- **검증 방법**
  - 테스트용 기사 몇 개를 준비하고 승인/거절 버튼을 눌러본다.
  - 처리 후 상태 필드와 코멘트 저장 여부를 API 응답 또는 데이터 저장소에서 확인한다.

### 1.4 발행 및 게이트 설정 (/api/items/{id}/publish, /api/config/gate_required)

- **설명**: 승인된 기사를 실제 사용자 서비스에 발행하고, 발행 최소 기준(게이트)을 설정·조정하는 기능. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/api/items/{id}/publish` 호출 시 해당 기사가 발행 상태로 변경되고, 사용자 화면에 노출된다.
  - `/api/config/gate_required` 값 변경 시 `/api/status` 의 `gate_required`가 즉시 갱신된다.
- **검증 방법**
  - 특정 기사에 대해 발행 API를 호출한 뒤 사용자 화면 노출 여부를 확인한다.
  - 게이트 값을 10→15 등으로 변경하고 `/api/status` 응답에서 값이 즉시 반영되는지 본다.

### 1.5 보고서 생성 및 결과물 내보내기 (/api/report, /api/export/*)

- **설명**: 특정 날짜/키워드에 대한 요약 보고서를 생성하고, MD/CSV 파일로 다운로드하는 기능. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/api/report` 호출 시 `ok=true`와 함께 보고서 파일 경로가 반환된다.
  - `/api/export/md`, `/api/export/csv` 로 해당 파일을 실제로 다운로드할 수 있다.
- **검증 방법**
  - 1일치 보고서를 생성하고 반환된 `path` 를 사용해 export API를 호출한다.
  - 다운로드된 MD/CSV 내용이 비어 있지 않은지 확인한다.

### 1.6 로그 및 백업 상태 확인 (/api/logs, /api/backup/status)

- **설명**: 시스템 로그와 백업 상태를 조회하고, 최근 성공/실패 여부를 모니터링한다. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/api/logs` 에서 로그 파일 목록이 JSON으로 조회된다.
  - `/api/logs/{name}`, `/api/logs/{name}/download` 로 개별 로그 열람·다운로드가 가능하다.
  - `/api/backup/status` 에서 최근 백업 성공/실패 시각과 메시지가 노출된다.
- **검증 방법**
  - `/api/logs` 응답에 대표 로그 파일 이름이 포함되는지 확인한다.
  - 그 중 하나를 열어 최신 내용이 출력되는지 본다.
  - `/api/backup/status` 응답에 마지막 백업 성과 시간이 올바르게 표시되는지 확인한다.

### 1.7 헬스체크 및 디버그용 API (/health, /api/debug/*)

- **설명**: 서비스 상태를 점검하는 헬스체크와, 운영 편의를 위한 디버그 정보 제공용 API. fileciteturn1file0turn1file1  
- **완료 기준**
  - `/health` 는 항상 200 OK 와 단순한 “ok” 메시지를 반환한다.
  - `/api/debug/*` 계열 엔드포인트는 런타임/설정/메트릭 등의 JSON 정보를 제공한다(운영/개발 환경에서만 사용).
- **검증 방법**
  - `/health` 를 여러 번 호출해 200 응답을 확인한다.
  - ` /api/debug/runtime`, `/api/debug/config` 등을 호출해 내부 정보가 JSON으로 반환되는지 확인한다.

---

## 2. 비기능 DoD 체크리스트 (요약)

### 2.1 응답 JSON 구조 및 HTTP 상태 코드 일관성

- **요구사항** fileciteturn1file0turn1file1  
  - 성공: HTTP 200 + `{ ok: true, data: {...} }`
  - 실패: HTTP 4xx/5xx + `{ ok: false, error, error_code, detail? }`
  - 모든 엔드포인트가 정의된 스키마(필드명·타입)를 공유해야 한다.
- **검증**
  - 대표 API 여러 개에 대해 성공/실패 케이스를 만들어 응답 구조와 상태 코드를 비교한다.
  - Pydantic `BaseModel` 등을 활용해 응답 스키마를 코드로도 강제한다.

### 2.2 인증/권한 안전성

- **요구사항** fileciteturn1file0turn1file3  
  - 관리자 모드 API는 환경 변수 `ADMIN_TOKEN`(또는 동등한 비밀 값) 기반으로 보호한다.
  - 민감 API는 **유효한 토큰 헤더**(예: `Authorization: Bearer <token>`, `X-Admin-Token`)가 있어야만 처리한다.
  - Cloud Run 서비스는 Invoker 권한이 특정 서비스 계정 등으로 제한된 **private 서비스**를 기본으로 한다.
- **검증**
  - 토큰 없이 호출 시 401/403 이 반환되는지 확인한다.
  - 올바른 토큰으로 호출 시 200 응답이 나오는지 확인한다.
  - Cloud Run 콘솔에서 Invoker 권한이 특정 주체에만 부여되어 있는지 점검한다.

### 2.3 보고서 및 플로우 성능

- **요구사항** fileciteturn1file0turn1file1  
  - 일반적인 기사량(수십 건 기준)에서 보고서 생성은 수 초 내에 끝나야 한다.
  - 일일/키워드 플로우 등 일괄 작업도 Cloud Run 타임아웃(기본 15분, 상한 60분) 내에서 완료되어야 한다.
  - 메모리 사용량은 설정 한도(예: 512MiB)를 넘지 않아야 한다.
- **검증**
  - 테스트 데이터 50~200건 정도를 준비해 `/api/report`, `/api/flow/daily` 처리 시간과 메모리를 모니터링한다.
  - Cloud Run 로그/모니터링 대시보드에서 타임아웃·OOM(No) 여부를 확인한다.

### 2.4 유지보수 가능성

- **요구사항** fileciteturn1file0turn1file1  
  - 데이터 모델과 요청/응답 스키마가 Pydantic 모델로 일관되게 선언되어 있어야 한다.
  - 상수·구조는 중복 없이 “한 곳에서만 정의”하는 단일 소스 원칙을 따른다.
  - 핵심 기능에 대한 자동화 테스트가 존재해 리팩토링 시 회귀를 빠르게 발견할 수 있어야 한다.
- **검증**
  - 새로운 필드를 추가해도 모델 한 곳만 수정하면 전체 테스트가 통과하는지 확인한다.
  - 동일 상수가 여러 파일에 하드코딩돼 있지 않은지 코드 리뷰로 점검한다.

---

## 3. 운영/환경 DoD 체크리스트 (요약)

### 3.1 Cloud Run 환경 제약 준수

- **요구사항** fileciteturn1file0turn1file1  
  - 모든 요청 처리가 Cloud Run 최대 타임아웃 이내에 끝나야 한다.
  - 메모리 사용량이 설정 한도를 넘지 않아야 한다.
  - 부하가 걸려도 인스턴스가 타임아웃/OOM으로 자주 죽지 않도록 설계한다.
- **검증**
  - 대용량 시나리오(기사 200건 수준)를 실행해 가장 긴 요청 시간이 타임아웃보다 충분히 짧은지 확인한다.
  - 모니터링에서 CPU/메모리 그래프를 확인해 여유를 확인한다.

### 3.2 배포 및 IAM 구성

- **요구사항** fileciteturn1file0turn1file3  
  - GitHub Actions 등 CI/CD 파이프라인이 main 브랜치 변경을 자동으로 Cloud Run에 배포한다.
  - 배포 후 최신 리비전에 100% 트래픽이 연결되어 있어야 한다.
  - Cloud Run Invoker 권한은 배포용/운영용 서비스 계정에만 부여한다.
  - `ADMIN_TOKEN` 등 비밀은 Secret Manager를 통해 환경 변수로 안전하게 주입한다.
- **검증**
  - 테스트 브랜치에서 PR/머지 후 새로운 리비전이 생성되고 트래픽이 정상 전환되는지 확인한다.
  - Cloud Run IAM 탭에서 Invoker 범위를 점검한다.
  - 환경 변수 화면에서 비밀 값이 정상 주입되었는지 확인한다.

---

## 4. DoD 충족 상태 점검 요약 (런북 v1.1 기반)

다음 내용은 **DoD(Definition of Done) 충족 상태 점검 요약서 + 인수인계 런북 v1.1** 를 바탕으로 한 “실행용 요약”이다. fileciteturn1file1turn1file3  

### 4.1 단계별 실행 계획(Plan – 프롬프트 체이닝 관점)

0단계. **기준 정합화**
- 입력: 정의서, Q&A, 현재 Cloud Run 리비전/트래픽/도메인, ADMIN_TOKEN 위치.
- 산출물: 엔드포인트·스키마·HTTP 코드·권한 맵, 현재 배포 스냅샷.
- 스모크: `/health`, `/api/status(토큰有/無)` 확인.

1단계. **API 명세·스키마 정합화**
- 목표: 모든 주요 API가 공통 응답 구조와 HTTP 코드 규약을 따른다.
- 작업: Pydantic 응답 모델 정의, 에러 코드 테이블 작성, 엔드포인트별 스펙 표준화.

2단계. **보안/IAM 하드닝**
- 목표: 토큰+Cloud Run IAM 2중 방어.
- 작업: 익명 접근 차단, Invoker 최소화, Scheduler/SA 권한 정리, 토큰 회전 절차.

3단계. **로그·백업·플로우 모니터링**
- 목표: `/api/logs`, `/api/backup/status`, `/api/tasks/flow`가 연계된 관제 체계를 갖춘다.
- 작업: 백업 상태 응답 형식 표준화, 로그/다운로드 케이스 정리, 플로우 실행→로그 연결 확인.

4단계. **성능/리소스·운영 자동화**
- 목표: 타임아웃/메모리 한도 내 처리 + 드리프트(리비전·트래픽) 방지.
- 작업: 성능 측정, 부하 테스트, CI에서 `update-traffic --to-latest` 정례화.

5단계. **릴리스 판정(Go/No-Go)**
- 목표: 기능/비기능/보안/운영 기준을 한 번에 보고 결정한다.
- 작업: 차단 이슈 목록과 완화책 정리, Go/No-Go 표 작성.

---

## 5. Cloud SQL Stop Policy (운영 정책 SSOT)

Cloud SQL 관련 내용은 **별도 정책 문서**를 SSOT로 삼는다. fileciteturn1file2  

### 5.1 기본 원칙

- Cloud SQL 인스턴스 `quali-pg` 는 **STOPPED + activationPolicy=NEVER** 상태가 “기본”이다.
- DB가 필요한 작업(배포, 마이그레이션, 스키마 점검 등)이 있을 때만 일시적으로 `ALWAYS` 로 변경해 사용한다.
- 작업 후 반드시 다시 `NEVER` 로 되돌려 STOP 상태를 유지한다.
- 애플리케이션에는 `DB_DISABLED=true` 를 설정해, DB가 꺼져 있어도 에러 없이 동작하도록 방어한다.

### 5.2 일상 스모크(매일 3줄)

- Cloud SQL 상태, Cloud Run 트래픽, Admin 도메인 HTML을 한 번에 확인한다.

```bash
gcloud sql instances list --format="table(name,region,state,settings.activationPolicy)"
gcloud run services describe quali-admin-domap --region asia-northeast1 --format="value(status.traffic)"
Invoke-WebRequest "https://admin.standardai.co.kr/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index.html"
```

- 기대 결과
  - SQL: `STOPPED / NEVER`
  - Run: `percent: 100`
  - HTML: 정상 다운로드

### 5.3 배포 차단 훅(CI)

- SQL이 STOP 상태이면 **배포를 실패**시키는 훅을 CI에 추가한다.

```bash
STATE=$(gcloud sql instances describe quali-pg --format="value(state)")
if [ "$STATE" = "STOPPED" ]; then
  echo "::error ::Cloud SQL is STOPPED; deployment blocked."
  exit 1
fi
```

---

## 6. DevOps 품질 게이트 (DoD v1.1 – Admin Tests + main 브랜치 보호)

이 절은 **QualiJournal 관리자 모드 – DevOps 품질 게이트 패키지 v1.1** 의 내용을 DoD 관점에서 요약한 것이다. fileciteturn1file3  

### 6.1 Admin Tests (자동화 테스트 품질 게이트)

- **목적**
  - 관리자 모드 배포 전 최소 동작을 보장하는 **자동화 스모크 테스트**.
  - Admin Tests 가 실패하면 main 브랜치에 병합할 수 없다.
- **필수 조건**
  - GitHub Actions 워크플로 `Admin Tests (pytest only)` 의 `test-admin` 잡이 성공 상태여야 한다.
  - `POST /api/report` 에 대해:
    - HTTP 200 OK
    - `ok=true`, `op="report"`, 유효한 `path`, 정수형 `count` 필드 존재
  - 해당 `path` 로 `GET /api/archive/{path}` 호출 시:
    - 200 OK
    - `Content-Type` 이 text/markdown 계열
    - 본문이 비어 있지 않은 리포트여야 한다.
- **체크리스트**
  - [ ] 최신 PR에서 `test-admin` 이 성공 상태다.
  - [ ] `/api/report` → `/api/archive/{path}` 까지 수동 스모크 결과가 기대와 같다.

### 6.2 main 브랜치 보호(브랜치 품질 게이트)

- **목적**
  - main 브랜치를 관리자 모드 배포의 **단일 진실 소스(SSOT)** 로 보호한다.
  - 직접 push 를 막고, 품질 게이트를 통과한 PR만 병합하게 한다.
- **필수 설정(GitHub Branch Protection)**
  - “Require a pull request before merging” = ON
  - “Require status checks to pass before merging” = ON
    - 필수 status check 목록에 `test-admin` 포함
  - “Require signed commits” = ON (모든 커밋 Verified)
  - “Require linear history” = ON (squash/rebase 기반 선형 히스토리)
- **체크리스트**
  - [ ] main 에 직접 push 하지 않고 PR로만 변경한다.
  - [ ] `test-admin` 이 필수 status check 로 설정되어 있다.
  - [ ] main 커밋이 모두 Verified 이다.
  - [ ] 로그가 선형 히스토리로 유지되고 있다.

### 6.3 DevOps 품질 게이트 선언

관리자 모드 인크리먼트가 **“Done”** 으로 인정되려면 반드시 아래 두 조건을 동시에 만족해야 한다.

1. 6.1 의 Admin Tests 를 통과한다.  
2. 6.2 의 main 브랜치 보호 규칙을 충족하는 PR을 통해 main 에 병합된다.

---

## 7. PR 템플릿 – DevOps 체크리스트 v1 (요약)

**목적**: 개발자가 PR을 만들 때 DoD-DevOps 기준을 간단히 다시 확인하도록 돕는 템플릿이다. fileciteturn1file4  

### 7.1 PR 기본 정보 섹션

- 제목, 관련 이슈, 변경 유형(신규 기능/버그 수정/리팩터링/설정 변경), 주요 변경 내용 3줄, 영향 범위 등을 간단히 적는다.

### 7.2 DevOps 신호등 체크리스트

1. **Admin Tests**
   - [ ] Actions 탭에서 `Admin Tests (pytest only)` / `test-admin` 이 green/passed 인지 확인했다.
2. **main 브랜치 보호**
   - [ ] main 에 직접 push 하지 않고 PR을 사용했다.
   - [ ] main 보호 규칙이 켜져 있는 화면을 확인했다.
3. **서명 커밋**
   - [ ] PR에 포함된 커밋이 GitHub 에서 Verified 로 보이는지 확인했다.
4. **선형 히스토리**
   - [ ] 머지 시 “Squash and merge” 또는 “Rebase and merge” 만 사용한다.

---

## 8. 릴리스 Go / No-Go 기준

다음 표는 DoD 점검 결과를 기반으로 릴리스 여부를 결정할 때 사용하는 상위 기준이다. fileciteturn1file1  

| 영역 | Go 기준 | No-Go(차단) | 기본 조치 |
|------|---------|-------------|-----------|
| 기능 | 정의서의 기능 항목을 모두 통과 | 주요 엔드포인트 1개라도 실패 | 핫픽스 또는 롤백 |
| 응답/코드 | 공통 JSON 스키마·HTTP 코드 일치 | 엔드포인트마다 구조가 제각각 | 응답 모델 정비 후 재배포 |
| 보안/IAM | 토큰+Invoker 최소 권한, 401/403 흐름 정상 | 공개 Invoker, 토큰 미적용 | 권한 재설정·익명 차단 |
| 배포/도메인 | 최신 리비전 100% 트래픽, 최신 HTML/JS | 구버전 UI/JS 노출 | 트래픽 최신화·캐시 무시 재배포 |
| 스케줄러 | 7일 연속 자동 보고서 성공률 ≥ 99% | 잦은 실패 또는 401 | 토큰 회전·잡 수정 |
| 백업 | 상태 OK, 버저닝/보존 정책 적용 | 마지막 성공 시각 없음 | 백업 잡/정책 보강 |
| Cloud SQL | STOP 정책 준수(기본 STOP) | 활성 상태 방치 | STOP 복구·차단 훅 적용 |

---

## 9. 샘플 명령 모음 (발췌)

### 9.1 헬스·상태·플로우·보고서 (cURL)

```bash
# 헬스 체크
curl -i "https://<DOMAIN>/health"

# 상태(KPI) – 성공/실패 비교
curl -s "https://<DOMAIN>/api/status" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN"
curl -s "https://<DOMAIN>/api/status"  # 기대: 401

# 플로우 트리거
curl -s -X POST "https://<DOMAIN>/api/tasks/flow?type=daily" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN"

# 보고서 생성 → 내보내기
curl -s -X POST "https://<DOMAIN>/api/report" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN"
curl -s "https://<DOMAIN>/api/export/md?date=YYYY-MM-DD"  -H "Authorization: Bearer DUMMY_ADMIN_TOKEN" -o report.md
curl -s "https://<DOMAIN>/api/export/csv?date=YYYY-MM-DD" -H "Authorization: Bearer DUMMY_ADMIN_TOKEN" -o report.csv
```

### 9.2 Cloud Run 트래픽 최신화 & 캐시 우회 (PowerShell)

```powershell
$SERVICE="quali-admin-domap"
$REGION="asia-northeast1"
$PROJECT="quali-journal-prod"

gcloud run services update-traffic $SERVICE --to-latest --region $REGION --project $PROJECT
Invoke-WebRequest "https://admin.standardai.co.kr/?v=$(Get-Random)" -Headers @{"Cache-Control"="no-cache"} -OutFile "deployed_index.html"
```

---

## 10. 운영자가 이 문서를 어떻게 사용하면 좋은지

1. **새 기능 개발 시작 전**  
   - 1~3장을 읽고 “이번 변경이 어느 DoD 항목과 관련돼 있는지” 표시한다.
2. **PR 열기 전**  
   - 6~7장의 DevOps 체크리스트를 따라 Admin Tests, 브랜치 보호, 서명 커밋 상태를 확인한다.
3. **릴리스 전 최종 점검**  
   - 4장(실행 계획)과 8장(Go/No-Go 기준)을 기반으로 체크리스트를 채우고, No-Go 항목이 없는지 확인한다.
4. **장애/이슈 발생 시**  
   - 1~3장으로 기능/환경 요구를 다시 확인하고, 5~6장으로 Cloud SQL·DevOps 쪽 설정이 어긋나지 않았는지 점검한다.

이 문서는 “관리자 모드 개발·운영의 기준선”으로 사용되며, 변경이 필요할 경우 **반드시 PR + 리뷰 + DevOps 품질 게이트**를 거쳐 갱신한다.
<!-- SSOT 변경 테스트 v2 (라벨 없이 CI 반응 확인용) -->

## 11. Admin API 응답 스키마 SSOT v1

> QualiJournal 관리자 모드 · 기능/비기능 DoD 기준 정렬용 응답 스키마 헌법

이 절은 관리자 모드(Admin)의 핵심 API에 대해
1) 대상 엔드포인트 목록(①),  
2) 공통 JSON 응답 스키마 정의(②),  
3) 엔드포인트별 매핑 표(③)  
를 한 곳에 모아 정리한 단일 기준(SSOT)이다.

### 11.1 대상 엔드포인트 목록 (①)

목적: 관리자가 실제로 사용하는 **핵심 Admin API**를 한눈에 정리하고,
각 엔드포인트의 **역할·현재 응답 구조(대략)**를 기록한다.

#### 11.1.1 핵심 엔드포인트 표

> 주의: `현재 응답 구조(대략)`은 정의서/런북에서 설명된 동작과 기존 구현을 바탕으로 한 요약이다.  
> 향후 실제 서버 코드 기준으로 재검증하여 필요 시 업데이트한다.

| path | method | 용도(한 줄) | 현재 응답 구조(요약) |
| --- | --- | --- | --- |
| `/api/status` | GET | KPI/게이트 상태 대시보드용 요약 지표 | `{ ok: true, total, ready_count, gate_required, ready_rate, gate_pass, ... }` |
| `/api/report` | POST | 특정 날짜/키워드 보고서 생성 | `{ ok: true, path }` (보고서 파일 경로) |
| `/api/export/md` | GET | MD 보고서 파일 다운로드 | MD 파일 스트림 응답 또는 파일 다운로드 응답 |
| `/api/export/csv` | GET | CSV 보고서 파일 다운로드 | CSV 파일 스트림 응답 또는 파일 다운로드 응답 |
| `/api/enrich/keyword` | POST | 키워드 기반 기사 보강/요약 작업 트리거 | `{ ok: true, job_id, ... }` |
| `/api/enrich/selection` | POST | 선정 기사 셋에 대한 추가 요약/보강 트리거 | `{ ok: true, job_id, ... }` |
| `/api/logs` | GET | 시스템 로그 파일 목록 조회 | `{ ok: true, logs: [{ name, size, mtime, ... }], ... }` |
| `/api/logs/{name}` | GET | 개별 로그 파일 내용 조회 | `text/plain` 또는 `{ ok: true, content: "..." }` |
| `/api/logs/{name}/download` | GET | 개별 로그 파일 다운로드 | 로그 파일 다운로드 응답 |
| `/api/backup/status` | GET | 정기 백업 최근 성공/실패 여부 및 시각 | `{ ok: true, last_success_at, last_failure_at, ... }` |
| `/api/config/gate_required` | GET | 현재 게이트 기준 값 조회 | `{ ok: true, gate_required }` |
| `/api/config/gate_required` | PATCH | 게이트 기준 값 변경 | `{ ok: true, gate_required }` (변경 후 값) |
| `/api/items` | GET | Ready/대기 기사 목록 조회 + 승인/거절 대상 | `{ ok: true, items: [{ id, title, source, state, ... }], ... }` |
| `/api/items/{id}/publish` | POST | 개별 기사 발행 | `{ ok: true, id, state: "published", ... }` |
| `/health` | GET | 헬스 체크 | `{ ok: true }` 또는 `"OK"` |

필요 시 `/api/debug/*` 등 추가적인 디버그/운영용 엔드포인트도 같은 형식으로 표에 추가한다.

---

### 11.2 공통 JSON 응답 스키마 정의 (②)

목적: 모든 Admin API에서 **성공/실패 응답 구조와 HTTP 코드 규칙**을 통일하기 위한 헌법을 정의한다.

#### 11.2.1 성공 응답 공통 스키마

- HTTP 코드: `200` (성공)
- JSON 형태(기본형):

```json
{
  "ok": true,
  "data": {
    "...": "엔드포인트별 payload"
  }
}
```

- 규칙
  - `ok`: 항상 `true`
  - `data`: 실제 비즈니스 데이터(엔드포인트마다 구조 상이)
  - 필요 시 페이징/메타 정보는 `data` 안 또는 `meta` 필드로 일관되게 추가한다.

- 예시
  - `/api/status`

    ```json
    {
      "ok": true,
      "data": {
        "total": 100,
        "ready_count": 80,
        "gate_required": 80,
        "ready_rate": 0.8,
        "gate_pass": true
      }
    }
    ```

  - `/api/report`

    ```json
    {
      "ok": true,
      "data": {
        "path": "reports/2025-11-15.md"
      }
    }
    ```

#### 11.2.2 실패 응답 공통 스키마

- HTTP 코드: `4xx` (클라이언트 오류) / `5xx` (서버 오류)
- JSON 형태(기본형):

```json
{
  "ok": false,
  "error": "사람이 읽을 수 있는 오류 메시지",
  "error_code": "E.ADMIN.SCOPE_CODE",
  "detail": {
    "...": "선택: 디버깅용 추가 정보"
  }
}
```

- 규칙
  - `ok`: 항상 `false`
  - `error`: 한 줄 오류 설명(한국어/영어 병행 가능, 운영자는 한국어 우선)
  - `error_code`: 기계가 읽기 좋은 짧은 코드(로그/모니터링 필터용)
  - `detail`: stack trace, 필드 오류 목록 등 선택 정보

#### 11.2.3 에러 코드 네이밍 규칙

- 접두어: `E.ADMIN.`
- 영역 코드(예시)
  - `STATUS`, `REPORT`, `EXPORT`, `ENRICH`, `LOGS`, `BACKUP`, `CONFIG`, `AUTH`, `COMMON` …
- 예시
  - `E.ADMIN.REPORT_NO_DATA`
  - `E.ADMIN.STATUS_UNAUTHORIZED`
  - `E.ADMIN.EXPORT_NOT_FOUND`

에러 코드 테이블은 별도 부록 또는 운영 문서에서 목록화할 수 있으며,
여기서는 네이밍 규칙과 대표 예시만 정의한다.

---

### 11.3 엔드포인트별 매핑 표 (③ 정의서 ↔ 실제 ↔ 타겟 스키마)

목적: 각 엔드포인트에 대해  
- 정의서/런북이 요구하는 동작,  
- 현재 구현(응답 구조),  
- 공통 스키마를 적용했을 때의 **타겟 스키마**  
를 한 줄로 비교해, 추후 구현/리팩터링 시 기준선으로 사용한다.

#### 11.3.1 매핑 표 (초안)

| path | method | 용도 | 현재 응답 구조(요약) | 타겟 스키마 (공통 규약 적용 후) |
| --- | --- | --- | --- | --- |
| `/api/status` | GET | KPI/게이트 요약 | `{ ok: true, total, ready_count, gate_required, ready_rate, gate_pass, ... }` | `200 + { ok: true, data: { total, ready_count, gate_required, ready_rate, gate_pass, ... } }` |
| `/api/report` | POST | MD/CSV 보고서 생성 | `{ ok: true, path }` | `200 + { ok: true, data: { path } }` |
| `/api/export/md` | GET | MD 보고서 다운로드 | MD 파일 스트림 또는 파일 응답 | `200 + 파일 응답` 또는 `302 + 파일 위치` (예외 정책으로 별도 정의) |
| `/api/export/csv` | GET | CSV 보고서 다운로드 | CSV 파일 스트림 또는 파일 응답 | 위와 동일 정책 적용 |
| `/api/logs` | GET | 로그 파일 목록 | `{ ok: true, logs: [...], ... }` | `200 + { ok: true, data: { logs: [{ name, size, mtime, ... }] } }` |
| `/api/logs/{name}` | GET | 로그 내용 보기 | `text/plain` 또는 `{ ok: true, content }` | `200 + { ok: true, data: { name, content } }` (대량 로그일 경우 스트리밍 전략 별도 정의) |
| `/api/logs/{name}/download` | GET | 로그 파일 다운로드 | 파일 다운로드 응답 | `200 + 파일 응답` (JSON 예외, 문서에 명시) |
| `/api/backup/status` | GET | 백업 상태 | `{ ok: true, last_success_at, last_failure_at, ... }` | `200 + { ok: true, data: { last_success_at, last_failure_at, last_success_status, ... } }` |
| `/api/config/gate_required` | GET | 게이트 기준 조회 | `{ ok: true, gate_required }` | `200 + { ok: true, data: { gate_required } }` |
| `/api/config/gate_required` | PATCH | 게이트 기준 변경 | `{ ok: true, gate_required }` | `200 + { ok: true, data: { gate_required } }` (변경 후 `/api/status` 결과와 일치) |
| `/api/items` | GET | 기사 목록 조회(Ready/대기) | `{ ok: true, items: [{ id, title, state, ... }], ... }` | `200 + { ok: true, data: { items: [...] } }` |
| `/api/items/{id}/publish` | POST | 기사 발행 | `{ ok: true, id, state: "published", ... }` | `200 + { ok: true, data: { id, state: "published", ... } }` |
| `/api/enrich/keyword` | POST | 키워드 enrich 작업 트리거 | `{ ok: true, job_id, ... }` | `200 + { ok: true, data: { job_id, ... } }` |
| `/api/enrich/selection` | POST | 선정 기사 enrich 작업 | `{ ok: true, job_id, ... }` | `200 + { ok: true, data: { job_id, ... } }` |
| `/health` | GET | 헬스 체크 | `{ ok: true }` 또는 `"OK"` | `200 + { ok: true }` (헬스도 공통 스키마 사용 권장) |

실제 구현과의 차이는 향후 서버 코드(server_quali.py 등)를 기준으로 재검증해,
필요 시 `현재 응답 구조` 열을 수정하거나, 구현을 타겟 스키마에 맞게 정렬한다.

---

### 11.4 다음 단계 메모

1. 위 표에서 `현재 응답 구조`가 실제 코드와 다른 부분은 실제 구현 기준으로 정정한다.  
2. 공통 스키마(11.2)를 FastAPI Pydantic 모델(BaseResponse 등)로 옮기고,  
   각 엔드포인트의 응답 타입을 해당 모델을 상속/활용하는 구조로 단계적으로 전환한다.  
3. 자동화 테스트에서 성공/실패 응답 스냅샷을 비교해,
   공통 스키마가 지켜지는지 회귀 테스트를 수행한다.



---

## III장. QualiJournal 관리자 모드 DoD – Admin API 응답 스키마 마감 보고서 원문

# QualiJournal 관리자 모드 DoD – Admin API 응답 스키마 마감 보고서

작성일: 20251116 (KST)  
Task Name: Admin API 응답 스키마 DoD 1차 마감  
Stage: Release 직전 (Build/Unit Test 완료)

---


## 1. 작업 개요

- 대상: admin/server_quali.py 의 Admin API 전체
- 목표: 성공 응답을 가능한 한 모두 `HTTP 200 + { ok: true, data: { ... } }` 패턴으로 통일
- 예외: SSE(`/api/tasks/{job_id}/stream`), 파일 다운로드(`/api/export/*`, `/api/logs/{name}/download`, `/api/archive/*`)

## 2. 주요 변경 축 요약

### 2.1 Items / Community

- `GET /api/items`  
  - candidate / ready / rejected / published / all 모든 state에서
    `{ ok:true, data:{ date, keyword, state, items }, ...레거시필드 }` 응답.

- `GET /api/community`  
  - snapshot 기반으로 total / approved / pending / articles 집계 후
    payload로 래핑하여 반환.

- `POST /api/community/save`  
  - 실제 변경(changed) > 0일 때만 `_sync_after_save()` 호출.
  - 변경이 없으면 sync를 건너뛰고
    `sync_log.stdout = "no changes; skip sync_selected_for_publish"` 로 응답.

- `POST /api/items/{item_id}/publish`  
  - 단건 발행 후 sync 결과를 `{ synced, item_id }` payload로 래핑.

### 2.2 Tasks

- `POST /api/tasks/flow`  
  - Task 생성 후 `{ ok:true, data:{ job_id, status, kind, args }, ... }` 응답.

- `GET /api/tasks/recent`  
  - 최근 로그 파일을 `{ ok:true, data:{ items:[{ id, size }, ...] } }` 구조로 반환.

- `GET /api/tasks/{job_id}`  
  - Task 존재 시 상태/시간 정보를 payload로 래핑,
    미존재 시 404 "job not found" 유지.

- `POST /api/tasks/{job_id}/cancel`  
  - 취소 플래그 설정 후 `{ ok:true, data:{ job_id, status, canceled:true } }` 응답.

- `GET /api/tasks/{job_id}/stream`  
  - SSE(text/event-stream) 그대로 유지.

### 2.3 Gate / KPI / Logs / Backup

- `GET /api/config/gate_required` (신규)  
  - 런타임 gate 값을 조회하여 `{ ok:true, data:{ gate_required }, gate_required }` 반환.

- `PATCH /api/config/gate_required`  
  - gate 값을 갱신하고 동일한 스키마로 응답.

- `GET /api/status`  
  - selection/community/kpi/gate 정보를 모두 집계한 payload를
    `{ ok:true, data: payload, ...레거시필드 }` 로 반환.

- `GET /api/logs` / `GET /api/backup/status`  
  - 로그 목록 / 백업 상태를 payload로 래핑하는 공통 패턴 적용.

### 2.4 Report / Enrich

- `GET /api/report`  
  - 날짜별 보고서 파일 목록을 `{ ok:true, data:{ date, files }, ... }` 구조로 반환.

- `POST /api/report`  
  - Markdown 리포트 생성 후
    `{ ok:true, data:{ op:"report", path, count, ts, duration_ms }, ... }` 응답.

- `POST /api/enrich/keyword` / `POST /api/enrich/selection`  
  - 요약 md 생성 후
    `{ ok:true, data:{ op, path, count, ts, duration_ms }, ... }` 응답.

## 3. 패치 위치(앵커) 메모

- Items: `@app.get("/api/items")`
- Community: `@app.get("/api/community")`, `@app.post("/api/community/save")`
- Publish: `@app.post("/api/items/{item_id}/publish")`
- Tasks: `@app.post("/api/tasks/flow")`, `@app.get("/api/tasks/recent")`,
  `@app.get("/api/tasks/{job_id}")`, `@app.post("/api/tasks/{job_id}/cancel")`
- Gate: `@app.get("/api/config/gate_required")`, `@app.patch("/api/config/gate_required")`
- Report/Enrich: `@app.get("/api/report")`, `@app.post("/api/report")`,
  `@app.post("/api/enrich/keyword")`, `@app.post("/api/enrich/selection")`
- Logs/Backup: `@app.get("/api/logs")`, `@app.get("/api/backup/status")`
- KPI: `@app.get("/api/status")`

## 4. PowerShell 스모크 요약

```powershell
$ADMIN_DOMAIN = "http://127.0.0.1:8080"
$ADMIN_TOKEN  = "admin-local-031128"
$headers = @{ "X-Admin-Token" = $ADMIN_TOKEN }

# gate GET/PATCH
$response = Invoke-WebRequest -Uri "$ADMIN_DOMAIN/api/config/gate_required" -Headers $headers
$response.Content | ConvertFrom-Json | Format-List

$body = @{ gate_required = 18 } | ConvertTo-Json
$response = Invoke-WebRequest -Uri "$ADMIN_DOMAIN/api/config/gate_required" -Method PATCH -Headers $headers -ContentType "application/json" -Body $body
$response.Content | ConvertFrom-Json | Format-List

# report
$body = @{ date = "2025-10-26" } | ConvertTo-Json
$response = Invoke-WebRequest -Uri "$ADMIN_DOMAIN/api/report" -Method POST -Headers $headers -ContentType "application/json" -Body $body
$response.Content | ConvertFrom-Json | Format-List

# enrich (keyword)
$response = Invoke-WebRequest -Uri "$ADMIN_DOMAIN/api/enrich/keyword" -Method POST -Headers $headers -ContentType "application/json" -Body "{}"
$response.Content | ConvertFrom-Json | Format-List

# status
$response = Invoke-WebRequest -Uri "$ADMIN_DOMAIN/api/status" -Headers $headers
$response.Content | ConvertFrom-Json | Format-List
```

## 5. 메모

- Admin API 응답 스키마 DoD Blocker는 모두 해소된 상태.
- 이후 과제는 Cloud SQL STOP Policy, Cloud Run/Scheduler 등 운영 레벨 DoD 설계.

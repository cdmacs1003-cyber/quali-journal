# PROJECT_DEVELOPMENT_MEMORY.md

문서 상태: QLIB Track A / Skillup Beta 프로젝트 기억 및 반복 문제 방지 가드레일  
작성일: 2026-05-26 KST  
적용 범위: QLIB Track A Skillup Beta, Bridge Runtime MVP, F13/Bridge/Skillup 개발 게이트, Codex 작업 운영  
문서 역할: `COMMON_DEVELOPMENT_WORKFLOW.md` 하위의 프로젝트별 실행 기억이며, `AGENTS.md`가 실제 작업 시 읽어야 하는 프로젝트 특수 규칙이다.

---

## 0. 문서 지위

이 문서는 QLIB Track A Skillup Beta 작업의 프로젝트별 기억이다.

상위 안전 기준은 `COMMON_DEVELOPMENT_WORKFLOW.md` 또는 `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`이다. 이 문서는 상위 안전 기준을 완화할 수 없고, QLIB/Track A에 필요한 더 엄격한 규칙만 추가한다.

충돌 시:

```text
USER_LATEST_EXPLICIT_REQUEST > COMMON_SAFETY_RULES > PROJECT_DEVELOPMENT_MEMORY > AGENTS_EXECUTION_RULES > SESSION/TASK_NOTES > TEMPORARY_ASSUMPTIONS
```

단, 다음 안전 영역은 어떤 하위 문서도 완화할 수 없다.

```text
SECRET_EXPOSURE_FORBIDDEN=true
PASS_WITHOUT_EVIDENCE_FORBIDDEN=true
UNAPPROVED_FILE_MODIFICATION_FORBIDDEN=true
UNAPPROVED_DELETE_MOVE_FORBIDDEN=true
UNAPPROVED_RUNTIME_OR_HTTP_FORBIDDEN=true
```

---

## 1. 프로젝트 목적

QLIB Track A의 목적은 2026년 7월 첫 주 Skillup 교육 베타에 적용 가능한 제한 버전을 준비하는 것이다.

핵심 검증 항목:

```text
Evidence 기반 답변
Bridge policy 차단
Skillup answer/HOLD flow
Raw leak 0
Feedback 회수
```

Track A의 P0 우선순위:

1. Bridge Runtime MVP
2. Skillup answer/HOLD flow
3. course_library_binding
4. raw leak / policy block
5. feedback queue
6. Beta Release Board

---

## 2. 현재 고정 상태

S5CW 이후 현재 고정 상태는 다음과 같다.

```text
CURRENT_RUNTIME_SMOKE_THREAD_STATUS=CLOSED_AT_EVIDENCE_CLOSURE
NEXT_RUNTIME_SMOKE_GATE=NONE
S5CX_CONTINUATION=NOT_RECOMMENDED
BRIDGE_FUNCTIONAL_200_BEHAVIOR=NOT_VERIFIED
RUNTIME_PASS=NOT_GRANTED
BRIDGE_SMOKE_PASS=NOT_GRANTED
BRIDGE_FUNCTIONAL_200_PASS=NOT_GRANTED
TRACK_A_PASS=NOT_GRANTED
BETA_PASS=NOT_GRANTED
F13_PASS=NOT_GRANTED
DEPLOYMENT_RELEASE_FINAL_APPROVAL=NOT_GRANTED
```

해석:

1. 현재 bounded unauthenticated runtime-smoke thread는 닫혔다.
2. S5CX continuation은 권장하지 않는다.
3. 다음 runtime-smoke gate는 없다.
4. 401 auth-boundary evidence는 authenticated functional 200 증거가 아니다.
5. Bridge functional 200 behavior는 계속 `NOT_VERIFIED`다.
6. PASS escalation은 금지한다.

---

## 3. 핵심 반복 문제 기록

| 날짜 | 문제 | 원인 | 다음부터 막는 규칙 |
|---|---|---|---|
| 2026-05-26 | clean worktree에서 required documents와 Bridge/F13 source surfaces가 `NOT_FOUND`가 되어 07SD가 `REVIEW_REQUIRED`로 종료 | old dirty worktree에 있던 핵심 산출물이 Git 정본, ProofPack, Recovery Package로 봉인되지 않은 untracked 상태였고, 새 clean worktree 생성 시 따라오지 않음 | Artifact Promotion State Policy와 Clean Worktree 전환 전 Gate를 적용한다 |

이번 문제는 PowerShell 종료 자체가 원인이 아니다. PowerShell 종료는 계기였고, 실제 원인은 산출물의 승격 상태가 정본화되지 않은 상태에서 clean worktree로 전환된 것이다.

---

## 4. Artifact Promotion State Policy

모든 작업 산출물은 다음 상태 중 하나로 분류해야 한다.

| State | 정의 | 사용 가능 여부 |
|---|---|---|
| `DRAFT` | Codex 또는 작업자가 만든 미검토 산출물 | 다음 작업의 기준으로 사용 금지 |
| `CANDIDATE` | 다음 작업 후보로 볼 수 있으나 승인 전 산출물 | 검토 전 clean worktree 반입 금지 |
| `APPROVED_SOURCE` | 사용자 또는 감독자가 사용 승인한 문서/소스 | 승인 범위 안에서 사용 가능 |
| `PROOFPACKED` | 경로, 증거, hash, 판정이 기록된 산출물 | 해당 범위의 증거로 사용 가능 |
| `CANONICAL` | 다음 clean worktree 또는 후속 작업의 정본 | 후속 작업 기준으로 사용 가능 |
| `QUARANTINE` | secret-like, out-of-scope, provenance 불명, 위험 파일 | 열람·복사·삭제 금지 |

필수 원칙:

```text
UNTRACKED_FILE_IS_NOT_CANONICAL=true
MISSING_IN_CLEAN_WORKTREE_DOES_NOT_MEAN_UNNECESSARY=true
DIRTY_WORKTREE_IS_EVIDENCE_UNTIL_CLASSIFIED=true
SECRET_FILE_IS_NOT_A_RECOVERY_SOURCE=true
```

---

## 5. Clean Worktree 전환 전 Gate

새 clean worktree를 만들거나 작업 기준 폴더를 전환하기 전에 다음을 확인한다.

필수 확인:

```text
git status --short
git log -1 --oneline
tracked modified list
untracked names list
required document path availability
required source surface path availability
secret-like filename presence
```

다음 조건 중 하나라도 있으면 clean worktree 전환 또는 다음 구현 게이트 진행을 멈춘다.

| 조건 | 판정 | 조치 |
|---|---|---|
| 핵심 문서가 untracked 상태 | `REVIEW_REQUIRED` | Recovery Package 또는 canonical placement 필요 |
| 핵심 소스가 untracked 상태 | `REVIEW_REQUIRED` | selective recovery approval 필요 |
| schema/tests/reports가 untracked 상태 | `REVIEW_REQUIRED` | artifact classification 필요 |
| `.env`, `.env.*`, secret-like 파일명 존재 | `QUARANTINE` | 열람·복사·삭제 금지 |
| old dirty worktree에만 required surface 존재 | `REVIEW_REQUIRED` | recovery planning gate 필요 |
| ProofPack 또는 manifest가 없음 | `NOT_VERIFIED` | PASS 금지 |

---

## Global Codex Completion Report Policy

For this project, every completed Codex task must create exactly one external Codex Completion Report markdown file.

External report root:
    H:\장기기억\docs\codex

Normal task report location:
    H:\장기기억\docs\codex\<YYYY>\<MM>\

Filename format:
    <YYYYMMDD>_<TASK_ID>_Completion_Report.md

This policy exists because screenshots are supporting evidence only. The external Completion Report .md is the primary evidence for downstream review, handover, and long-term memory.

Evidence priority:
1. Completion Report .md
2. Full terminal log .txt
3. Screenshot
4. User summary

The external report must preserve:
- task ID
- purpose
- repository path
- branch
- starting HEAD
- final HEAD
- changed files
- commands executed
- verification
- NOT_EXECUTED
- NOT_VERIFIED
- NOT_GRANTED claims
- risks
- rollback plan
- next recommended task
- final recommendation

The report policy does not grant runtime/server, real HTTP, DB/network, deploy, release, secret inspection, or broad regression permission. Those actions still require separate explicit approval.

---

## 6. Recovery Package 규칙

Recovery Package는 dirty worktree에서 clean worktree로 필요한 자료를 옮기기 전 필요한 승인 단위다.

Recovery Package에는 다음이 포함되어야 한다.

| 항목 | 필수 여부 |
|---|---:|
| source path | 필수 |
| target path | 필수 |
| artifact state | 필수 |
| provenance | 필수 |
| secret scan policy | 필수 |
| do-not-open register | 필수 |
| expected hash 또는 hash 생성 계획 | 권장 |
| recovery reason | 필수 |
| rollback plan | 필수 |
| explicit user approval | 필수 |

금지:

```text
OLD_DIRTY_WORKTREE_WHOLESALE_COPY=FORBIDDEN
SECRET_LIKE_FILE_RECOVERY=FORBIDDEN
ENV_BACKUP_CONTENT_INSPECTION=FORBIDDEN
GIT_CLEAN_BEFORE_CLASSIFICATION=FORBIDDEN
GIT_RESET_BEFORE_CLASSIFICATION=FORBIDDEN
GIT_RESTORE_BEFORE_CLASSIFICATION=FORBIDDEN
```

---

## 7. Required Documents for QLIB Track A / F13 / Bridge work

작업 시작 전 아래 문서를 찾는다.

```text
AGENTS.md
COMMON_DEVELOPMENT_WORKFLOW.md
COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md
PROJECT_DEVELOPMENT_MEMORY.md
PROJECT_DEVELOPMENT_MEMORY_PROMPT_CODEX_FINAL.md
QLIB_TA1_06S5CW_TO_TRACK_A_HANDOVER_REPORT_20260527.md
QLIB_FINAL_DEVELOPMENT_SCHEDULE_REPORT_20260522_FINAL.md
QLIB_COMPLETE_DEVELOPMENT_GUIDEBOOK_20260511_v1_2.md
ONTOLOGY_SEMANTIC_WORKFLOW_ENHANCEMENT_GUIDEBOOK_20260516_v1_0_FINAL.md
WAREHOUSE_DEVELOPMENT_COMPLETION_REPORT_창고 연동.md
PROJECT_DEVELOPMENT_GUIDEBOOK_창고_v1_0_FINAL.md
```

문서가 없으면:

```text
REQUIRED_DOCUMENT_STATUS=NOT_FOUND
IMPLEMENTATION=FORBIDDEN
NEXT_ACTION=DOCUMENT_AND_SURFACE_RECOVERY_PLANNING
```

---

## 8. Required Bridge/F13 Source Surfaces

Bridge Runtime MVP 또는 F13/Skillup 관련 작업은 아래 표면의 존재 여부를 확인한다.

```text
admin/f13_bridge_api.py
admin/f13_runtime_guard.py
schemas/f13_bridge_evidence_response.schema.json
schemas/f13_bridge_check_policy_response.schema.json
schemas/f13_bridge_explain_trace_response.schema.json
admin/tests/test_f13_bridge_api.py
admin/tests/test_f13_runtime_guard.py
admin/tests/test_f13_bridge_contract_regression.py
admin/tests/test_f13_bridge_evidence_response_schema.py
reports/f13/
docs/f13/
docs/feature_specs/F13_library_auto_intake_and_curation_v0.1.md
gap_maps/F13_current_gap_map.md
schemas/
shapes/
```

필수 표면이 clean worktree에서 없으면 구현으로 진행하지 않는다.

```text
BRIDGE_F13_SOURCE_SURFACES_STATUS=NOT_FOUND_OR_PARTIAL
NEXT_ACTION=RECOVERY_PLANNING_GATE
```

---

## 9. Secret and quarantine policy

다음 파일 또는 패턴은 `QUARANTINE`으로 분류한다.

```text
.env
.env.*
.env.bak
*.pem
*.key
secrets.*
credentials.*
service-account*.json
*credential*
*secret*
*token*
*key*
```

허용:

```text
filename-level observation for classification only
```

금지:

```text
content read
copy
summary
hashing if it requires opening content
cleanup
delete
restore
commit
```

---

## 10. Task Gate sequence for current recovery path

현재 Track A 복구 경로는 다음을 따른다.

```text
T-A1-07SE_REVIEW_REQUIRED_DOCUMENT_AND_SURFACE_RECOVERY
T-A1-07SF_DOCUMENT_AND_SOURCE_SURFACE_RECOVERY_APPROVAL_PACKET_DRAFT_ONLY
T-A1-07SG_APPROVED_DOCUMENT_AND_SOURCE_SURFACE_PLACEMENT_ONLY
T-A1-07SH_POST_RECOVERY_STATIC_VERIFICATION_GATE
T-A1-07SD-R_IMPLEMENTATION_PACKET_DRAFT_RETRY
```

07SE~07SF는 파일 수정·복사·생성 금지다. 07SG부터도 명시 승인 범위 안에서만 배치한다.

---

## 11. Status vocabulary

이 프로젝트에서 미확인·미실행 상태는 다음처럼 표기한다.

| Status | 의미 |
|---|---|
| `NOT_EXECUTED` | 실행하지 않음 |
| `NOT_VERIFIED` | 검증하지 않음 |
| `NOT_GRANTED` | 승인 또는 PASS 권한 없음 |
| `REVIEW_REQUIRED` | 사람 검토 또는 복구 계획 필요 |
| `FORBIDDEN` | 현재 게이트에서 금지 |
| `DEFERRED` | 후속 게이트로 이관 |
| `QUARANTINE` | 열람·복사·삭제 금지 |

금지:

```text
실행하지 않은 것을 PASS로 표기 금지
검증하지 않은 것을 DONE으로 표기 금지
401 auth-boundary를 authenticated functional 200 PASS로 표기 금지
Warehouse PASS를 QLIB 전체 PASS로 확장 금지
```

---

## 12. Completion and handover requirements

작업 종료 보고에는 다음을 포함한다.

| 항목 | 필수 여부 |
|---|---:|
| current worktree path | 필수 |
| old dirty worktree handling | 해당 시 필수 |
| git status summary | 필수 |
| required documents status | 필수 |
| source surfaces status | 해당 시 필수 |
| artifact state table | 필수 |
| changed files | 변경 시 필수 |
| created files | 생성 시 필수 |
| tests executed | 실행 시 필수 |
| tests not executed | 미실행 시 필수 |
| secret inspection status | 필수 |
| rollback plan | 변경 시 필수 |
| next one task | 필수 |

---

## 13. Final operating principle

```text
작업 산출물은 정본화되기 전까지 작업물일 뿐이다.
작업물은 Git commit, Recovery Package, ProofPack manifest 중 하나로 봉인되어야 다음 작업의 기준이 된다.
clean worktree는 정리된 작업대이지 자동 복구 장치가 아니다.
old dirty worktree는 지우는 대상이 아니라 분류해야 할 증거다.
```

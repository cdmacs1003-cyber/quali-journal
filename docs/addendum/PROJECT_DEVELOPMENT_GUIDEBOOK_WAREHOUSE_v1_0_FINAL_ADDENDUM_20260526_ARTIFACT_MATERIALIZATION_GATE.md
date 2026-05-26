# PROJECT_DEVELOPMENT_GUIDEBOOK_창고_v1.0 ADDENDUM — Artifact Materialization and Clean Worktree Gate

작성일: 2026-05-26 KST  
상위 기준: `COMMON_DEVELOPMENT_WORKFLOW.md`, `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`, `PROJECT_DEVELOPMENT_MEMORY.md`, `AGENTS.md`  
대상 원본: `PROJECT_DEVELOPMENT_GUIDEBOOK_창고_v1_0_FINAL.md`  
변경 방식: 원본 v1.0 직접 수정 금지. 본 문서는 Addendum으로만 적용한다.

---

## 0. Addendum 판정

```text
ADDENDUM_DECISION=APPROVE_AS_GOVERNANCE_HYGIENE_PATCH
ORIGINAL_GUIDEBOOK_DIRECT_MODIFICATION=FORBIDDEN
CHANGE_POLICY=ADDENDUM_ONLY
SESSION_WORK_PLAN_FILE_CREATION=NOT_EXECUTED
TASK_EXECUTION_NOTE_FILE_CREATION=NOT_EXECUTED
```

본 부록은 창고 v1.0 고정본의 내용을 바꾸지 않는다. 원본 문서가 선언한 `CHANGE_POLICY=ADDENDUM_OR_NEW_VERSION_ONLY` 원칙에 따라 반복 문제 방지 규칙만 추가한다.

---

## 1. 적용 배경

QLIB Track A / Bridge Runtime MVP 작업 중 clean worktree에서 required documents와 Bridge/F13 source surfaces가 `NOT_FOUND`가 되어 07SD가 `REVIEW_REQUIRED`로 종료되었다.

주요 원인:

```text
핵심 산출물이 Git 정본, ProofPack, 또는 Recovery Package로 봉인되기 전에 old dirty worktree의 untracked 파일로만 존재했고, 새 clean worktree 생성 시 따라오지 않았다.
```

이는 창고 문서의 원본 보존, trace, ProofPack, backup/restore 원칙과 같은 계열의 운영 문제다. 따라서 창고 개발·운영·승격·봉인 절차에도 산출물 물질화 규칙을 추가한다.

---

## 2. Artifact Promotion State Policy

창고 및 QLIB 관련 산출물은 다음 상태 중 하나로 분류한다.

| State | 정의 | 처리 |
|---|---|---|
| `DRAFT` | 생성 또는 수정되었으나 검토 전 | 기준 사용 금지 |
| `CANDIDATE` | 정본 후보 | review 필요 |
| `APPROVED_SOURCE` | 사용자 또는 감독자가 승인한 소스 | 승인 범위 내 사용 가능 |
| `PROOFPACKED` | 증거·경로·hash로 봉인 | 해당 범위 증거로 사용 가능 |
| `CANONICAL` | 다음 작업의 정본 | clean worktree 반입 가능 |
| `QUARANTINE` | secret-like, 범위 밖, 출처 불명, 위험 파일 | 열람·복사·삭제 금지 |

운영 원칙:

```text
UNTRACKED_FILE_IS_NOT_CANONICAL=true
DIRTY_WORKTREE_IS_EVIDENCE_UNTIL_CLASSIFIED=true
MISSING_IN_CLEAN_WORKTREE_DOES_NOT_MEAN_UNNECESSARY=true
SECRET_FILE_IS_NOT_A_RECOVERY_SOURCE=true
```

---

## 3. Clean Worktree 전환 전 Gate

새 clean worktree 생성 또는 작업 폴더 전환 전 다음을 확인한다.

| Check | Required |
|---|---:|
| current path | YES |
| latest commit | YES |
| `git status --short` | YES |
| tracked modified list | YES |
| untracked filename list | YES |
| required documents status | YES |
| required source surfaces status | YES |
| secret-like filename classification | YES |
| next worktree basis | YES |
| old dirty worktree handling | YES |

차단 조건:

| Condition | Decision | Handling |
|---|---|---|
| 핵심 문서가 untracked | `REVIEW_REQUIRED` | Document placement approval 필요 |
| 핵심 소스가 untracked | `REVIEW_REQUIRED` | Source surface recovery approval 필요 |
| schema/tests/reports가 untracked | `REVIEW_REQUIRED` | Artifact classification 필요 |
| `.env`, `.env.*`, secret-like 파일명 존재 | `QUARANTINE` | 내용 열람 금지 |
| ProofPack/manifest 없음 | `NOT_VERIFIED` | PASS 금지 |

---

## 4. Recovery Package 규칙

old dirty worktree에서 clean worktree로 항목을 가져오기 전 Recovery Package를 작성한다.

Recovery Package 필수 항목:

| Field | Required |
|---|---:|
| source path | YES |
| target path | YES |
| artifact state | YES |
| provenance | YES |
| risk classification | YES |
| secret exclusion check | YES |
| approval scope | YES |
| rollback plan | YES |
| final recommendation | YES |

금지:

```text
OLD_DIRTY_WORKTREE_WHOLESALE_COPY=FORBIDDEN
SECRET_LIKE_FILE_RECOVERY=FORBIDDEN
GIT_CLEAN_BEFORE_CLASSIFICATION=FORBIDDEN
GIT_RESET_BEFORE_CLASSIFICATION=FORBIDDEN
GIT_RESTORE_BEFORE_CLASSIFICATION=FORBIDDEN
```

---

## 5. 창고 인수인계 보강 항목

창고 또는 QLIB 장기 작업 인수인계에는 다음을 추가한다.

| 항목 | 설명 |
|---|---|
| current worktree | 다음 작업 기준 폴더 |
| do-not-touch worktree | 수정·정리 금지 폴더 |
| artifact state table | DRAFT/CANDIDATE/APPROVED_SOURCE/PROOFPACKED/CANONICAL/QUARANTINE 분류 |
| recovery package status | 필요한 경우 approval packet 상태 |
| secret-like filename register | 파일명 수준만 기록, 내용 열람 금지 |
| clean worktree eligibility | YES / NO / REVIEW_REQUIRED |

---

## 6. SESSION_WORK_PLAN.md / TASK_EXECUTION_NOTE.md 처리

현재 사용자는 별도 `SESSION_WORK_PLAN.md`와 `TASK_EXECUTION_NOTE.md` 파일을 생성하지 않았다.

따라서 본 Addendum은 두 파일을 생성하지 않는다.

대신 해당 정보는 다음 위치 중 하나에 기록한다.

1. Codex completion report
2. Handover report
3. Recovery Package
4. ProofPack manifest
5. PROJECT_DEVELOPMENT_MEMORY.md

---

## 7. 최종 한 줄

```text
창고와 QLIB 산출물은 정본화되기 전까지 작업물이며, 작업물은 Git commit, Recovery Package, ProofPack manifest 중 하나로 봉인되어야 다음 clean worktree와 다음 작업의 기준이 된다.
```

# PROJECT_DEVELOPMENT_MEMORY_POST_R9ZNW259_ADDENDUM_20260702

문서 상태: ADDENDUM_CANDIDATE_FOR_CANONICAL_PLACEMENT  
작성일: 2026-07-02 KST  
대상 원본: `PROJECT_DEVELOPMENT_MEMORY.md`  
변경 방식: 원본 직접 수정 금지. 본 문서는 post-R9ZNW259 최신 상태를 반영하는 Addendum으로만 적용한다.  
적용 범위: QLIB Track A / Skillup Beta / Bridge Runtime MVP / R9ZNW-254~259 evidence lineage  
권장 repo 저장 위치: `H:\a\퀄리저널_track_a_clean_standalone\docs\governance\addenda\PROJECT_DEVELOPMENT_MEMORY_POST_R9ZNW259_ADDENDUM_20260702.md`  
권장 장기기억 보관 위치: `H:\장기기억\docs\codex\2026\07\governance_addenda\PROJECT_DEVELOPMENT_MEMORY_POST_R9ZNW259_ADDENDUM_20260702.md`

---

## 0. Addendum 판정

```text
ADDENDUM_DECISION=APPROVE_AS_POST_R9ZNW259_PROJECT_MEMORY_UPDATE_CANDIDATE
ORIGINAL_PROJECT_DEVELOPMENT_MEMORY_DIRECT_MODIFICATION=FORBIDDEN
CHANGE_POLICY=ADDENDUM_ONLY_UNTIL_EXPLICIT_REPO_PLACEMENT_GATE
CURRENT_ACCEPTED_HEAD_AFTER_R9ZNW259=d31b7d3 admin: show safe beta answer and guard admin background calls
```

본 Addendum은 `PROJECT_DEVELOPMENT_MEMORY.md`를 대체하지 않는다.  
상위 안전 기준은 계속 `COMMON_DEVELOPMENT_WORKFLOW.md` 또는 `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`이며, 실행 규칙은 `AGENTS.md`를 따른다.

---

## 1. 목적

R9ZNW-254부터 R9ZNW-259까지 실제 베타 최소 UI와 Skillup safe answer 구조가 변화했다.  
기존 `PROJECT_DEVELOPMENT_MEMORY.md`는 Track A 초기 recovery/Bridge runtime 상태를 반영하지만, 현재 상태인 Beta Minimal Skillup input/output UI, `safe_short_answer`, admin background call guard를 충분히 반영하지 못한다.

따라서 본 Addendum은 다음을 고정한다.

```text
1. 현재 베타 상태의 정확한 이름
2. 완료된 것과 아직 완료되지 않은 것의 경계
3. 다음 기능 개발 순서
4. PASS 확장 금지 경계
5. R9ZNW 후속 작업의 필수 읽기 기준
```

---

## 2. 현재 상태명

현재 상태는 다음으로만 부른다.

```text
BETA_MINIMAL_SKILLUP_INPUT_OUTPUT_UI_READY_WITH_LIMITS
SAFE_SHORT_ANSWER_STRUCTURE_COMMITTED_WITH_LIMITS
ADMIN_BACKGROUND_CALL_GUARD_COMMITTED_WITH_LIMITS
QAIEMBLEM_BETA_CANDIDATE_BRANDING_COMMITTED_WITH_LIMITS
```

아래 표현은 금지한다.

```text
BETA_PASS
TRACK_A_PASS
F13_PASS
QLIB_PRODUCT_DONE
RELEASE_READY
PRODUCTION_READY
CANONICAL_BRAND_APPROVED
LEGAL_TRADEMARK_APPROVED
BROWSER_AUTOMATION_PASS
```

---

## 3. R9ZNW-254~259 최신 증거 상태

| Gate | 현재 판정 | 의미 | 제한 |
|---|---|---|---|
| R9ZNW-254 | `BETA_MINIMAL_TEST_PACKAGE_READY_WITH_LIMITS` | 자동화 기술 패키지와 human visual handoff board 정리 | broad Beta PASS 아님 |
| R9ZNW-255 | `APPROVE_NEXT_HUMAN_VISUAL_REVIEW_RERUN_WITH_LIMITS` | 기본 beta user view에서 Admin Ops, ADMIN_TOKEN, 미인증 숨김 | 시각 재검토 필요 |
| R9ZNW-256 | `APPROVE_NEXT_HUMAN_VISUAL_REVIEW_RERUN_WITH_LIMITS` | 영어 중심 UI를 쉬운 한글 copy로 교체 | runtime/browser 미검증 |
| R9ZNW-257 | `APPROVE_NEXT_HUMAN_VISUAL_REVIEW_RERUN_WITH_LIMITS` | 중복 로고, admin approval section, internal debug/status 숨김 | human visual review 필요 |
| R9ZNW-258 | `APPROVE_NEXT_FINAL_HUMAN_VISUAL_REVIEW_WITH_LIMITS` | visible `test_minimal` → `베타 테스트용 화면` | broad PASS 아님 |
| R9ZNW-259 | `APPROVE_NEXT_HUMAN_BETA_SUBMIT_REVIEW_WITH_LIMITS` | `safe_short_answer` 구조와 admin background call guard 커밋 | live safe answer validation은 다음 게이트 |

---

## 4. 현재 완료된 범위

| 항목 | 상태 | 증거/설명 |
|---|---|---|
| Beta Minimal Skillup UI | `COMMITTED_WITH_LIMITS` | 손님용 질문창, 질문 보내기, 답변/보류, 근거/상태 요약, 피드백 버튼 |
| Korean user copy | `COMMITTED_WITH_LIMITS` | 쉬운 한글 표시 |
| Admin surface hiding | `COMMITTED_WITH_LIMITS` | Admin Ops, ADMIN_TOKEN, 기사 승인, 내부 status 숨김 |
| QAIEMBLEM beta/candidate brand | `COMMITTED_WITH_LIMITS` | beta/candidate 용도 한정 |
| selected route POST evidence | `CLOSED_WITH_LIMITS` | exact local route evidence only |
| root/submit smoke evidence | `CLOSED_WITH_LIMITS` | local bounded evidence only |
| `safe_short_answer` schema/field | `COMMITTED_WITH_LIMITS` | safe answer display 구조 추가 |
| admin background call guard | `COMMITTED_WITH_LIMITS` | test_minimal 기본 화면에서 admin polling/fetch guard |

---

## 5. 아직 완료되지 않은 범위

| 항목 | 상태 | 다음 처리 |
|---|---|---|
| Live safe answer validation | `NOT_VERIFIED` | human beta submit review 또는 bounded API validation |
| Library Core 최소 evidence 연동 | `NOT_EXECUTED` | Phase 1에서 수행 |
| Warehouse → Library → Bridge → Skillup E2E | `NOT_EXECUTED` | Phase 7에서 ProofPack으로 수행 |
| DB/external/provider/cloud 연동 | `NOT_GRANTED` | 별도 승인 게이트 |
| Browser automation rendering | `HOLD_BROWSER_AUTOMATION_TOOLING_LIMITATION` | 제품 blocker가 아닌 tooling limitation으로 분리 |
| legal/trademark/canonical brand approval | `NOT_GRANTED` | 별도 법무/브랜드 승인 |
| release/deploy/prod readiness | `NOT_GRANTED` | Release Board 전까지 금지 |
| Track A PASS / F13 PASS | `NOT_GRANTED` | 독립 증거 필요 |

---

## 6. 추가된 운영 상수

```text
CURRENT_BETA_IS_SKILLUP_IO_VERTICAL_SLICE=true
CURRENT_BETA_IS_NOT_FULL_PRODUCT=true
SAFE_SHORT_ANSWER_FIELD_IS_ALLOWED_ONLY_WHITELISTED=true
SKILLUP_DIRECT_LIBRARY_DB_ACCESS_FORBIDDEN=true
SKILLUP_DIRECT_WAREHOUSE_DB_ACCESS_FORBIDDEN=true
BRIDGE_EVIDENCE_TRACE_REQUIRED=true
RAW_JSON_DISPLAY_FORBIDDEN=true
RAW_STANDARD_TEXT_DISPLAY_FORBIDDEN=true
ADMIN_BACKGROUND_CALLS_FOR_TEST_MINIMAL_FORBIDDEN=true
BROWSER_AUTOMATION_FAILURE_IS_TOOLING_LIMITATION_UNTIL_PRODUCT_DEFECT_EVIDENCE=true
```

---

## 7. R9ZNW 후속 작업 우선순위

### P0 — 현재 베타 세로 절단면 닫기

```text
목표:
스킬업 질문 → Bridge/route → safe_short_answer → UI 표시 → 피드백 버튼
```

완료 조건:

```text
[ ] “솔더링이란?” 질문에 safe_short_answer 또는 승인된 safe fallback 표시
[ ] raw JSON 미노출
[ ] 관리자 API 401 소음 없음
[ ] token/key/path/secret 미노출
[ ] 피드백 버튼 작동
[ ] 결과는 PASS_WITH_LIMITS로만 표기
```

### P1 — 퀄리도서관 최소 연동

```text
목표:
로컬 낱말카드가 아니라 Library Core의 승인된 Evidence/Reference에서 safe_short_answer를 가져온다.
```

완료 조건:

```text
[ ] Library seed evidence 존재
[ ] evidence_id 존재
[ ] bridge_trace_id 존재
[ ] Bridge.retrieve_evidence 계약 통과
[ ] Skillup 직접 DB 조회 없음
[ ] 원문 장문 노출 없음
```

### P2 — 창고 → 도서관 승격 루프 재사용

```text
목표:
Warehouse 후보 자료가 review/approval/dry-run/promotion trace를 거쳐 Library Core로 승격된다.
```

완료 조건:

```text
[ ] warehouse_item 생성
[ ] raw_hash / provenance / rights / sensitivity 확인
[ ] review / approval 기록
[ ] promotion dry-run
[ ] promotion trace
[ ] Library card / evidence pointer 생성
[ ] ProofPack 봉인
```

---

## 8. Completion vocabulary update

| 표현 | 허용 여부 | 조건 |
|---|---:|---|
| `PASS_WITH_LIMITS` | YES | 제한 증거와 한계가 함께 기록될 때 |
| `READY_WITH_LIMITS` | YES | 베타 최소 시작 조건만 충족할 때 |
| `COMMITTED_WITH_LIMITS` | YES | commit 되었으나 runtime/live 검증이 분리될 때 |
| `NOT_EXECUTED` | YES | 실행하지 않은 항목 |
| `NOT_VERIFIED` | YES | 검증하지 않은 항목 |
| `NOT_GRANTED` | YES | 승인·판정 권한 없음 |
| `DONE` | NO | 증거 범위 불명확 |
| `BETA_PASS` | NO | 독립 증거와 release board 전까지 금지 |

---

## 9. Artifact state update

| Item | Path / Ref | State | Evidence | Next handling |
|---|---|---|---|---|
| Beta Minimal HTML source | `admin/index.html` | `CANONICAL_WITH_LIMITS` | R9ZNW-259 lineage | human beta submit review |
| Beta Minimal HTML dist | `admin/dist/index.html` | `CANONICAL_WITH_LIMITS` | R9ZNW-259 lineage | human beta submit review |
| F13 bridge route | `admin/f13_bridge_api.py` | `CANONICAL_WITH_LIMITS` | R9ZNW-259 commit | bounded validation needed |
| Skillup answer schema | `schemas/skillup_answer_hold_response.schema.json` | `CANONICAL_WITH_LIMITS` | R9ZNW-259 commit | schema/runtime validation later |
| Original brand packet | `docs/brand/` | `CANDIDATE_OUT_OF_SCOPE_PRESERVE_NOT_CANONICAL` | untracked candidate | do not mutate/stage/commit |
| This Addendum | recommended repo addenda path | `CANDIDATE` until committed | user-provided generated addendum | place via explicit Codex gate |

---

## 10. Required future reading order addition

For future R9ZNW tasks after placement, read this Addendum after `PROJECT_DEVELOPMENT_MEMORY.md` and before task-specific completion reports.

Recommended future order:

```text
1. COMMON_DEVELOPMENT_WORKFLOW.md or COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md
2. PROJECT_DEVELOPMENT_MEMORY.md
3. PROJECT_DEVELOPMENT_MEMORY_POST_R9ZNW259_ADDENDUM_20260702.md
4. AGENTS.md
5. Latest relevant R9ZNW completion report
6. Quali UI skill / governance docs when UI is involved
```

---

## 11. Final recommendation

```text
FINAL_RECOMMENDATION=APPROVE_AS_PROJECT_MEMORY_ADDENDUM_CANDIDATE
NEXT_ACTION=PLACE_IN_REPO_AND_LONG_MEMORY_WITH_EXPLICIT_COMMIT_GATE
DO_NOT_EDIT_ORIGINAL_PROJECT_DEVELOPMENT_MEMORY_YET=true
```

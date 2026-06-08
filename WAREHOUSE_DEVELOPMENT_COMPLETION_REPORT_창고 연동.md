# 퀄리 창고 백엔드/API 및 QualiLibrary Ripple 연동 완료 보고서

| 항목 | 내용 |
|---|---|
| 문서명 | WAREHOUSE_DEVELOPMENT_COMPLETION_REPORT_20260514.md |
| 작성일 | 2026-05-14 |
| 프로젝트 루트 | `H:\a\퀄리저널_pr_clean` |
| 운영 장기기억 루트 | `H:\장기기억` |
| FINAL_DECISION | PASS |
| SCOPE | WAREHOUSE_BACKEND_API_AND_LIBRARY_RIPPLE_INTEGRATION |
| OVERALL_QLIB_PRODUCT_DONE | NO |
| UI | NOT_EXECUTED |
| BRIDGE_RUNTIME | NOT_EXECUTED |
| SKILLUP | NOT_EXECUTED |
| ANALYTICS | NOT_EXECUTED |
| INTEGRATED_UI_SHELL | NOT_EXECUTED |
| Handover | REPORT_INTEGRATED |

이 문서의 PASS는 전체 QLIB 제품군 완성 선언이 아니다. PASS 범위는 **퀄리 창고 백엔드/API와 QualiLibrary Ripple 실제 연동**으로 고정한다.

```text
FINAL_DECISION=PASS
SCOPE=WAREHOUSE_BACKEND_API_AND_LIBRARY_RIPPLE_INTEGRATION

PASS 대상:
- Warehouse backend/API
- review/approval/dry-run/promote flow
- QualiLibrary Ripple 실제 연동
- production-root smoke
- ProofPack / Release Board / Backup-Restore dry-run

NOT_EXECUTED:
- Warehouse UI
- Library UI
- Bridge Runtime
- Skillup Education Runtime
- Analytics
- Integrated UI Shell

OVERALL_QLIB_PRODUCT_DONE=NO
```

---

## 1. 결론

2026-05-14 기준으로 퀄리 창고는 **창고 백엔드/API 및 도서관 Ripple 연동 범위에서 PASS**다.

완료된 것은 다음이다.

| 영역 | 판정 |
|---|---|
| 창고 item 생성, raw 보존, hash 검증 | PASS |
| provenance, rights, sensitivity 검증 | PASS |
| review, approval, dry-run, promote 상태 흐름 | PASS |
| 실제 `qualilibrary_ripple add --dry-run` | PASS |
| 실제 `qualilibrary_ripple add` | PASS |
| 실제 `qualilibrary_ripple verify` | PASS |
| 실제 `ripple rebuild` | PASS |
| 실제 `ripple search` | PASS |
| 실제 `show node` | PASS |
| 운영 루트 `H:\장기기억` 쓰기 및 산출물 검증 | PASS |
| ProofPack, Release Board, Backup/Restore dry-run | PASS |

완료되지 않은 것은 다음이다.

| 영역 | 판정 |
|---|---|
| Warehouse UI | NOT_EXECUTED |
| Library UI | NOT_EXECUTED |
| Bridge Runtime | NOT_EXECUTED |
| Skillup Education Runtime | NOT_EXECUTED |
| Analytics | NOT_EXECUTED |
| Integrated UI Shell | NOT_EXECUTED |

---

## 2. 기준 문서

| 우선순위 | 문서 | 역할 |
|---|---|---|
| 1 | 사용자 최신 명시 요청 | 실제 개발, 시뮬레이션 테스트, 실제 테스트, 운영 100% 확인 요청 |
| 2 | `H:\헌법_프롬프트\헌법\COMMON_DEVELOPMENT_WORKFLOW.md` | 최상위 개발 헌법 |
| 3 | `H:\헌법_프롬프트\헌법\QLIB_COMPLETE_DEVELOPMENT_GUIDEBOOK_20260511_v1_2.md` | QLIB 전체 모듈 기준 |
| 4 | `H:\헌법_프롬프트\가이드북\PROJECT_DEVELOPMENT_GUIDEBOOK_창고_v1_0_FINAL.md` | 창고 실행 가이드북 |

적용 원칙:

- 증거 없는 PASS 금지
- 실행하지 않은 항목은 PASS로 표시하지 않음
- 창고는 Library Core를 직접 오염시키지 않음
- dry-run 없는 promote 금지
- 승인 없는 promote 금지
- Skillup은 Library DB나 Warehouse DB를 직접 조회하지 않고 Bridge 계약을 통해 Evidence만 사용해야 함
- UI는 raw path, secret, token, paid standard raw text, 내부 경로를 공개 사용자에게 노출하면 안 됨

---

## 3. 구현 요약

작업명:

```text
Warehouse Promote → QualiLibrary Ripple 실제 승격 어댑터 구현
```

구현된 실제 흐름:

```text
Warehouse item 생성
-> 상태 전이: captured -> untriaged -> triaged -> pending_review
-> reviewer review
-> approver approval
-> promotion dry-run
-> qualilibrary_ripple add --dry-run
-> promote
-> qualilibrary_ripple add
-> qualilibrary_ripple verify
-> qualilibrary_ripple ripple rebuild
-> qualilibrary_ripple ripple search
-> qualilibrary_ripple show
-> promotion trace
-> proofpack
-> backup / restore dry-run
-> release board
```

핵심 설계:

- 별도 복잡한 서비스가 아니라 기존 `qualilibrary_ripple` CLI를 subprocess로 호출한다.
- 실제 도서관 엔진 사용은 `QUALI_LIBRARY_RIPPLE_ENABLED=1`일 때만 켜진다.
- 엔진이 켜진 상태에서는 dry-run PASS 없이 promote가 불가능하다.
- `add`, `verify`, `rebuild`, `search`, 필수 파일 존재 중 하나라도 실패하면 promote는 409로 중단된다.
- trace와 proofpack에 실행 명령, 결과, 산출물 경로, node id, memory item id가 남는다.

---

## 4. 주요 파일

| 파일 | 상태 | 역할 |
|---|---|---|
| `admin/warehouse_core.py` | IN_SCOPE | 창고 API, validator, dry-run, promote, QualiLibrary Ripple adapter |
| `admin/tests/test_warehouse_core_api.py` | IN_SCOPE | 창고 시뮬레이션 및 실제 Ripple 통합 테스트 |
| `server_quali.py` | IN_SCOPE_PARTIAL | root app에 warehouse router 연결, 테스트 호환 모델 |
| `admin/server_quali.py` | IN_SCOPE_PARTIAL | admin app에 warehouse router 연결. 기존 dirty diff와 분리 확인 필요 |
| `reports/warehouse_operational_smoke/run_20260514041108/` | IN_SCOPE_EVIDENCE | 운영 스모크 증거 |
| `reports/WAREHOUSE_DEVELOPMENT_COMPLETION_REPORT_20260514.md` | IN_SCOPE_REPORT | 완료 및 인수인계 보고서 |

주의:

- `admin/server_quali.py`에는 창고 작업 외 기존 대규모 dirty diff가 섞여 있다.
- `server_quali.py`, `admin/server_quali.py`는 Release Board에 `IN_SCOPE_PARTIAL`로 명시했다.
- worktree 전체를 임의 revert/delete하면 안 된다.

---

## 5. 창고 API 구현 범위

기본 prefix:

```text
/api/warehouse
```

| Method | Path | 역할 |
|---|---|---|
| GET | `/api/warehouse/manifest` | 창고 manifest 조회 |
| GET | `/api/warehouse/status` | 상태 및 gate 조회 |
| POST | `/api/warehouse/items` | item 생성 |
| GET | `/api/warehouse/items` | item 목록 조회 |
| GET | `/api/warehouse/items/{item_id}` | item 단건 조회 |
| PATCH | `/api/warehouse/items/{item_id}/status` | 상태 전이 |
| POST | `/api/warehouse/items/{item_id}/reviews` | 리뷰 등록 |
| POST | `/api/warehouse/items/{item_id}/approve` | 도서관 승격 승인 |
| POST | `/api/warehouse/items/{item_id}/promotion-dry-run` | 승격 dry-run |
| POST | `/api/warehouse/items/{item_id}/promote` | 실제 승격 |
| GET | `/api/warehouse/traces/{trace_id}` | promotion trace 조회 |
| POST | `/api/warehouse/validate` | 창고 또는 item 검증 |
| POST | `/api/warehouse/backup/run` | 백업 |
| POST | `/api/warehouse/backup/restore-dry-run/{backup_id}` | 복원 dry-run |
| GET | `/api/warehouse/release-board` | release board 조회 |
| POST | `/api/warehouse/release-board/update` | release board 갱신 |

---

## 6. 실제 QualiLibrary Ripple 연동

운영 환경 변수:

```powershell
$env:QUALI_LIBRARY_RIPPLE_ENABLED="1"
$env:QUALI_LIBRARY_RIPPLE_PYTHONPATH="H:\장기기억\_tmp\qualilibrary_ripple\work\qualilibrary_ripple_integrated_extensible_v0.3.2"
$env:QUALI_LIBRARY_LTM_ROOT="H:\장기기억"
$env:LTM_ROOT="H:\장기기억"
```

dry-run 명령 패턴:

```powershell
python -m qualilibrary_ripple add <raw_path> QLIB <doc_code> v1 2026 --source-lang EN --doc-kind REFERENCE --title-en "<title>" --dry-run --no-db --no-map --no-ripple
```

promote 명령 패턴:

```powershell
python -m qualilibrary_ripple add <raw_path> QLIB <doc_code> v1 2026 --source-lang EN --doc-kind REFERENCE --title-en "<title>"
python -m qualilibrary_ripple verify
python -m qualilibrary_ripple ripple rebuild
python -m qualilibrary_ripple ripple search "<query>" --k 5
python -m qualilibrary_ripple show "QLIB:<doc_code>@v1"
```

필수 산출물:

| 산출물 | 경로 패턴 |
|---|---|
| brain DB | `H:\장기기억\brain.db` |
| graph DB | `H:\장기기억\graph.db` |
| raw | `H:\장기기억\LIBRARY\raw\*.txt` |
| template | `H:\장기기억\LIBRARY\templates\*.yml` |
| reference card | `H:\장기기억\LIBRARY\exports\reference_cards\*.md` |
| ripple index | `H:\장기기억\LIBRARY\ripple\ripple_index.sqlite` |

---

## 7. 테스트 결과

문법 검사:

```powershell
python -m py_compile admin\warehouse_core.py server_quali.py admin\server_quali.py
```

결과:

```text
PASS
```

pytest:

```powershell
python -m pytest -q admin\tests\test_warehouse_core_api.py --basetemp .pytest_tmp_warehouse_run2
```

결과:

```text
4 passed, 1 warning in 14.99s
```

테스트 목록:

| 테스트 | 판정 | 목적 |
|---|---|---|
| `test_warehouse_full_simulation_flow` | PASS | 창고 내부 full simulation |
| `test_warehouse_promote_calls_real_qualilibrary_ripple` | PASS | 실제 zip 기반 Ripple add/verify/rebuild/search/show |
| `test_warehouse_blocks_unknown_rights_approval` | PASS | rights unknown 승인 차단 |
| `test_warehouse_blocks_invalid_state_transition` | PASS | 잘못된 상태 전이 차단 |

warning:

```text
PendingDeprecationWarning: Please use import python_multipart instead.
```

기능 실패가 아니라 외부 패키지 deprecation warning이다.

---

## 8. 운영 루트 최종 스모크

운영 루트:

```text
H:\장기기억
```

쓰기 권한:

```text
write_probe=PASS
probe_hash=19EF91F6B27BF67FE8D3B7CB7AEFD39111175F513F527E5A67412F84F3599811
removed=True
```

기존 도서관 포인터 검증:

```powershell
python -m qualilibrary_ripple verify
```

결과:

```text
[OK] Library pointers look clean
```

운영 스모크 주요 값:

| 항목 | 값 |
|---|---|
| warehouse item id | `WHI-20260514-041109-7D75` |
| promotion trace id | `PTR-20260514-041132-CD92` |
| backup id | `BAK-20260514-041132-F446` |
| memory item id | `58` |
| library node id | `QLIB:warehouse-prod-smoke-20260514041108@v1` |
| run root | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108` |
| final decision | PASS |

운영 스모크 결과:

| 항목 | 결과 |
|---|---|
| dry-run | PASS |
| library engine | PASS |
| add | PASS |
| verify | PASS |
| ripple rebuild | PASS |
| ripple search | PASS |
| ripple search hit count | 1 |
| 창고 validate | PASS |
| release board | PASS |
| backup/restore dry-run | PASS |

---

## 9. 운영 산출물 해시

| 산출물 | 경로 | 크기 | SHA256 |
|---|---|---:|---|
| brain DB | `H:\장기기억\brain.db` | 1,581,056 | `2D6D848D7B6E76C398135B74F50639AA3E4B07E74C51668F70FFC6BCECFC814E` |
| graph DB | `H:\장기기억\graph.db` | 143,360 | `AE6B4FDD12EE9F78281B79625348D6FD3441E4A7470B734B5CFC8FA229D20285` |
| raw | `H:\장기기억\LIBRARY\raw\QLIB__warehouse-prod-smoke-20260514041108__v1__2026__EN__REFERENCE__v1.txt` | 192 | `1B172FE38935E6061E8FC38ADAF69182FB41D634A1B2471BA1883081E390472A` |
| template | `H:\장기기억\LIBRARY\templates\library_item.QLIB__warehouse-prod-smoke-20260514041108__v1__2026__EN__REFERENCE__v1.txt.yml` | 812 | `192E8CB3EEB5F881C0D7E1E6426FB37947D561903DCC7D11388397BFA6386A5E` |
| reference card | `H:\장기기억\LIBRARY\exports\reference_cards\QLIB__warehouse-prod-smoke-20260514041108__v1.md` | 1,238 | `8CA2C3E26B9A96A4A62A88C8834DF1559AC12C0D73291FF174436C6A42777168` |
| ripple index | `H:\장기기억\LIBRARY\ripple\ripple_index.sqlite` | 1,802,240 | `96BFBB2442831714C68C8F1F0ED2E40AD3990D075A40FA3EE4AB19B685A26198` |

---

## 10. ProofPack Index

| 구성요소 | 상태 | 경로 | SHA256 | 역할 | Gate |
|---|---|---|---|---|---|
| Operational smoke summary | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\OPERATIONAL_SMOKE_RESULT.json` | `34B9E08FFA793EFFE19C9FD132D7684958AEEC86DD407F98C5EC1046CCD59E8B` | 운영 스모크 최종 요약 | W-G1~W-G9 |
| Promotion ProofPack | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\reports\proofpacks\warehouse\PTR-20260514-041132-CD92.json` | `98D19FDC6328D8223DE947DB3AEB388161C858FF2B8DAB4178B094C339AE5E39` | dry-run, promote, gate 결과 묶음 | W-G6 |
| Validator / item history | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\data\warehouse\warehouse_items.jsonl` | `C597FD896C9B336A134CD261BA65E7A830E1AB8006F78F68FCC3C9896839EFA8` | item 상태 전이, review, approval, promotion 기록 | W-G2~W-G6 |
| Promotion trace index | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\data\warehouse\trace\promotion_trace.jsonl` | `5C2790098AFD080541E4445BF4C5A3B90259CF1D99C382F0B79179E05D61A421` | warehouse item과 library node 연결 증거 | W-G6 |
| Dry-run result | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\data\warehouse\trace\dry_run\DRY-20260514-041109-63E3.json` | `380BC3EF5C704A66EA286B306509CD350873A963579CBE2A94DA9780360185C8` | promote 전 dry-run 증거 | W-G6 |
| Review record | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\data\warehouse\review\REV-20260514-041109-70F2.json` | `F286B21B31E8FC967DBE024D22AD7C2EBAE5DCD5D60BF511828A8982277985A4` | 사람 리뷰, 품질 점수, confidence 증거 | W-G4 |
| Warehouse manifest | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\data\warehouse\warehouse_manifest.json` | `032B3FBA92C5A14727FC395118B5020C4485268FCC8082606CAB98E08630517E` | 창고 manifest와 root/index 기준 | W-G1 |
| Backup manifest | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\backup\warehouse\BAK-20260514-041132-F446\backup_manifest.json` | `91B3C67474C650BEADF7BD017508B5794BD24D344CAD1298C1DA4DA589C65F8A` | 백업 범위, 파일 수, restore dry-run 증거 | W-G7 |
| Release board | PASS | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\releases\warehouse\release_board.json` | `2B6EE2BEC920F3BEDBAE6C5300D32FF35493771E8F71504E4A5739EA0FB7C983` | 릴리즈 판정, gate, rollback, 승인자 증거 | W-G9 |
| Handover report | PASS | `H:\a\퀄리저널_pr_clean\reports\WAREHOUSE_DEVELOPMENT_COMPLETION_REPORT_20260514.md` | self-referential, final hash reported outside document | 현재 완료 보고서가 인수인계 보고서 역할을 겸함 | HANDOVER |
| 캡처 PNG | NOT_EXECUTED | N/A | N/A | UI가 이번 범위 밖이라 화면 캡처 없음 | UI=NOT_EXECUTED |

---

## 11. Release Board 요약

Release Board 정합성 보정 완료:

| 필드 | 값 |
|---|---|
| `release_id` | `REL-20260514-041132-502F` |
| `date` | `2026-05-14T04:11:32Z` |
| `scope` | `warehouse operational production-root smoke` |
| `decision` | `PASS` |
| `changed_files` | `admin/warehouse_core.py`; `admin/tests/test_warehouse_core_api.py`; `server_quali.py — IN_SCOPE_PARTIAL, warehouse router 연결부`; `admin/server_quali.py — IN_SCOPE_PARTIAL, warehouse router 연결부, 기존 dirty diff와 분리 확인 필요` |
| `validators` | `validate_manifest`, `validate_item_schema`, `validate_raw_hash`, `validate_provenance`, `validate_rights`, `validate_security_scan` |
| `gate_results` | `W-G1`~`W-G9` 모두 `PASS` |
| `test_results` | `production_root_promote_smoke=PASS` |
| `backup_id` | `BAK-20260514-041132-F446` |
| `rollback_plan` | smoke record는 trace evidence로 유지. 롤백은 운영자 승인 후 QualiLibrary DB와 LIBRARY artifacts 제거 필요 |
| `approver` | `codex-operator` |
| `handover_path` | `H:\a\퀄리저널_pr_clean\reports\WAREHOUSE_DEVELOPMENT_COMPLETION_REPORT_20260514.md` |
| `handover_status` | `REPORT_INTEGRATED` |

이 조정으로 `changed_files`와 `IN_SCOPE_FILES`가 서로 맞고, `Handover report NOT_CREATED`와 `handover_path` 사이의 모순도 해소되었다.

---

## 12. Backup/Restore Manifest 요약

| 필드 | 값 |
|---|---|
| `backup_id` | `BAK-20260514-041132-F446` |
| `backup_date` | `2026-05-14T04:11:32Z` |
| `backup_scope` | `full_warehouse` |
| `backup_path` | `H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\backup\warehouse\BAK-20260514-041132-F446` |
| `included_roots` | warehouse root, proofpack root, release root |
| `file_count` | `9` |
| `total_bytes` | `54252` |
| `sha256_manifest` | `sha256:b6c4f963da6b300368081d85f1b278e99d2fe0dc0ae74ca0a793489835093fee` |
| `restore_dry_run_required` | `true` |
| `restore_dry_run_pass` | `true` |
| `restore_dry_run_issues` | `[]` |

---

## 13. Security / Quality / Contract

### Security Scan

| 항목 | 결과 |
|---|---|
| `SECURITY_SCAN` | PASS |
| 검사 대상 | raw text, item metadata, validator, W-G8 gate, release validators |
| secret/token/API key 발견 | 0 |
| private key 패턴 발견 | 0 |
| `sensitivity` | `internal` |
| `visibility` | `library_candidate` |
| no_export 여부 | false |
| W-G8 | PASS |
| 예외 | 없음 |

내부 경로는 proofpack과 운영 보고서 증거용으로만 기록한다. 공개 UI, 학생 화면, 공개 리포트에는 표시하면 안 된다.

### Quality Threshold / LOW_CONFIDENCE

| 항목 | 기준 | 운영 스모크 값 | 판정 |
|---|---:|---:|---|
| `quality_score` | 80 이상 | 95 | PASS |
| `confidence_score` | 0.85 이상 | 0.98 | PASS |
| `confidence` | high 권장 | high | PASS |
| `LOW_CONFIDENCE` 사람 리뷰 강제 | 0.85 미만일 때 적용 | 해당 없음 | NOT_IN_SCOPE |
| review decision | approved_for_library | approved_for_library | PASS |

근거:

```text
H:\a\퀄리저널_pr_clean\reports\warehouse_operational_smoke\run_20260514041108\data\warehouse\review\REV-20260514-041109-70F2.json
```

### QLIB Contract Test 범위

| 계약 | 상태 | 근거 |
|---|---|---|
| `CT-WH-001` Warehouse manifest/item schema/raw/provenance/rights 기본 계약 | PASS | `test_warehouse_full_simulation_flow`, `validate_warehouse`, manifest/item validators |
| `CT-WH-002` Warehouse approval/promotion trace/library ripple integration 계약 | PASS | `test_warehouse_promote_calls_real_qualilibrary_ripple`, 운영 root promote smoke |

미범위:

| 계약 범위 | 상태 | 이유 |
|---|---|---|
| Bridge contract runtime | NOT_EXECUTED | Bridge Runtime 미구현 |
| Skillup module/binding/answer flow | NOT_EXECUTED | Skillup Runtime 미구현 |
| Analytics governance contract | NOT_EXECUTED | Analytics 모듈 미구현 |
| Integrated UI customer view leak test | NOT_EXECUTED | UI 미구현 |

---

## 14. 작성 시점 실행 증거와 향후 재검증

작성 시점 실행 증거:

| 항목 | 실행 여부 | 결과 |
|---|---|---|
| Python syntax check | EXECUTED | PASS |
| Warehouse API pytest | EXECUTED | `4 passed, 1 warning` |
| real zip based qualilibrary_ripple integration test | EXECUTED | PASS |
| `H:\장기기억` write probe | EXECUTED | PASS |
| `qualilibrary_ripple verify` | EXECUTED | PASS |
| production-root warehouse promote smoke | EXECUTED | PASS |
| `ripple search` production smoke node | EXECUTED | hit 1 |
| UI screenshot validation | NOT_EXECUTED | UI 미구현 |
| Bridge runtime E2E | NOT_EXECUTED | Bridge Runtime 미구현 |
| Skillup answer E2E | NOT_EXECUTED | Skillup Runtime 미구현 |

향후 재검증 명령:

```powershell
cd H:\a\퀄리저널_pr_clean
python -m py_compile admin\warehouse_core.py server_quali.py admin\server_quali.py
python -m pytest -q admin\tests\test_warehouse_core_api.py --basetemp .pytest_tmp_warehouse_verify

$env:LTM_ROOT='H:\장기기억'
$env:PYTHONPATH='H:\장기기억\_tmp\qualilibrary_ripple\work\qualilibrary_ripple_integrated_extensible_v0.3.2'
python -m qualilibrary_ripple verify
python -m qualilibrary_ripple show QLIB:warehouse-prod-smoke-20260514041108@v1
python -m qualilibrary_ripple ripple search "Warehouse operational smoke 20260514041108" --k 3
```

---

## 15. UI와 Skillup 상태

UI는 가이드에 적시되어 있으나 개발 완료가 아니다.

| 영역 | 상태 |
|---|---|
| Warehouse UI | NOT_EXECUTED |
| Library UI | NOT_EXECUTED |
| Bridge UI | NOT_EXECUTED |
| Skillup UI | NOT_EXECUTED |
| Integrated Admin UI Shell | NOT_EXECUTED |

Skillup 관점에서 완성된 것은 **정본 지식 공급 경로**다.

```text
Warehouse -> Review -> Approval -> QualiLibrary Ripple -> Evidence/Index
```

아직 남은 것은 다음이다.

| 항목 | 상태 |
|---|---|
| Bridge Contract Runtime | NOT_EXECUTED |
| course_library_binding | NOT_EXECUTED |
| module_manifest 검증 | NOT_EXECUTED |
| Skillup evidence answer/HOLD flow | NOT_EXECUTED |
| 학생/강사/검토자/관리자 UI | NOT_EXECUTED |
| Analytics | NOT_EXECUTED |

---

## 16. Dirty Worktree 분리

현재 관찰된 dirty worktree에는 이번 작업 범위와 범위 밖 변경이 섞여 있다.

### IN_SCOPE_FILES

| 경로 | 상태 | 처리 |
|---|---|---|
| `admin/warehouse_core.py` | IN_SCOPE | 커밋 대상 |
| `admin/tests/test_warehouse_core_api.py` | IN_SCOPE | 커밋 대상 |
| `server_quali.py` | IN_SCOPE_PARTIAL | warehouse router 연결부 확인 후 커밋 대상 |
| `admin/server_quali.py` | IN_SCOPE_PARTIAL | warehouse router 연결부만 확인. 기존 dirty diff 분리 필요 |
| `reports/warehouse_operational_smoke/run_20260514041108/` | IN_SCOPE_EVIDENCE | 보존 대상 |
| `reports/WAREHOUSE_DEVELOPMENT_COMPLETION_REPORT_20260514.md` | IN_SCOPE_REPORT | 보존/커밋 대상 |

### OUT_OF_SCOPE_DIRTY_FILES

| 경로 | 상태 | 처리 |
|---|---|---|
| `.github/PULL_REQUEST_TEMPLATE.md` | OUT_OF_SCOPE_DIRTY | 임의 revert 금지 |
| `docs/commit_message_template_with_constitution_trace.txt` | OUT_OF_SCOPE_DIRTY | 임의 revert 금지 |
| `admin/body.json` | OUT_OF_SCOPE_UNTRACKED | 사용자 확인 전 삭제 금지 |
| `admin/body_approve.json` | OUT_OF_SCOPE_UNTRACKED | 사용자 확인 전 삭제 금지 |
| `admin/body_link.json` | OUT_OF_SCOPE_UNTRACKED | 사용자 확인 전 삭제 금지 |
| `admin/body_promote.json` | OUT_OF_SCOPE_UNTRACKED | 사용자 확인 전 삭제 금지 |
| `admin/body_record.json` | OUT_OF_SCOPE_UNTRACKED | 사용자 확인 전 삭제 금지 |
| `admin/data/` | OUT_OF_SCOPE_UNTRACKED | 사용자 확인 전 삭제 금지 |
| `commit_message_template_with_constitution_trace.txtq` | OUT_OF_SCOPE_UNTRACKED | 출처 확인 전 삭제 금지 |

커밋 전 처리 지시:

1. `git diff -- admin/warehouse_core.py admin/tests/test_warehouse_core_api.py server_quali.py admin/server_quali.py` 확인
2. `admin/server_quali.py`는 기존 dirty diff가 크므로 라인 단위 확인
3. `OUT_OF_SCOPE_DIRTY_FILES`는 사용자 확인 전 revert/delete 금지
4. 커밋 메시지에 `SCOPE=WAREHOUSE_BACKEND_API_AND_LIBRARY_RIPPLE_INTEGRATION` 명시

---

## 17. 운영 주의사항

스모크 레코드는 trace 증거로 남긴다. 삭제나 롤백은 별도 승인 작업으로 처리해야 한다.

주의 대상:

```text
H:\장기기억\LIBRARY\raw\QLIB__warehouse-prod-smoke-20260514041108__v1__2026__EN__REFERENCE__v1.txt
H:\장기기억\LIBRARY\templates\library_item.QLIB__warehouse-prod-smoke-20260514041108__v1__2026__EN__REFERENCE__v1.txt.yml
H:\장기기억\LIBRARY\exports\reference_cards\QLIB__warehouse-prod-smoke-20260514041108__v1.md
H:\장기기억\brain.db
H:\장기기억\graph.db
H:\장기기억\LIBRARY\ripple\ripple_index.sqlite
```

DB와 index에는 다른 운영 데이터도 포함되므로 단순 파일 삭제로 롤백하면 안 된다.

---

## 18. 다음 권장 작업

1. `QLIB Integrated UI Shell + Warehouse/Library/Bridge/Skillup 화면 설계 및 구현`
2. `Library Evidence Bridge Contract Runtime 구현`
3. `QLIB Skillup Education MVP 구현`

다음 작업은 UI/Bridge/Skillup 범위이므로 새 release board와 새 proofpack이 필요하다.

---

## 19. Self-Check

| 항목 | 결과 |
|---|---|
| 최종 판정 범위 고정 | PASS |
| `OVERALL_QLIB_PRODUCT_DONE=NO` 명시 | PASS |
| UI/Bridge/Skillup/Analytics NOT_EXECUTED 분리 | PASS |
| Release Board changed_files와 IN_SCOPE 정합성 | PASS |
| Handover report 모순 해소 | PASS |
| ProofPack Index 추가 | PASS |
| Release Board 필드 요약 | PASS |
| Backup/Restore manifest 요약 | PASS |
| Security Scan 결과 명시 | PASS |
| Quality Threshold 증거 명시 | PASS |
| Dirty worktree 범위 분리 | PASS |
| 작성 시점 실행 증거와 향후 재검증 절차 분리 | PASS |
| QLIB Contract Test 범위 명시 | PASS |

---

## 20. 최종 선언

```text
FINAL_DECISION=PASS
SCOPE=WAREHOUSE_BACKEND_API_AND_LIBRARY_RIPPLE_INTEGRATION

PASS 대상:
- Warehouse backend/API
- review/approval/dry-run/promote flow
- QualiLibrary Ripple 실제 연동
- production-root smoke
- ProofPack / Release Board / Backup-Restore dry-run

NOT_EXECUTED:
- Warehouse UI
- Library UI
- Bridge Runtime
- Skillup Education Runtime
- Analytics
- Integrated UI Shell

OVERALL_QLIB_PRODUCT_DONE=NO
```

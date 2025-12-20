<!-- SSOT-AUTOHEAL-ONEPAGER START -->
# SSOT/헌법/Runbook/AGENTS 자가치유 체크리스트 (1장 고정)

## 0) 변경 요약 (한 문장)
- 

## 1) 변경 유형 (하나 이상 체크)
- [ ] 문서(SSOT/헌법/Runbook/AGENTS)만
- [ ] 코드만
- [ ] 운영/배포(CI/Cloud Run/Scheduler/Secret/IAM/Traffic)만
- [ ] 혼합(문서+코드+운영)

## 2) 영향 범위 (체크)
- [ ] admin/
- [ ] tools/
- [ ] feeds/
- [ ] .github/workflows/
- [ ] docs/
- [ ] docs/infra/ (인프라 정의/배포 자산)

## 3) “문서→코드→AGENTS” 동시 갱신 (자가치유 핵심)
- [ ] (문서) SSOT/DoD 업데이트: `docs/SSOT_Admin_DoD.md`
- [ ] (문서) Runbook/헌법 업데이트: `docs/QualiJournal_Constitution_C_Operations_Runbook_Gates_FINAL_*.md`
- [ ] (문서) AGENTS 업데이트: `docs/AGENTS.md` 또는 `.github/workflows/AGENTS.md`
- [ ] (코드/운영) 변경이 있다면, 위 문서 중 최소 1개는 같은 PR에서 함께 갱신했다

## 4) 검증 증거 (3줄 스모크 기록)
- health: [ ] 200 OK
- api/status: [ ] 200 OK(토큰) / [ ] 401~403(무토큰 기대동작)
- api/report: [ ] 200 OK(토큰)

## 5) 롤백 플랜(필수, 1~2줄)
- 

<!-- SSOT-AUTOHEAL-ONEPAGER END -->

---


<!-- QualiJournal PR Template v1 – Branch/Test-Branch Constitution Aware -->

## 1. 브랜치 & 변경 정보

- base 브랜치:
  - [ ] main  
  - [ ] main-signed-ssot  
  - [ ] ssot-ready-fix-YYYYMMDD  
  - [ ] domap-deploy-ssot  
  - [ ] docs/runbook(-v2)  
  - [ ] 기타: ________________________

- head 브랜치:
  - [ ] feature/test-... (테스트/실험 브랜치)  
  - [ ] fix/...  
  - [ ] chore/...  
  - [ ] docs/...  
  - [ ] 기타: ________________________

- 변경 유형(복수 선택 가능):
  - [ ] 코드 기능 변경 (Admin API / domap / 표준 큐레이션 로직 등)
  - [ ] CI/DevOps 설정 변경 (ci-test-admin, ci-deploy-prod.yml 등)
  - [ ] SSOT/DoD 문서 변경 (SSOT_*.md, Admin_DoD_*.md 등)
  - [ ] Runbook/헌법 문서 변경 (branch-constitution-*.md, TestBranch 헌법 등)
  - [ ] 기타: ________________________

---

## 2. 브랜치 헌법 체크리스트 (공통)

- [ ] 이 PR이 대상으로 하는 브랜치가  
      `docs/branch-constitution-QualiJournal-20251125.md` 에 정의된 유형  
      (Base / Signed-SSOT / Release / Deploy / Tools / Test / Docs) 중 어디에 속하는지 확인했다.
- [ ] main / main-signed-ssot / ssot-ready-fix-* / domap-deploy-ssot 에 대해서는  
      **직접 push 하지 않고 PR로만 변경**했다.
- [ ] 위 브랜치들에 대해 **force push / delete** 를 시도하지 않았다.
- [ ] 실험/테스트 브랜치(feature/test-*, ssot-guard-*)에서 나온 변경은  
      필요 부분만 정리해서 대상 브랜치로 옮겼고,  
      실험 브랜치는 나중에 Runbook/헌법에 정리 후 정리(삭제) 계획이 있다.

---

## 3. 테스트 브랜치인 경우 (head 가 feature/test-* 인 경우)

> 해당 없으면 이 섹션은 건너뛰어도 된다.

- [ ] 이 브랜치는  
      **“한 실험 = 한 브랜치 = 한 파일 = 한 목적”** 원칙을 따른다.  
      (예: ci-deploy-prod.yml 한 파일만 수정하는 실험이라면, 다른 파일 diff 가 끼어 있지 않음)
- [ ] 이 브랜치에서 `git merge`, `git rebase` 를 사용하지 않았다.  
      (예외가 있었다면, 대상 파일 목록·백업·승인 기록을 PR 설명에 적었다.)
- [ ] 이 브랜치에서 `git reset --hard`, `git clean -fdx` 등  
      위험 명령을 사용하지 않았다.
- [ ] PR Checks 가 0, Actions 에서 Run workflow 버튼이 없거나,  
      예상 밖의 파일이 diff 에 섞여 나오는 등 **이상 징후가 보였을 때**  
      실험을 멈추고 캡처·메모·회고를 남겼다 (또는 지금 PR 설명에 정리했다).

---

## 4. SSOT/DoD 문서 변경인 경우

> SSOT 문서 예: `SSOT_*.md`, `Admin_DoD_*.md`, 통합 헌법 v2, Admin DoD SSOT 등

- [ ] 이 PR에서 SSOT/DoD 문서(SSOT_*.md, Admin_DoD_*.md, 통합 헌법 등)를 변경했다면,  
      PR에 **라벨 `ssot-change`** 를 추가했다.
- [ ] `.github/workflows/ssot-check.yml` 의 **SSOT Check 워크플로**가  
      성공으로 끝났는지 확인했다.
- [ ] 변경한 내용이  
      통합 헌법 / Admin DoD SSOT / Ground Truth 문서(1115 정의서 등)와  
      서로 모순되지 않는지 검토했다.

---

## 5. Admin Tests / DevOps 품질 게이트

- [ ] 로컬 또는 CI에서 Admin Tests(`ci-test-admin` 관련 워크플로)를 실행했고,  
      이 PR 변경으로 인한 실패가 없는 것을 확인했다.
- [ ] main / main-signed-ssot 브랜치 보호 설정이  
      SSOT에서 정의한 값(Require PR / Require status checks / Require signed commits / Require linear history)을 그대로 유지하도록 PR을 구성했다.
- [ ] 이 PR이 병합되었을 때,  
      `/api/status`, `/api/ready` 등의 Admin API DoD(응답 스키마, HTTP 코드, 필드 기준)를 깨지 않는다고 판단했다.

---

## 6. 변경 요약 & 롤백 플랜

- 변경 요약 (1~3줄):

  > 예) “ci-deploy-prod.yml 의 리뷰 게이트 스텝을 A1 런북 기준으로 정리했음.  
  >      Admin Tests/SSOT Check 를 통과했고, domap-deploy-ssot 에는 영향 없음.”

- 롤백 방법(필요 시):

  > 예) “문제 발생 시 `main-signed-ssot` 기준으로 새 브랜치 생성 후,  
  >      이 PR의 변경 파일만 부분 revert 한다.”

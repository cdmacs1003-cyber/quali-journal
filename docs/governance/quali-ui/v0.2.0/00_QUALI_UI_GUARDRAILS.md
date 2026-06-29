# Quali UI Guardrails

> 파일명: `00_QUALI_UI_GUARDRAILS.md`  
> 프로젝트: Quali UI System  
> 버전: v0.2.0  
> 상태: Active  
> 기준일: 2026-06-29  
> 우선순위: Constitution 다음, 구현 전 필수 적용

---

## 1. 목적

이 문서는 퀄리 UI 결과물이 헌법을 벗어나지 않도록 막는 가드레일이다.  
디자인, 문구, 정보 구조, 접근성, 컴포넌트, 산출물 검사를 모두 포함한다.

---

## 2. 절대 금지

아래 항목은 승인 없이 사용할 수 없다.

### 2.1 디자인 금지

- 의미 없는 그라데이션 남발
- 장식용 이모지 중심 레이아웃
- 무지개색 팔레트
- 페이지별 임의 HEX 색상 추가
- 장식형 폰트 사용
- 과한 그림자와 광택 효과
- 카드만 나열하는 자료실형 UI
- 주요 버튼을 한 화면에 여러 개 노출
- 좁은 여백으로 정보를 빽빽하게 배치
- 한글 대형 제목을 브라우저 자동 줄바꿈에 방치
- 조사, 접미사, 핵심 명사가 단독 줄로 떨어지는 제목
- 핵심 용어가 의미 없이 분리되는 행갈이
- 최종 Line Break Gate 없이 긴 한글 제목을 배포

### 2.2 표준·근거 금지

- 표준번호 없는 판정
- 개정·버전 없는 표준 표시
- 근거 없는 요약
- 출처 없는 “검증 완료” 문구
- 추정과 문서 근거를 구분하지 않는 설명
- 내부 기준과 국제 표준을 같은 레벨로 혼합 표기

### 2.3 접근성 금지

- 색상만으로 PASS/HOLD/FAIL 전달
- 아이콘만 있는 버튼에 `aria-label` 누락
- 검색 입력에 시각 label 누락
- 포커스 링 제거
- 대비 미달 텍스트 사용
- 테이블 caption 또는 header 구조 누락

### 2.4 운영 금지

- Memory 기록 없이 재사용 가능한 새 규칙 추가
- Guidebook 업데이트 없이 새 공통 컴포넌트 추가
- HOLD 항목을 PASS처럼 표현
- 검증하지 않은 결과를 완료로 보고
- 브랜드명 임의 변경
- 문서 변경 후 `SHA256SUMS.txt` 미갱신
- 상태·토큰·컴포넌트 기준을 상위 문서와 다르게 기록
- Evidence Schema 필수 필드 없이 근거 데이터를 재사용

---

## 3. 필수 포함 요소

모든 주요 화면은 아래 컴포넌트 중 최소 5개 이상을 포함한다.

| 컴포넌트 | 필수 상황 |
|---|---|
| Trust Banner | 공식성·신뢰성·검증성을 보여야 할 때 |
| Standard Context Bar | 표준·기준 기반 작업 전체 |
| Standard Code Badge | 표준번호·개정·Class를 짧게 식별할 때 |
| Workbench Search | 문서·표준·용어·고객 질문 검색 |
| Module Card | 제품·모듈 진입 또는 선택 화면 |
| Status Badge | 상태 표시 전체 |
| Evidence Box | 판정·요약·보고 전체 |
| Trace Table | 입력과 기준, 판정, 근거를 연결할 때 |
| Beginner Box | 교육·고객 설명·초보자 화면 |
| Expert Box | 표준 조항·전문가 검토 화면 |
| Action Checklist | 다음 행동이 필요한 화면 |
| Output Gate | 출력·제출·보고서 생성 직전 |
| Line Break Gate | 한글 제목·핵심 문장 최종 검사 |

---

## 4. 색상 가드레일

### 4.1 승인 색상

```css
:root {
  --q-navy-900: #061F4A;
  --q-blue-700: #0B3D91;
  --q-blue-500: #105BD8;

  --q-text-900: #212121;
  --q-text-700: #323A45;
  --q-text-500: #64748B;

  --q-bg: #F8FAFC;
  --q-panel: #FFFFFF;
  --q-line: #D9E2EC;

  --q-pass: #15803D;
  --q-hold: #92400E;
  --q-fail: #B91C1C;
  --q-focus: #105BD8;
}
```

### 4.2 사용 규칙

| 토큰 | 허용 용도 | 금지 용도 |
|---|---|---|
| `--q-navy-900` | 헤더, 신뢰 배너, 기준 영역 | 장식 배경 반복 |
| `--q-blue-700` | 링크, 기준 강조, 선택 상태 | 경고·오류 표시 |
| `--q-blue-500` | 주요 액션 버튼 | 여러 버튼 동시 남발 |
| `--q-pass` | PASS, 적합, 완료 | 긍정 홍보 문구 |
| `--q-hold` | HOLD, 확인 필요, 보류 | 일반 강조 |
| `--q-fail` | FAIL, 위험, 삭제, 차단 | 장식 강조 |

### 4.3 대비 기준

- 일반 텍스트: 최소 4.5:1
- 큰 텍스트: 최소 3:1
- UI 경계, 포커스, 비텍스트 요소: 최소 3:1
- 색상 단독 전달 금지

---

## 5. 폰트 가드레일

```css
:root {
  --q-font-main: "Noto Sans KR", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  --q-font-code: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
```

| 용도 | 허용 |
|---|---|
| 제목 | Noto Sans KR, 굵기 700~800 |
| 본문 | Noto Sans KR, 16~17px |
| 표준번호 | monospace, 13~14px |
| 파일명·경로 | monospace |
| 장식 폰트 | 금지 |

---


## 6. 한글 타이포그래피와 행갈이 가드레일

한글 화면은 브라우저 기본 줄바꿈에 맡기지 않는다.  
퀄리 UI는 표준·품질·교육·검증을 다루므로, 대형 제목과 핵심 문장이 조사·접미사·핵심 명사 단위로 어색하게 끊기면 미완성 UI로 판정한다.

### 6.1 기본 원칙

1. 한글 대형 제목은 브라우저 자동 줄바꿈에 맡기지 않는다.
2. H1/H2는 의미 단위로 직접 행을 나누고, 필요한 경우 `span.line`을 사용한다.
3. 조사, 접미사, 핵심 명사가 단독 줄로 떨어지면 FAIL이다.
4. 제품명, 표준명, 핵심 용어는 가능한 한 한 줄 안에서 유지한다.
5. 본문에는 `word-break: keep-all`을 적용한다.
6. 코드, 경로, 표준번호, 긴 식별자는 `overflow-wrap: anywhere`를 허용한다.
7. 대형 제목은 최대 3줄, 섹션 제목은 최대 2줄을 원칙으로 한다.
8. 최종 출력 전 Line Break Gate를 수행한다.

### 6.2 보호 용어

아래 용어는 임의 분리하지 않는다.

| 범주 | 보호 용어 예시 |
|---|---|
| 공식 제품명 | 퀄리창고, 퀄리도서관, 브릿지, 스킬업 |
| 우주·표준 용어 | 우주 임무보증연구회, 임무보증, 뉴 스페이스, 리스크 구조, 핵심 원자료, 시각 요약 |
| 표준·기술 용어 | K-MAR, COTS-EEE, NASA·ESA·JAXA, IPC-A-610, J-STD-001, IPC/WHMA-A-620 |
| 상태·판정 용어 | PASS, HOLD, FAIL, REVIEWED, READY, DRAFT, Output Gate, Line Break Gate |

### 6.3 승인 CSS 패턴

```css
html,
body {
  word-break: keep-all;
  overflow-wrap: break-word;
}

p,
li,
td,
th,
.lead,
.card-text,
.section-copy {
  word-break: keep-all;
  overflow-wrap: break-word;
  line-height: 1.72;
}

.display-title,
.section-title,
.ko-heading {
  word-break: keep-all;
  text-wrap: balance;
  line-height: 1.05;
  letter-spacing: 0;
}

.display-title .line,
.section-title .line,
.ko-heading .line {
  display: block;
}

.nowrap,
.term,
.standard-code {
  white-space: nowrap;
}

code,
.path,
.file-name,
.long-id {
  word-break: normal;
  overflow-wrap: anywhere;
}
```

### 6.4 제목 작성 규칙

#### 금지 예

```text
임무보증을 한 문장으
로 정리하면
```

```text
뉴 스페이스 시대의 리
스크 구조
```

```text
핵심 원자료 시각요
약
```

#### 권장 예

```html
<h1 class="display-title">
  <span class="line">우주 임무의 신뢰를</span>
  <span class="line">설계하고 검증하는</span>
  <span class="line">협의체</span>
</h1>
```

```html
<h2 class="section-title">
  <span class="line">뉴 스페이스 시대의</span>
  <span class="line">리스크 구조</span>
</h2>
```

```html
<h2 class="section-title">
  <span class="line">핵심 원자료</span>
  <span class="line">시각 요약</span>
</h2>
```

### 6.5 Line Break Gate

최종 출력 전 아래 항목을 검사한다.

| 검사 | PASS 조건 | 실패 시 판정 |
|---|---|---|
| H1 행갈이 | 의미 단위 2~3줄 | HOLD 또는 FAIL |
| H2 행갈이 | 의미 단위 1~2줄 | HOLD |
| 조사 분리 | `은/는/이/가/을/를/으로` 단독 줄 없음 | FAIL |
| 핵심어 분리 | 보호 용어가 중간에서 끊기지 않음 | FAIL |
| 본문 줄 길이 | 한글 기준 과도하게 길거나 짧지 않음 | HOLD |
| 모바일 제목 | 4줄 이상 무너지지 않음 | HOLD |
| 최종 인상 | 웹페이지 제작 중단 느낌이 없음 | PASS 필요 |

---

## 7. 정보 구조 가드레일

모든 결과 화면은 아래 질문에 답해야 한다.

| 순서 | 질문 | 누락 시 판정 |
|---:|---|---|
| 1 | 사용자가 무엇을 입력했는가? | HOLD |
| 2 | 어떤 표준·기준을 적용했는가? | FAIL |
| 3 | 판정 또는 결과는 무엇인가? | HOLD |
| 4 | 근거는 어디인가? | FAIL |
| 5 | 다음 조치는 무엇인가? | HOLD |

---

## 8. Evidence Schema 가드레일

판정, 요약, 보고, 교육 피드백, 고객 답변에 쓰이는 근거 데이터는 Evidence Schema의 필수 필드를 가져야 한다.

| 필드 | 목적 | 누락 시 판정 |
|---|---|---|
| `standard_id` | 표준번호 또는 기준 식별 | FAIL |
| `revision` | 개정·버전 | HOLD 또는 FAIL |
| `clause` | 조항·절·근거 위치 | HOLD 또는 FAIL |
| `source_file` | 원본 파일 또는 출처 | FAIL |
| `page` | 페이지·범위·위치 | HOLD |
| `evidence_level` | direct, summarized, inferred, pending | FAIL |
| `verification_status` | PASS, HOLD, FAIL | FAIL |
| `limitation` | 근거 한계·주의사항 | HOLD |
| `next_action` | 다음 확인 또는 조치 | HOLD |

허용 `evidence_level`:

- `direct`: 원문 조항 또는 공식 출처에 직접 근거
- `summarized`: 직접 근거를 요약
- `inferred`: 근거에서 추론했으나 추가 확인 필요
- `pending`: 근거 확보 전, HOLD 상태

---

## 9. 상태 가드레일

### 9.1 허용 상태

| 상태 | 의미 | 색상 |
|---|---|---|
| PASS | 기준 충족, 완료 가능 | Green |
| HOLD | 확인 필요, 보류 | Amber |
| FAIL | 기준 미달, 차단 | Red |
| INFO | 참고 정보 | Blue |
| DRAFT | 초안, 검토 전 | Gray |
| REVIEWED | 검토 완료 | Blue |
| READY | 제출 또는 사용 가능 | Green |
| ARCHIVED | 보관, 비활성 | Gray |

### 9.2 상태 문구 규칙

- 상태는 반드시 텍스트로 표시한다.
- 색상은 보조 신호다.
- HOLD 이유는 한 문장 이상 작성한다.
- FAIL은 차단 조건과 복구 조건을 함께 표시한다.
- REVIEWED는 검토 완료를 뜻하지만, READY 또는 PASS를 대체하지 않는다.

---

## 10. 컴포넌트 생성 가드레일

새 컴포넌트는 아래 조건을 모두 만족할 때만 만든다.

- 기존 컴포넌트 조합으로 해결 불가
- 2개 이상 모듈에서 재사용 가능
- props 정의 가능
- 접근성 요구사항 정의 가능
- Guidebook에 반영 가능
- Memory에 결정 기록 가능

### 새 컴포넌트 제안 양식

```text
컴포넌트명:
목적:
기존 컴포넌트로 해결 불가한 이유:
사용 모듈:
필수 props:
상태 값:
접근성 고려:
Guidebook 반영 항목:
Memory 기록 항목:
```

---

## 11. 문구 가드레일

### 11.1 버튼

버튼은 동사형으로 쓴다.

좋은 예:
- 표준 찾기
- 근거 보기
- 판정표 만들기
- 교육자료 만들기
- 고객 답변 생성

나쁜 예:
- 고도화된 표준 기반 지식 체계
- 더 알아보기
- 시작하기

### 11.2 설명

- 초보자 설명은 2~3문장 이하
- 전문가 근거는 표준번호·조항·개정·상태 포함
- 추정은 반드시 “확인 필요” 표시
- 고객용 문구는 책임 범위를 명확히 표시

---

## 12. 출력 가드레일

모든 산출물 출력 전 아래 Output Gate를 수행한다.

| 검사 | PASS 조건 |
|---|---|
| 목적 | 화면 또는 문서 목적이 한 문장으로 설명됨 |
| 기준 | 표준·규칙·정책이 명시됨 |
| 근거 | Evidence Box 또는 근거 표가 있음 |
| 상태 | PASS/HOLD/FAIL 등 상태 라벨이 있음 |
| 접근성 | 색상 단독 전달 없음, 대비 기준 충족 |
| 토큰 | 승인 색상·폰트 사용 |
| 모듈성 | 기존 컴포넌트 재사용 |
| Memory | 반복 결정 기록됨 |
| 한글 행갈이 | H1/H2/핵심 문장이 의미 단위로 줄바꿈됨 |
| Line Break Gate | 조사·접미사·핵심어 분리 실패 없음 |
| Evidence Schema | 필수 필드와 근거 수준이 충족됨 |
| 문서 정합성 | 상태·토큰·컴포넌트 기준이 상위 문서와 충돌하지 않음 |
| 해시 | 문서 변경 후 `SHA256SUMS.txt`가 갱신됨 |

---

## 13. 자동 검증 가드레일

아래 검사는 v0.3 자동 검증의 기준이며, 실행 가능한 스크립트가 있으면 최종 보고에 결과를 포함한다.

| 검사 | 목적 | 실패 시 판정 |
|---|---|---|
| `check:hash` | `SHA256SUMS.txt` 무결성 확인 | HOLD 또는 FAIL |
| `check:tokens` | 승인되지 않은 HEX·토큰 사용 확인 | FAIL |
| `check:contrast` | 승인 토큰 대비 확인 | FAIL |
| `check:doc-sync` | 상태·토큰·컴포넌트 문서 정합성 확인 | HOLD 또는 FAIL |
| `check:korean-linebreak` | 한글 제목 행갈이 정적 검사 | HOLD |
| `check:evidence-schema` | Evidence Schema 필수 필드 확인 | FAIL |

---

## 14. 실패 복구 규칙

| 실패 유형 | 복구 방법 |
|---|---|
| 근거 없음 | Evidence Box 추가, 근거 수준 HOLD 표시 |
| 다음 행동 불명확 | Action Checklist 추가 |
| 디자인 과잉 | 색상·그림자·카드 수 축소 |
| 카드 남발 | 테이블 또는 패널로 전환 |
| 접근성 실패 | 대비 재계산, 라벨, 포커스 링 추가 |
| 모듈 불일치 | 공통 토큰·컴포넌트로 재구성 |
| 한글 행갈이 실패 | 제목을 `span.line` 기반 의미 단위로 재작성하고 `word-break: keep-all` 적용 |
| 핵심어 분리 | 보호 용어에 `.nowrap` 또는 `.term` 적용 |
| 문서 정합성 실패 | Constitution과 Guardrails 기준으로 하위 문서, Schema, Skill을 동기화 |
| 해시 실패 | 검증 후 `SHA256SUMS.txt` 재생성 |
| Evidence Schema 실패 | 필수 근거 필드 추가 또는 `verification_status: HOLD`로 전환 |

---


## 15. 변경 이력

### v0.2.0 — 2026-06-29

- `REVIEWED` 상태를 Constitution과 정합화
- Evidence Schema 가드레일 추가
- 자동 검증 가드레일과 Output Gate 검사 항목 추가
- 문서 변경 후 해시 갱신 의무 추가
- 한글 제목 자간 정책을 `letter-spacing: 0`으로 정리

### v0.1.1 — 2026-06-29

- 한글 타이포그래피와 행갈이 가드레일 추가
- H1/H2 의미 단위 줄바꿈 원칙 추가
- 보호 용어와 Line Break Gate 추가
- Output Gate에 한글 행갈이 검사 항목 추가
- 실패 복구 규칙에 한글 행갈이 복구 방법 추가

### v0.1.0 — 2026-06-29

- 초기 Guardrails 생성
- 공식 제품명 고정 반영
- 색상, 폰트, 정보 구조, 상태, 컴포넌트, 출력 가드레일 정의

---

## 근거 문서

본 문서 세트는 다음 공식·1차 기준의 공통 패턴을 퀄리 시리즈에 맞게 재구성한다.

- ISO Standards: 표준 카탈로그, 분야 분류, 검색 중심 구조.
- IEC Webstore / IEC Standards: 전기·전자 표준 카테고리와 문서형 제품 구조.
- OSHA SIC Search: 공식성 배너, 코드·키워드 기반 검색 흐름.
- TTA 고객서비스포털 및 표준화 자료: 한국어 업무형 표준·시험·인증 흐름.
- PRI / Nadcap: 항공·방산·우주 공급망 품질, 감사, 인증, 컴플라이언스 메시지.
- ECSS Active Standards: 문서번호, 상태, 표준 분야, 변경 요청 중심 구조.
- NASA Technical Standards: 문서번호, 버전, 제목, 날짜, 재검증일 중심 테이블 구조.
- JAXA Satellite Projects: 개발·운영·종료 등 프로젝트 상태 생애주기 표시.
- NASA Web Design System: 절제된 파랑·무채색 팔레트, 명확한 정보 위계, 여백.
- U.S. Web Design System: 디자인 토큰, 배너, 검색, 테이블, 알림, 버튼 컴포넌트 운영 방식.
- W3C WCAG 2.2: 대비, 색상 단독 전달 금지, 포커스 표시, 비텍스트 대비 기준.
- OpenAI Codex AGENTS.md / Skills documentation: 프로젝트 지시 파일과 Skill 디렉터리 구조.

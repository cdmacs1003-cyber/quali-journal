# PROJECT_DEVELOPMENT_GUIDEBOOK.md

> 파일명: `PROJECT_DEVELOPMENT_GUIDEBOOK.md`  
> 프로젝트: Quali UI System  
> 버전: v0.1.0  
> 상태: Active  
> 기준일: 2026-06-29  
> 목적: 퀄리 UI의 법전, 설계서, 운영 가이드

---

## 1. 문서의 역할

이 문서는 퀄리 UI의 전체 가이드북이다.  
헌법, 가드레일, 제품 구조, 디자인 토큰, 컴포넌트, 페이지 패턴, 개발 절차, 리뷰 체크리스트를 통합한다.

---

## 2. 공식 제품 구조

| 제품 | 공식명 | 역할 | 1차 사용자 |
|---|---|---|---|
| 1 | 퀄리창고 | 원본 자료 저장·버전 추적·메타데이터 관리 | 내부 관리자, 표준 관리자 |
| 2 | 퀄리도서관 | 표준 지식화·용어집·판정 기준·교육 설명 | 강사, 품질 담당자, 학습자 |
| 3 | 브릿지 | 고객 질문·표준 답변·보고서·교육 연결 | 고객, 협력사, 내부 담당자 |
| 4 | 스킬업 | 교육·훈련·실습·복습·평가 | 교육생, 강사, 고객사 관리자 |
| 5 | 표준 모듈 | IPC, ECSS, NASA 등 표준별 작업대 | 전문가, 실무자 |

---

## 3. 정보 구조 원칙

모든 화면은 아래 순서를 기본으로 한다.

```text
1. Trust Banner
2. Standard Context Bar
3. Workbench Search 또는 입력 영역
4. Main Work Area
5. Evidence Area
6. Action Checklist
7. Output Gate
```

모바일에서는 아래 순서로 접는다.

```text
Trust → Context → Search/Input → Main → Evidence → Action → Gate
```

---

## 4. 디자인 토큰

### 4.1 CSS 토큰

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

  --q-font-main: "Noto Sans KR", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  --q-font-code: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;

  --q-fs-h1: 32px;
  --q-fs-h2: 24px;
  --q-fs-h3: 18px;
  --q-fs-body: 17px;
  --q-fs-small: 14px;

  --q-lh-h1: 40px;
  --q-lh-h2: 34px;
  --q-lh-h3: 28px;
  --q-lh-body: 28px;
  --q-lh-small: 21px;

  --q-space-1: 4px;
  --q-space-2: 8px;
  --q-space-3: 16px;
  --q-space-4: 24px;
  --q-space-5: 32px;
  --q-space-6: 48px;

  --q-radius-sm: 8px;
  --q-radius-md: 12px;
  --q-radius-lg: 16px;

  --q-shadow-sm: 0 1px 2px rgba(6,31,74,0.06);
  --q-shadow-md: 0 8px 24px rgba(6,31,74,0.08);
  --q-shadow-lg: 0 16px 40px rgba(6,31,74,0.10);
}
```

### 4.2 타입 스케일

| 용도 | 크기 | 줄간격 | 굵기 |
|---|---:|---:|---:|
| H1 | 32px | 40px | 800 |
| H2 | 24px | 34px | 750 |
| H3 | 18px | 28px | 700 |
| Body | 17px | 28px | 400~500 |
| Small | 14px | 21px | 400~600 |
| Code | 13~14px | 20px | 600 |

---

## 5. 핵심 컴포넌트 명세

### 5.1 Trust Banner

목적: 표준 기반, 근거 추적, 출력 검사, 추측 금지 상태를 상단에서 표시한다.

기본 문구:

```text
표준 기반 · 근거 추적 · 출력 검사 · 추측 금지
```

사용 위치:
- 모든 제품 첫 화면
- 고객용 브릿지
- 보고서 미리보기
- 출력 전 검사 화면

### 5.2 Standard Context Bar

필수 필드:
- 표준번호
- 개정/버전
- 적용 범위
- 상태

예:

```text
기준 표준: IPC-A-610 | 개정: H | 적용: Class 3 | 상태: HOLD | 근거: 조항 매핑 필요
```

### 5.3 Workbench Search

목적: 표준번호, 문서명, 키워드, 고객 질문을 검색한다.

구조:
- label
- search input
- primary submit button
- optional filters
- empty state

### 5.4 Status Badge

허용 상태:
- PASS
- HOLD
- FAIL
- INFO
- DRAFT
- REVIEWED
- READY
- ARCHIVED

상태는 색상과 텍스트로 동시에 표시한다.

### 5.5 Evidence Box

필수 필드:
- 출처
- 표준번호/조항
- 발췌 또는 요약
- 근거 수준
- 확인 필요 여부

### 5.6 Trace Table

기본 열:

| 입력 | 기준 | 판정 | 근거 | 조치 |
|---|---|---|---|---|

표준, 조항, 버전, 판정, 근거 비교는 카드가 아니라 Trace Table을 우선한다.

### 5.7 Beginner Box

목적: 초보자가 바로 이해하도록 설명한다.

문구 규칙:
- 짧게
- 쉬운 단어
- 비유 허용
- 추정은 “확인 필요” 표시

### 5.8 Expert Box

목적: 전문가가 판정 근거를 확인하도록 한다.

필수:
- 표준번호
- 개정
- 조항 또는 근거 위치
- 적용 범위
- 한계 또는 확인 필요

### 5.9 Action Checklist

목적: 다음 행동을 명확히 한다.

예:

```text
[ ] 기준 표준 선택
[ ] 근거 조항 확인
[ ] 판정표 생성
[ ] 고객용 설명 검토
[ ] 출력 게이트 실행
```

### 5.10 Output Gate

출력 전 검사 패널이다.

검사 항목:
- 기준 표시
- 근거 표시
- 상태 표시
- 접근성
- 토큰 준수
- 금지어 또는 정책 위반
- 메타데이터/출처 형식

---

## 6. 제품별 화면 패턴

### 6.1 퀄리창고

```text
Trust Banner
Search / Filter
Source Table
Metadata Panel
Version History
Action Checklist
```

필수 정보:
- 문서명
- 표준번호
- 개정
- 연도
- 출처
- 업로드일
- 상태
- 태그

### 6.2 퀄리도서관

```text
Trust Banner
Standard Context Bar
Workbench Search
Beginner Box
Expert Box
Evidence Box
Trace Table
```

필수 정보:
- 쉬운 설명
- 전문가 근거
- 관련 표준
- 적용 Class
- 주의사항

### 6.3 브릿지

```text
Trust Banner
Customer Question Input
Standard Selector
Decision Result
Evidence Summary
Action Checklist
Output Gate
```

필수 정보:
- 고객 질문
- 적용 기준
- 판정
- 근거 요약
- 다음 조치

### 6.4 스킬업

```text
Learning Goal
Concept Summary
Beginner Explanation
Practice Task
Quiz
Feedback
Progress Status
```

### 6.5 표준 모듈

각 표준 모듈은 같은 뼈대를 공유한다.

```text
표준명
적용 대상
기준 조항
판정 기준
근거
교육 변환
보고서 변환
```

---

## 7. 페이지 패턴

| 패턴 | 용도 | 주 컴포넌트 |
|---|---|---|
| Dashboard | 제품 진입 | Trust Banner, Module Card |
| Search & Filter | 표준/문서 탐색 | Workbench Search, Trace Table |
| Standard Detail | 표준 상세 | Context Bar, Beginner/Expert Box |
| Evidence Review | 근거 검토 | Evidence Box, Trace Table |
| Training Page | 교육 | Beginner Box, Quiz, Checklist |
| Customer Bridge | 고객 소통 | Input, Decision, Evidence, Action |
| Report Preview | 출력 전 | Output Gate, Evidence Summary |

---

## 8. 레이아웃 규칙

### Desktop

```text
좌측: 필터/모듈
중앙: 작업 영역
우측: 근거/상태/조치
```

### Tablet

```text
상단: 검색/컨텍스트
중앙: 작업 영역
하단 또는 접힘: 근거/조치
```

### Mobile

```text
Trust
Context
Input/Search
Main
Evidence
Action
Gate
```

---

## 9. 콘텐츠 규칙

### 버튼

동사형 우선:
- 표준 찾기
- 근거 보기
- 판정표 만들기
- 교육자료 만들기
- 고객 답변 생성
- 출력 검사

### 상태 문구

- PASS: 기준 충족
- HOLD: 확인 필요
- FAIL: 기준 미달 또는 차단
- INFO: 참고 정보
- DRAFT: 초안
- READY: 사용 가능

### 금지 문구

- 완벽히 검증됨
- 무조건 맞음
- 전문가 수준 보장
- 근거 생략 가능
- 그냥 사용 가능

---

## 10. 리뷰 체크리스트

| 항목 | 질문 | 판정 |
|---|---|---|
| 쉬움 | 첫 행동이 10초 안에 보이는가? | PASS/HOLD/FAIL |
| 표준성 | 표준번호·개정·상태가 있는가? | PASS/HOLD/FAIL |
| 신뢰성 | 근거가 판정과 연결되는가? | PASS/HOLD/FAIL |
| 전문성 | 표준기관형 인상이 있는가? | PASS/HOLD/FAIL |
| 간결성 | 불필요한 장식이 제거되었는가? | PASS/HOLD/FAIL |
| 여백 | 정보가 숨 쉴 공간이 있는가? | PASS/HOLD/FAIL |
| 접근성 | 대비·라벨·포커스가 충족되는가? | PASS/HOLD/FAIL |
| 모듈성 | 기존 토큰/컴포넌트를 재사용했는가? | PASS/HOLD/FAIL |

---

## 11. 운영 규칙

- Guidebook 변경 시 Memory에 변경 요약 기록
- 공통 컴포넌트 변경 시 영향 모듈 점검
- 상태 체계 변경은 MAJOR 변경으로 간주
- 토큰 추가는 최소화
- 예외 허용은 문서화 없이 금지
- 공식 제품명 변경 금지

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

---

## 12. 변경 이력

### v0.1.0 — 2026-06-29

- 정식 Guidebook 생성
- 공식 제품명 고정
- Constitution / Guardrails 요약 통합
- 토큰, 컴포넌트, 페이지 패턴, 리뷰 체크리스트 정의

---
name: quali-standard-ui
description: Use this skill when creating or modifying Quali UI, 퀄리창고, 퀄리도서관, 브릿지, 스킬업, standard-based dashboards, evidence-driven HTML, PASS/HOLD/FAIL panels, standard search UI, education screens, customer bridge UI, Korean typography, line-break polishing, or design-token-based components.
---

# Quali Standard UI Skill

> 파일 위치: `.agents/skills/quali-standard-ui/SKILL.md`  
> 버전: v0.2.0  
> 상태: Active  
> 기준일: 2026-06-29

---

## Purpose

Create and modify Quali Series UI using the approved Constitution, Guardrails, design tokens, components, and workflow.

The skill must produce UI that is easy for beginners and credible for experts.

Core flow:

```text
Input → Standard → Decision → Evidence → Action
```

Korean working expression:

```text
입력 → 기준 → 판정 → 근거 → 조치
```

---

## Official Product Names

Use these names exactly.

- 퀄리창고
- 퀄리도서관
- 브릿지
- 스킬업

Do not rename these products.  
English helper names may appear only in code, IDs, API, or international-facing documentation.

---

## When to Use

Use this skill for:

- Quali UI system work
- 퀄리창고 UI
- 퀄리도서관 UI
- 브릿지 UI
- 스킬업 UI
- IPC-A-610 UI
- J-STD-001 UI
- IPC/WHMA-A-620 UI
- ECSS UI
- NASA standards UI
- Evidence-based report UI
- Standard search UI
- PASS/HOLD/FAIL decision UI
- HTML educational guide UI
- Design-token-based refactoring

Do not use this skill for unrelated creative landing pages, entertainment pages, or decorative-only design tasks.

---

## Required Reading

Before starting, read:

1. `00_QUALI_UI_CONSTITUTION.md`
2. `00_QUALI_UI_GUARDRAILS.md`
3. `AGENTS.md`
4. `PROJECT_DEVELOPMENT_GUIDEBOOK.md`
5. `PROJECT_DEVELOPMENT_MEMORY.md`
6. `COMMON_DEVELOPMENT_WORKFLOW.md`
7. `QUALI_UI_GOVERNANCE_ADVANCEMENT_STRATEGY.md` when editing the governance pack

For coding or file edits, also read:

8. `COMMON_DEVELOPMENT_WORKFLOW_CODEX_FINAL.md`

---

## Core Design Rules

1. Show the first action within 10 seconds.
2. Show the standard/rule context.
3. Show the decision status.
4. Show the evidence.
5. Show the next action.
6. Use approved tokens only.
7. Reuse existing components first.
8. Use text + color for status.
9. Separate beginner explanation and expert evidence.
10. Prefer tables for standards, clauses, versions, decisions, and traceability.
11. Do not leave Korean H1/H2 line breaks to browser defaults.
12. Use semantic line spans for major Korean titles.
13. Keep protected Korean terms and standard names together.
14. Run Line Break Gate before final output.
15. Keep status, token, component, and Evidence Schema definitions synchronized across governance documents.
16. Update and verify `SHA256SUMS.txt` after governance-pack changes.

---

## Approved Tokens

Use these tokens only.

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
  --q-font-main: "Noto Sans KR", system-ui, sans-serif;
  --q-font-code: ui-monospace, Menlo, Consolas, monospace;
}
```

---


## Korean Typography & Line Break Rules

Korean UI quality is part of Quali UI trust.  
A page that breaks Korean titles in the middle of particles, suffixes, or key terms must be returned as HOLD or FAIL.

### Required Rules

1. Do not leave Korean display titles to automatic browser wrapping.
2. Use `span.line` for H1/H2 when the title is large or brand-critical.
3. Do not allow Korean particles, suffixes, or key nouns to appear alone on a line.
4. Keep official product names and protected terms together.
5. Apply `word-break: keep-all` to Korean prose.
6. Allow `overflow-wrap:anywhere` only for code, paths, standard numbers, and long IDs.
7. H1 should normally use 2–3 semantic lines.
8. H2 should normally use 1–2 semantic lines.
9. Run Line Break Gate before final output.

### Protected Terms

```text
퀄리창고
퀄리도서관
브릿지
스킬업
우주 임무보증연구회
임무보증
뉴 스페이스
리스크 구조
핵심 원자료
시각 요약
K-MAR
COTS-EEE
NASA·ESA·JAXA
IPC-A-610
J-STD-001
IPC/WHMA-A-620
```

### Required CSS Pattern

```css
html,
body {
  word-break: keep-all;
  overflow-wrap: break-word;
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

### Required Markup Pattern

```html
<h1 class="display-title">
  <span class="line">우주 임무의 신뢰를</span>
  <span class="line">설계하고 검증하는</span>
  <span class="line">협의체</span>
</h1>
```

### Line Break Gate

Before final output, check:

- H1 semantic line breaks
- H2 semantic line breaks
- no isolated Korean particles
- no broken protected terms
- acceptable mobile title wrapping
- no “unfinished webpage” impression

---

## Approved Components

Use these first:

- Trust Banner
- Standard Context Bar
- Standard Code Badge
- Workbench Search
- Module Card
- Status Badge
- Evidence Box
- Trace Table
- Beginner Box
- Expert Box
- Action Checklist
- Output Gate
- Line Break Gate

Create a new component only when the existing components cannot solve the task and the new pattern is reusable across at least two modules.

---

## Approved Status Values

Use these status values consistently:

| Status | Meaning |
|---|---|
| DRAFT | Draft, not reviewed |
| INFO | Informational, not a decision |
| HOLD | Needs confirmation |
| FAIL | Blocked or below standard |
| REVIEWED | Reviewed, but not READY or PASS |
| READY | Ready for submission or use |
| PASS | Meets the required basis |
| ARCHIVED | Stored and inactive |

Do not use `REVIEWED` as a substitute for `READY` or `PASS`.

---

## Evidence Schema

Evidence-based UI must include these fields when a judgment, answer, report, or training feedback depends on evidence:

```json
{
  "standard_id": "IPC-A-610",
  "revision": "H",
  "clause": "TBD",
  "source_file": "TBD",
  "page": "TBD",
  "evidence_level": "direct | summarized | inferred | pending",
  "verification_status": "PASS | HOLD | FAIL",
  "limitation": "TBD",
  "next_action": "TBD"
}
```

Use `verification_status: "HOLD"` when evidence is incomplete.

---

## Output Patterns

### Standard Detail

```text
Trust Banner
Standard Context Bar
Beginner Box
Expert Box
Evidence Box
Trace Table
Action Checklist
```

### Customer Bridge

```text
Trust Banner
Question Input
Standard Selector
Decision Result
Evidence Summary
Action Checklist
Output Gate
```

### Training Page

```text
Learning Goal
Beginner Explanation
Expert Reference
Practice Task
Quiz
Feedback
Progress Status
```

### Warehouse Page

```text
Trust Banner
Search / Filter
Source Table
Metadata Panel
Version History
Action Checklist
```

---

## Preflight Checklist

Before implementation, answer:

```text
Target product:
Primary user:
First user action:
Input:
Standard/rule:
Decision status:
Evidence source:
Next action:
Existing components:
New component needed:
Verification:
Evidence Schema:
Document sync:
Hash update:
Korean title plan:
Protected terms:
Line Break Gate:
HOLD:
```

---

## Verification Checklist

Before final output, check:

- [ ] Beginner can identify the first action.
- [ ] Standard/rule context is visible.
- [ ] Decision status is visible.
- [ ] Evidence is visible.
- [ ] Next action is visible.
- [ ] Approved tokens are used.
- [ ] Existing components are reused.
- [ ] Status uses text + color.
- [ ] `REVIEWED` is not used as `READY` or `PASS`.
- [ ] Evidence Schema required fields are present when evidence is reused.
- [ ] Focus and labels are present.
- [ ] Mobile order remains logical.
- [ ] Korean H1/H2 line breaks are semantic.
- [ ] No particle, suffix, or protected term is broken across lines.
- [ ] Line Break Gate is completed.
- [ ] Reusable decisions are recorded in Memory.
- [ ] Document sync checks pass or are reported as HOLD.
- [ ] `SHA256SUMS.txt` is updated after governance-pack changes.

---

## Verification Commands

When the governance pack contains these scripts, run:

```text
npm run check:doc-sync
npm run check:tokens
npm run check:contrast
npm run check:evidence-schema
npm run check:korean-linebreak
npm run check:hash
```

If a command cannot run, report the reason as HOLD.

---

## Final Response Format

```text
Summary:
Changed files:
Reused components:
New components:
Verification:
Accessibility:
Memory update:
Evidence Schema:
Document sync:
SHA256SUMS:
Korean title plan:
Protected terms:
Line Break Gate:
HOLD:
Next action:
```

---

## Failure Conditions

Return HOLD or FAIL if:

- Evidence is missing.
- Status is color-only.
- Approved tokens are bypassed.
- Official product names are changed.
- User cannot identify the next action.
- Accessibility is not checked.
- Evidence Schema required fields are missing.
- Document sync fails.
- `SHA256SUMS.txt` is stale after governance-pack changes.
- Korean line breaks are not checked.
- H1/H2 titles break particles, suffixes, or protected terms.
- New reusable decision is not recorded.

---

## Notes

This skill is instruction-first, but the governance pack may also include schemas, scripts, templates, fixtures, and examples. When they exist, use them before hand-checking the same rule.

## Change History

### v0.2.0 — 2026-06-29

- Added governance advancement strategy to required reading for governance edits.
- Added approved status values, including `REVIEWED`.
- Added Evidence Schema requirements.
- Added verification commands and SHA256SUMS update rule.
- Added Line Break Gate to approved components.
- Updated Korean heading CSS to `letter-spacing: 0`.

### v0.1.1 — 2026-06-29

- Added Korean Typography & Line Break Rules.
- Added protected terms.
- Added Line Break Gate.
- Added required CSS and semantic title markup pattern.
- Updated verification and failure conditions.

### v0.1.0 — 2026-06-29

- Initial Quali Standard UI Skill created.

# Quali 헌법 Trace 템플릿 (PR/커밋 공통)

## 1) PR 본문에 붙이는 1줄(필수)
**Constitution-Trace:** 전자제조산업 표준AI 개발_헌법 <조항ID/제목>  <이번 변경이 헌법 완성에 기여하는 한 문장>

예시:
- Constitution-Trace: 전자제조산업 표준AI 개발_헌법 2.1(SSOT)  데이터/설정의 단일 출처를 강제해 운영 오류를 0으로 만든다.

## 2) 커밋 메시지 Footer(트레일러)로 붙이는 2줄(권장)
Constitution-Doc: 전자제조산업 표준AI 개발_헌법
Constitution-Trace: <조항ID/제목>  <기여 문장>

예시:
Constitution-Doc: 전자제조산업 표준AI 개발_헌법
Constitution-Trace: 2.1(SSOT)  관리자 배포 플로우를 SSOT로 고정한다.

## 3) 커밋 헤더(Conventional Commits 스타일 권장)
<type>(<scope>): <subject>

예시:
feat(admin): add constitution trace footer to PR template
fix(report): ensure proofpack index line includes sha256

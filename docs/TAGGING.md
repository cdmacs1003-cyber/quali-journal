# TAGGING.md

파일 머리말(주석)에 아래 3개 키를 권장:

```
component: admin|orchestrator|ci|docs|tools|data|other
feature: 짧은 설명(예: qa-auto, export, status-api)
anchors: 콤마로 구분된 앵커 문자열들(clearAdminTokenJJ, runQa, btn-qa)
```

- **anchors**는 나중에 앵커 기반 패치를 자동 검증하는 기준입니다(`tools/find_anchor.py`).
- 인벤토리(`/_inventory/inventory.json`)엔 파일 경로·해시·크기·컴포넌트·앵커가 기록됩니다.
- 매니페스트(`MANIFEST-SSOT.md`)는 사람용 요약 표입니다.


# QualiJournal 초간단 Health‑Check Kit (초딩 버전)

이 키트는 **로컬 저장소 경로**만 넘기면 9개 항목을 PASS/WARN/FAIL로 빠르게 점검합니다.

## 1) 실행 방법
### Windows (PowerShell)
```powershell
cd "/mnt/data/quali_healthcheck"
python .\run_all.py --repo "C:\바탕 화면\퀄리저널" --net
```
> 인터넷이 막힌 환경이면 `--net` 을 빼세요.

### macOS/Linux (bash)
```bash
cd "/mnt/data/quali_healthcheck"
python3 run_all.py --repo "/Users/you/Desktop/퀄리저널" --net
```

## 2) 결과물
- 터미널에 요약 출력
- `health_report.md` : 상세 보고서 (각 항목 근거/해결 팁 포함)

## 3) 빠른 원라이너(초간단 수동 점검)
아래는 저장소 루트에서 바로 확인 가능한 1줄 점검 예시입니다.

**1. config 스키마 일치**  
- (bash) `jq '.gate_required, .features.require_editor_approval' config.json`
- (PS)   `Get-Content config.json | jq '.gate_required, .features.require_editor_approval'`
- (키 존재 여부) `Select-String -Path config_schema_definition.json -Pattern 'gate_required'`

**2. 소스 설정(갯수/임계값/죽은 링크)**  
- `jq 'length' feeds/official_sources.json`  
- `jq '..|.score_threshold?|numbers' feeds/community_sources.json`  
- (링크 헤드) `curl -Is $(jq -r '.[]|.url? // .rss? // empty' feeds/official_sources.json | head -n 3) | head -n1`

**3. 오케스트레이션/인코딩/PS 의존성**  
- `Select-String admin/server_quali.py -Pattern 'encoding="utf-8"'`  
- `Test-Path run_quali_today.py` / `Test-Path run_quali_today.ps1`

**4. tools 중복**  
- `ls tools | findstr /i "repair_selection sync_selected force_approve augment"`

**5. admin ↔ server 연계(폴백/인코딩)**  
- `Select-String admin/index.html -Pattern 'meta charset="UTF-8"'`  
- `Select-String admin/index.html -Pattern '/api/tasks/flow'`

**6. selected_* 무결성(구조/중복/승인)**  
- `jq '.articles|length' selected_articles.json`  
- `jq '..|.approved?|booleans' selected_keyword_articles.json | sort | uniq -c`

**7. archive 파일명 충돌 예방**  
- `ls archive | sort` (중복 베이스 이름 존부 확인)

**8. requirements 누락**  
- `Select-String requirements.txt -Pattern 'fastapi|uvicorn|pydantic|praw|feedparser|beautifulsoup4|lxml|requests'`

**9. 경로/모듈 호출 일관성**  
- `Select-String -Path orchestrator.py,admin/server_quali.py -Pattern 'Path\(__file__\)|PROJECT_ROOT'`

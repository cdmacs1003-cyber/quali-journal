
# 퀄리저널 **초딩 운영자 매뉴얼** (현장용 · 1분 점검)

> 버전: 2025-11-09 · 대상: 첫 운영자(비개발) · 환경: Windows PowerShell + 로컬 Admin 서버(127.0.0.1:8010)

---

## 0. 준비 (한 번만)
1) **토큰** 넣기  
   - PowerShell 열기 → 아래 2줄을 복사/실행
   ```powershell
   cd "C:\Users\user\Desktop\퀄리저널\admin"
   $env:ADMIN_TOKEN="여기에_관리자토큰"
   ```

2) **서버 켜기**(안 켜져 있으면)  
   ```powershell
   uvicorn server_quali:app --host 0.0.0.0 --port 8010 --reload
   ```

3) **콘솔 UTF-8 고정**(한글 안 깨지게)  
   ```powershell
   chcp 65001 > $null
   [Console]::InputEncoding  = New-Object System.Text.UTF8Encoding
   [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
   ```

---

## 1. 원클릭 점검(매일) — `qa.ps1`
> 위치: `C:\Users\user\Desktop\퀄리저널\admin\qa.ps1`  
> 실행: PowerShell에서 `.\qa.ps1`

### 점검 항목(자동 출력)
1) **Reason Tags** — 기사 스티커(왜 채택/보류/불량인지)
2) **Ready KPI** — 오늘 준비 현황판
3) **Snapshot (monthly)** — 이달호 완본 저장(POST)
4) **Weekly Diff** — 이번 주 변경만 요약(POST)
5) **Verify files (hash)** — 실제 파일/해시 검증
6) **ETag 200→304** — 같은 내용은 재다운로드 않음

### 정상 출력 예시
- Snapshot: `ok: True`, `path: archive/snapshots/monthly/2025-11.json`
- Diff: `summary: {{added=…; removed=…; modified=…}}`
- Verify: `Snapshot/Diff 파일명 + 해시값`
- ETag: `HTTP 304` (혹은 200 — 데이터가 바뀐 경우)

---

## 2. 관리자 화면 버튼(오른쪽 Admin Ops)
- **월간 스냅샷**: `archive/snapshots/monthly/연-월.json` 생성
- **주간 Diff**: `archive/diffs/weekly/연도-주차.json` 생성
- **ETag 테스트**: 결과 `status: 304`면 정상

> 버튼은 항상 **POST**로 실행되므로, 파일이 생성되지 않는 “GET 404” 문제를 막아줍니다.

---

## 3. 자주 막히는 3가지와 해결
- **401** (인증 오류): 토큰 미설정 → `$env:ADMIN_TOKEN="..."` 다시 넣고 실행
- **한글 깨짐**: UTF-8 3줄(위 0-3단계) 다시 실행
- **스냅샷 404**: GET으로 부름 → **반드시 POST** (qa.ps1/버튼은 자동 POST)

---

## 4. 주간/월간 운영 루틴(샘플)
- **매일 아침**: `.\qa.ps1` → Ready KPI 확인 → 승인/발행 진행
- **주간 회의 전**: **Weekly Diff** 결과 캡처 → “이번 주 변화” 공유
- **월말 결산**: **Snapshot** 생성 + 해시값 엑셀 기록(감사/보고 대비)

---

## 5. 복구/백업 팁
- 서버 파일 교체 전엔 항상 `server_quali.py.bak` 백업 유지
- 스냅샷/디프 폴더(zip) 주기적 보관:  
  - `archive/snapshots/monthly/`  
  - `archive/diffs/weekly/`

---

## 6. 빠른 참조 (명령어 모음)

```powershell
# 토큰 설정
$env:ADMIN_TOKEN="..."

# 서버 실행
uvicorn server_quali:app --host 0.0.0.0 --port 8010 --reload

# 스냅샷(POST)
Invoke-RestMethod http://127.0.0.1:8010/api/archive/snapshot-monthly -Method POST -Headers @{{Authorization="Bearer $env:ADMIN_TOKEN"}}

# 주간 Diff(POST)
Invoke-RestMethod http://127.0.0.1:8010/api/archive/diff-weekly -Method POST -Headers @{{Authorization="Bearer $env:ADMIN_TOKEN"}}

# 파일/해시 검증
Get-ChildItem .\archive\snapshots\monthly\*.json | Sort-Object LastWriteTime -Desc | Select -First 1
Get-ChildItem .\archive\diffs\weekly\*.json     | Sort-Object LastWriteTime -Desc | Select -First 1
```

---

### 문의/점검 노트
- 버튼이 보이지 않으면 **브라우저 Ctrl+Shift+R**
- 관리자 토큰은 우측 상단 **토큰 바**에 입력/확인
- 에러 화면은 캡처하여 슬랙/이메일로 공유

> 쉬운 것이 답이다. 오늘도 원클릭으로 끝! 😊

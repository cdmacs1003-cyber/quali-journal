[초딩버전] Cloud Run 오류 해결 가이드

✅ 지금 만든 파일 (다운로드해서 프로젝트에 넣으세요)
1) requirements.fixed.txt  -> 기존 requirements.txt와 교체
2) server_quali.fixed.py   -> admin/server_quali.py(또는 루트/server_quali.py)와 교체
3) Dockerfile               -> 루트에 저장
4) start.sh                 -> 루트에 저장 (실행 권한 필요)

───────────────────────────────────────────────────────────
1단계) 파일 교체
- requirements.txt를 requirements.fixed.txt 내용으로 덮어쓰기
- server_quali.py를 server_quali.fixed.py 내용으로 덮어쓰기
  (위치가 admin/server_quali.py이면 그대로 덮어쓰고, 루트에 있으면 루트에 덮어쓰기)

2단계) 로컬 테스트
- pip install -r requirements.txt
- ./start.sh
- 브라우저에서 http://127.0.0.1:8080 열어보기

3단계) Cloud Run 배포 (도커 사용)
------------------------------------------------
PROJECT_ID=your-gcp-project
REGION=asia-northeast3
IMAGE=gcr.io/$PROJECT_ID/quali-journal

gcloud builds submit --tag $IMAGE
gcloud run deploy quali-journal \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated
------------------------------------------------

4단계) (급할 때) 콘솔에서 컨테이너 명령만 바꾸는 초간단 방법
- Cloud Run > 서비스 편집 > "컨테이너, 변수 & 비밀" > "컨테이너 명령어"
  Command: uvicorn
  Args:    admin.server_quali:app --host 0.0.0.0 --port 8080
  (server_quali.py가 루트면 Args를: server_quali:app --host 0.0.0.0 --port 8080)

왜 이걸로 끝나나요?
- 모듈 경로를 명확히 지정: admin/server_quali.py -> admin.server_quali:app
- 포트/호스트를 Cloud Run 규칙에 맞춤: 0.0.0.0:8080
- python-dotenv가 없어도 앱이 죽지 않게 보호 (try/except) + requirements에 추가

체크리스트
- [ ] admin/server_quali.py 맞으면: uvicorn admin.server_quali:app ...
- [ ] 외부접속: --host 0.0.0.0
- [ ] 포트: --port 8080
- [ ] .env 사용: requirements에 python-dotenv 포함
- [ ] 배포 후 로그에: "Uvicorn running on 0.0.0.0:8080"

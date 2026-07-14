# qlib-skillup-runtime R470 deployment runbook

Status: `READY_NOT_EXECUTED` only after R469A technical validation passes.

This runbook is for `qlib-skillup-runtime` in `asia-northeast1` only. It must never target or mutate `quali-admin-domap`. It does not grant deployment authorization. Every command below is deferred to a separately authorized R470 packet.

## Fixed boundary

- Mode: authenticated limited field beta.
- Anonymous/public access: forbidden.
- Custom domain: deferred and not authorized.
- Production DB, Production Library, Cloud SQL, migrations, queues and schedulers: no dependency and no mutation.
- Raw user text retention, raw standard export, unknown-rights use and external-provider writes: forbidden.
- Resources: min 0, max 2, CPU 1, memory 512Mi, concurrency 80, timeout 300 seconds.
- Initial stable rollback revision: `qlib-skillup-runtime-00002-d9g`.

## Required R470 variables

R470 must resolve these without printing credentials or secret values:

```powershell
$PROJECT_ID = '<separately-authorized-project-id>'
$REGION = 'asia-northeast1'
$SERVICE = 'qlib-skillup-runtime'
$STABLE_REVISION = 'qlib-skillup-runtime-00002-d9g'
$SOURCE_COMMIT = git rev-parse HEAD
$SOURCE_DATE_EPOCH = git show -s --format=%ct $SOURCE_COMMIT
$OCI_ARTIFACT = '<task-owned-new-path>/qlib-skillup-runtime.oci.tar'
$LOCAL_IMAGE = "qlib-skillup-runtime:r469a-$($SOURCE_COMMIT.Substring(0,12))"
$IMMUTABLE_IMAGE = '<registry>/qlib-skillup-runtime@sha256:<resolved-digest>'
```

Abort unless `$SOURCE_COMMIT` is the final validated R469A commit and `git status --short` contains only the 17 preserved inherited untracked candidates.

## 1. Pre-deploy evidence and service capture

1. Verify R463-R469A ProofPack manifests and SHA256 registers with zero missing, unregistered or mismatched files.
2. Verify the image labels `source_repository`, `source_branch`, `source_commit`, `task_id`, `target_service`, and `target_mode`.
3. Export current configuration before any mutation:

```powershell
gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID --format=json
gcloud run services get-iam-policy $SERVICE --region $REGION --project $PROJECT_ID --format=json
gcloud run revisions list --service $SERVICE --region $REGION --project $PROJECT_ID --format=json
```

Store only redacted/masked metadata in the R470 ProofPack. Capture the current revision and traffic, confirm `$STABLE_REVISION` still exists, and confirm no `allUsers` or `allAuthenticatedUsers` invoker binding.

## 2. Reproduce and publish the immutable artifact

The R469A local image hash must match the validated artifact. Registry write is allowed only by the future R470 authorization.

```powershell
powershell -File tools/build_qlib_runtime_artifact.ps1 -OutputPath $OCI_ARTIFACT -SourceCommit $SOURCE_COMMIT
docker load --input $OCI_ARTIFACT
# Publish only through the separately authorized registry workflow.
docker tag $LOCAL_IMAGE '<registry>/qlib-skillup-runtime:r469a-<12-char-source-commit>'
docker push '<registry>/qlib-skillup-runtime:r469a-<12-char-source-commit>'
docker buildx imagetools inspect '<registry>/qlib-skillup-runtime:r469a-<12-char-source-commit>'
```

Resolve `$IMMUTABLE_IMAGE` to the pushed `@sha256:` reference. Stop if its labels or digest differ from the R469A record.

## 3. Create a no-traffic private revision

```powershell
gcloud run deploy $SERVICE --project $PROJECT_ID --region $REGION --image $IMMUTABLE_IMAGE --no-traffic --no-allow-unauthenticated --port 8080 --min 0 --max 2 --cpu 1 --memory 512Mi --concurrency 80 --timeout 300 --tag r470-candidate
```

Capture the candidate revision name. Confirm no Cloud SQL attachment, volume, migration, queue, scheduler, secret binding, or Production DB/Library setting was added. Required application secret binding count is zero.

## 4. Revision-specific authenticated smoke at 0% service traffic

Using an authorized identity token without printing it, call only the candidate tag/revision URL:

- `GET /health`: HTTP 200 and `status=ok`.
- `GET /`: beginner screen loads and static asset requests succeed.
- `POST /api/f13/bridge/skillup/bridge-answer`: one approved packaged safe-summary question returns an answer with Evidence/Trace; one unsupported natural question returns additional review/HOLD.
- An unauthenticated request is rejected at the Cloud Run IAM boundary.
- No raw answer/query retention, internal path, secret, standard raw text, Production write, or cross-state contamination is observed.

Remain at 0% until every check passes. Any failure triggers the stop procedure; no service traffic is assigned.

## 5. Staged traffic and observations

At each stage, preserve the exact candidate/stable split, execute the health/auth/Evidence/Trace checks, and record error, latency, instance, CPU, memory and cost signals.

```powershell
gcloud run services update-traffic $SERVICE --project $PROJECT_ID --region $REGION --to-revisions "$CANDIDATE_REVISION=5,$STABLE_REVISION=95"
# Observe 10 minutes.
gcloud run services update-traffic $SERVICE --project $PROJECT_ID --region $REGION --to-revisions "$CANDIDATE_REVISION=20,$STABLE_REVISION=80"
# Observe 10 minutes.
gcloud run services update-traffic $SERVICE --project $PROJECT_ID --region $REGION --to-revisions "$CANDIDATE_REVISION=50,$STABLE_REVISION=50"
# Observe 15 minutes.
gcloud run services update-traffic $SERVICE --project $PROJECT_ID --region $REGION --to-revisions "$CANDIDATE_REVISION=100"
# Final observation: 15 minutes.
```

## 6. Mandatory stop and rollback

Stop on health failure, authentication-boundary regression, Evidence/Trace failure, raw/internal-path/secret exposure, unexpected Production write, error-rate increase above the baseline threshold captured immediately before deployment, cost/capacity anomaly, rollback-target loss, or owner stop instruction.

```powershell
gcloud run services update-traffic $SERVICE --project $PROJECT_ID --region $REGION --to-revisions "$STABLE_REVISION=100"
```

Verify the candidate receives 0% service traffic. Do not delete either revision during emergency rollback. Repeat authenticated health and flow checks against the restored stable service and record post-rollback evidence.

## 7. Completion evidence and cleanup

The R470 ProofPack must contain redacted pre/post configuration, stable and candidate revisions, immutable image digest and labels, traffic commands/results, stage observation timestamps, health/auth/Evidence/Trace results, leak/write zero counts, cost/capacity observations, rollback evidence if used, final repository state, manifest, and SHA256 register. Remove only task-owned local containers, networks and temporary artifacts. Create the single R470 Completion Report. Deployment remains incomplete until the owner provides explicit R470 execution authorization and the post-deploy evidence passes.

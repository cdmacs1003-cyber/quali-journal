# qlib-skillup-runtime limited field beta deployment runbook

Status: `R479_DEPLOYMENT_PRECONDITION_CONTRACT_DEFINED`; R480 deployment remains `NOT_GRANTED` without a separate exact owner decision.

This runbook is for `qlib-skillup-runtime` in `asia-northeast1` only. It must never target or mutate `quali-admin-domap`. It does not grant deployment authorization. Every command below is deferred to a separately authorized R476 packet.

The R476 and R478 execution authorizations are spent. This runbook defines the corrected rollback and decision contract but grants no candidate, traffic, IAM, registry or deployment mutation. Any retry requires a separate exact R480 owner decision.

## Fixed boundary

- Mode: authenticated limited field beta.
- Anonymous/public access: forbidden.
- Custom domain: deferred and not authorized.
- Production DB, Production Library, Cloud SQL, migrations, queues and schedulers: no dependency and no mutation.
- Raw user text retention, raw standard export, unknown-rights use and external-provider writes: forbidden.
- New-candidate resources: min 0, max 2, CPU 1, memory 512Mi, concurrency 80, timeout 300 seconds.
- The service template contract is min 0 / max 2. A fresh candidate independently requires effective min 0 / max 2 and immutable revision annotation `autoscaling.knative.dev/maxScale=2`.
- Service-template max 2 never substitutes for the fresh candidate immutable maxScale 2 gate.
- Historical rollback revision: `qlib-skillup-runtime-00002-d9g`.
- The historical rollback revision's immutable maxScale is 20. This is a pre-existing accepted rollback boundary, not a new-candidate capacity contract. In-place repair is forbidden.
- Rollback restores traffic to the existing Ready historical revision; it does not create a new maxScale-20 revision.
- Rejected revision `qlib-skillup-runtime-00003-som` must remain at 0% percentage traffic and must not be reused.
- Withdrawn R476 candidate `qlib-skillup-runtime-00010-zuj` must remain at 0% traffic and must not be reused.

## Required R474 variables

R474 must resolve these without printing credentials or secret values:

```powershell
$PROJECT_ID = '<runtime-resolved-active-project-id>'
$REGION = 'asia-northeast1'
$SERVICE = 'qlib-skillup-runtime'
$STABLE_REVISION = 'qlib-skillup-runtime-00002-d9g'
$REJECTED_REVISION = 'qlib-skillup-runtime-00003-som'
$SOURCE_COMMIT = git rev-parse HEAD
$SOURCE_DATE_EPOCH = git show -s --format=%ct $SOURCE_COMMIT
$OCI_ARTIFACT = '<task-owned-new-path>/qlib-skillup-runtime.oci.tar'
$LOCAL_IMAGE = "qlib-skillup-runtime:r471-$($SOURCE_COMMIT.Substring(0,12))"
$REGISTRY = "asia-northeast1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy"
$IMMUTABLE_IMAGE = "$REGISTRY/qlib-skillup-runtime@sha256:<resolved-digest>"
```

Abort unless `$SOURCE_COMMIT` is the final validated R473 commit and `git status --short` contains only the 17 preserved inherited untracked candidates.

The runtime-resolved project value must match the active authenticated project and must not be written to public reports. Abort unless the existing `cloud-run-source-deploy` repository remains a private `DOCKER` `STANDARD_REPOSITORY` in `asia-northeast1`.

## 1. Pre-deploy evidence and service capture

1. Verify R463-R473 ProofPack manifests and SHA256 registers with zero missing, unregistered or mismatched files.
2. Verify the image labels `source_repository`, `source_branch`, `source_commit`, `task_id`, `target_service`, and `target_mode`.
3. Live-read the retained registry manifest and config without publishing or changing tags. A sanitized final artifact must prove the exact manifest digest, config digest, all layer digests, six required labels, exact source commit, runtime user `10001:10001`, private repository access, no `latest` use, and zero task-window push/republication/deletion audit entries. Intermediate output or a missing completion marker is `NOT_VERIFIED`.
4. Export current configuration before any mutation:

```powershell
gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID --format=json
gcloud run services get-iam-policy $SERVICE --region $REGION --project $PROJECT_ID --format=json
gcloud run revisions list --service $SERVICE --region $REGION --project $PROJECT_ID --format=json
```

Store only redacted/masked metadata in the R476 ProofPack. Capture the current revision and traffic, confirm `$STABLE_REVISION` still exists, and confirm no `allUsers` or `allAuthenticatedUsers` invoker binding. Read and normalize the IAM policy before deployment, record its SHA256 and public-member count, and stop without mutation if the public-member count is nonzero.

The private boundary is read-only in this workflow. Never pass `--allow-unauthenticated` or `--no-allow-unauthenticated`; both flags can invoke `SetIamPolicy`. Before any candidate traffic, read and normalize the IAM policy again, compare it byte-for-byte/hash-for-hash with the predeploy snapshot, require public-member count 0, require policy delta 0, and require `SetIamPolicy` audit count 0. Any mismatch forbids candidate traffic and triggers the stop procedure; do not repair IAM in-band.

## 2. Reproduce and publish the immutable artifact

The R473 local image hash must match the newly owner-approved artifact digest. Registry write is allowed only by the future R474 authorization.

```powershell
powershell -File tools/build_qlib_runtime_artifact.ps1 -OutputPath $OCI_ARTIFACT -SourceCommit $SOURCE_COMMIT
docker load --input $OCI_ARTIFACT
# Publish only through the separately authorized registry workflow.
docker tag $LOCAL_IMAGE "$REGISTRY/qlib-skillup-runtime:r474-<12-char-source-commit>"
docker push "$REGISTRY/qlib-skillup-runtime:r474-<12-char-source-commit>"
docker buildx imagetools inspect "$REGISTRY/qlib-skillup-runtime:r474-<12-char-source-commit>"
```

Resolve `$IMMUTABLE_IMAGE` to the pushed `@sha256:` reference. Stop if its labels or digest differ from the owner-approved R473 record.

## 3. Create a no-traffic private revision

```powershell
gcloud run deploy $SERVICE --project $PROJECT_ID --region $REGION --image $IMMUTABLE_IMAGE --no-traffic --port 8080 --min=0 --max=2 --min-instances=0 --max-instances=2 --cpu 1 --memory 512Mi --concurrency 80 --timeout 300 --tag r476-candidate
```

Capture a newly created candidate revision name and abort if it equals `$REJECTED_REVISION` or `qlib-skillup-runtime-00004-kos`. Confirm no Cloud SQL attachment, volume, migration, queue, scheduler, secret binding, or Production DB/Library setting was added. Required application secret binding count is zero.

Immediately after the read-only candidate configuration check, repeat the IAM policy read and normalized hash. Require the pre/post hashes to match exactly, public member count 0, no policy delta, and zero `SetIamPolicy` audit calls. If any check fails, keep stable at 100%, assign candidate 0%, and do not run smoke or traffic.

Before revision-specific smoke or any percentage traffic, read the immutable revision annotation and require the exact value `2`:

```powershell
$REVISION_MAX_SCALE = (gcloud run revisions describe $CANDIDATE_REVISION --project $PROJECT_ID --region $REGION --format="value(metadata.annotations.'autoscaling.knative.dev/maxScale')").Trim()
if ($REVISION_MAX_SCALE -cne '2') {
    throw 'Candidate immutable maxScale must be exactly 2; missing, 20, or any other value forbids smoke and traffic.'
}
```

The Service-wide `--max=2` result is a separate defense-in-depth limit and cannot satisfy this gate.

## 4. Revision-specific authenticated smoke at 0% service traffic

Using an authorized identity token without printing it, call only the candidate tag/revision URL:

- `GET /health`: HTTP 200 and a health value that becomes exactly `ok` after case-insensitive, surrounding-whitespace-trimmed normalization. `OK`, `ok`, and surrounding whitespace are equivalent; every other value fails.
- `GET /`: beginner screen loads and static asset requests succeed.
- `POST /api/f13/bridge/skillup/bridge-answer`: one approved packaged safe-summary question returns an answer with Evidence/Trace; one unsupported natural question returns additional review/HOLD.
- An unauthenticated request is rejected at the Cloud Run IAM boundary.
- No raw answer/query retention, internal path, secret, standard raw text, Production write, or cross-state contamination is observed.

Remain at 0% until the immutable maxScale gate and every smoke check pass. Any failure triggers the stop procedure; no service traffic is assigned.

## 5. Staged traffic and observations

At each stage, preserve the exact candidate/stable split, execute the health/auth/Evidence/Trace checks, and record error, latency, instance, CPU, memory and cost signals.

Every 10/15-minute window must use the canonical detached controller. The launcher must return promptly, and every later poll must run from a new shell using the same observation id. PowerShell background jobs and a foreground sleep/observe loop are forbidden because their lifetime is coupled to the launching shell. The task-owned production sampler is passed only through the process environment, obtains identity material in its own memory, and emits sanitized JSON; its argv, identity material, raw URL, request text and raw output must never be persisted by the controller.

```powershell
$OBSERVER_ROOT = '<task-owned-proofpack-root>/observer'
$OBSERVATION_ID = 'stage-005-<approved-run-id>'
$env:QLIB_OBSERVER_PRODUCTION_SAMPLER = '["powershell","-NoProfile","-File","<approved-task-owned-sanitized-sampler>"]'
python -B tools/qlib_traffic_observer.py start --artifact-root $OBSERVER_ROOT --observation-id $OBSERVATION_ID --mode production --duration-seconds 600 --sample-interval-seconds 50 --max-gap-seconds 58 --stale-after-seconds 75
$env:QLIB_OBSERVER_PRODUCTION_SAMPLER = $null

# Run from a new shell; each invocation returns immediately.
python -B tools/qlib_traffic_observer.py poll --artifact-root $OBSERVER_ROOT --observation-id $OBSERVATION_ID
```

Use 600 seconds for the 5% and 20% stages and 900 seconds for the 50% and 100% stages. A stage is PASS only when `final.json` exists, `completion_marker=true`, `process_exit_code=0`, `monotonic_elapsed_seconds` is at least the requested duration, and the recorded maximum gap does not exceed the configured limit. Missing final markers, stale heartbeats, excess sample gaps, duplicate observation ids and child-process loss are `INCOMPLETE/NOT_VERIFIED` and trigger the stop procedure. Keep `start.json`, `events.ndjson`, `heartbeat.json`, `state.json`, and either `final.json` or `incomplete.json` in the ProofPack.

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

### Deterministic stage decision contract

The observer lifecycle gate and the operational metric gate are both required. A complete observer final artifact alone does not make unhealthy samples pass.

Functional/authentication contract:

- unauthenticated health status: 403;
- authenticated health status: 200;
- normalized health value: `ok`, using case-insensitive trimmed normalization;
- probe timeout: 10 seconds;
- allowed synthetic health failures, authentication failures, unexpected 5xx, Evidence/Trace omissions, raw/secret/internal-path exposures and Production writes: 0.

Latency contract:

```text
ABSOLUTE_P95_LATENCY_LIMIT_MS=3000
RELATIVE_P95_MULTIPLIER_FROM_STABLE_BASELINE=2.0
LATENCY_STOP_LIMIT_MS=max(3000, stable_baseline_p95_ms * 2.0)
```

A one-minute window qualifies only with at least 20 valid requests. Two consecutive qualifying windows above the limit are STOP. Any individual synthetic probe that exceeds the 10-second timeout is immediate STOP.

Error-rate contract:

```text
AGGREGATE_5XX_ABSOLUTE_LIMIT_PERCENT=1.0
AGGREGATE_5XX_BASELINE_DELTA_LIMIT_PERCENTAGE_POINT=0.5
ERROR_RATE_STOP_LIMIT_PERCENT=max(1.0, stable_baseline_5xx_rate_percent + 0.5)
```

Two consecutive qualifying one-minute windows above that limit are STOP. Synthetic unexpected 5xx allowance remains 0.

Low-volume contract: a window with fewer than 20 requests is `INSUFFICIENT_DATA`, never aggregate PASS, and requires fallback. `PASS_WITH_LOW_VOLUME_LIMITS` is allowed only when synthetic health/auth failures, unexpected synthetic 5xx, timeouts and Evidence/Trace omissions are all 0; synthetic p95 is at or below the latency stop limit; capacity is PASS; cost proxy is `PASS_WITH_LIMITS`; and the observer final artifact is PASS.

Fresh-candidate capacity contract:

```text
effective min/max=0/2
immutable maxScale=2
active instances<=2
failed startup=0
request drop or throttle=0
pending Not Ready<=120 seconds
concurrency=80
CPU=1
memory=512Mi
timeout=300 seconds
```

The historical rollback revision's maxScale 20 must not be used to judge fresh-candidate capacity.

Cost proxy contract:

```text
candidate min/max=0/2
candidate revision creation count<=1
total authorized staged observation seconds<=3000
unexpected billable resource delta=0
image push=0
Cloud SQL binding=0
additional service or scheduler=0
active candidate instances<=2
```

Every proxy condition is required for `COST_PROXY=PASS_WITH_LIMITS`; any excess is STOP. Real-time Cloud Billing amount remains `NOT_VERIFIED` unless authoritative billing data is actually available, and proxy compliance must never be reported as an amount PASS.

## 6. Mandatory stop and rollback

Stop on any functional/authentication failure, Evidence/Trace failure, raw/internal-path/secret exposure, Production write, synthetic timeout, unexpected synthetic 5xx, two consecutive qualifying latency or aggregate-5xx windows above the exact limits, a failed low-volume fallback, fresh-candidate capacity failure, cost-proxy excess, observer incomplete/final-marker failure, rollback-target loss, IAM/public-member/SetIamPolicy drift, traffic mismatch, or owner stop instruction.

```powershell
gcloud run services update-traffic $SERVICE --project $PROJECT_ID --region $REGION --to-revisions "$STABLE_REVISION=100"
```

Verify the candidate receives 0% service traffic. Do not delete either revision during emergency rollback. Repeat authenticated health and flow checks against the restored stable service and record post-rollback evidence.

## 7. Completion evidence and cleanup

The R474 ProofPack must contain redacted pre/post configuration, stable and candidate revisions, immutable image digest and labels, traffic commands/results, stage observation timestamps, health/auth/Evidence/Trace results, leak/write zero counts, cost/capacity observations, rollback evidence if used, final repository state, manifest, and SHA256 register. Remove only task-owned local containers, networks and temporary artifacts. Create the single R474 Completion Report. Deployment remains incomplete until the owner provides exact new-digest R474 execution authorization and the post-deploy evidence passes.

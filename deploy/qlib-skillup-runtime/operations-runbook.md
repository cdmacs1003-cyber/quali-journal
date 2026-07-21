# qlib-skillup-runtime limited field beta deployment runbook

Status: `R483_STAGE_OBSERVER_RELIABILITY_DEFINED`; R484 deployment remains `NOT_GRANTED` without a separate exact owner decision and a resolved split-traffic compatibility strategy.

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

## R482 incident closure and compatibility HOLD

R482 established `PRODUCT_SPLIT_TRAFFIC_COMPATIBILITY_DEFECT`. The no-traffic smoke used the candidate-tagged revision target and passed. The 5% stage used the service-wide target while its sampler also called candidate-only UI, ANSWERED, HOLD, Evidence and Trace surfaces. The historical stable revision does not provide those surfaces and returned HTTP 404 in the recorded stable-service reproduction. A service-wide split therefore cannot make those candidate-only functional calls revision-consistent. This is an actual staged-user compatibility risk, not merely an observer routing inconvenience.

The canonical observer must preserve that STOP. It may classify the failure as `SAMPLER_HTTP_404` with sanitized structured metadata, but it must never convert the product incompatibility to PASS. R484 deployment remains NOT_GRANTED until an owner selects and validates one strategy:

| Strategy | Safety effect | Required follow-up |
|---|---|---|
| Backward-compatible surface on every traffic-bearing revision | Service-wide user requests remain route-compatible | Separate runtime/product repair and immutable-image approval |
| Candidate-tagged or revision-specific limited-user route | Functional beta requests reach only the candidate | Separate owner decision; does not prove service-wide split compatibility |
| Blue/green single-revision promotion after no-traffic validation | Avoids mixed incompatible surfaces | Separate cutover and rollback authorization |

Session affinity is not a substitute for compatible traffic-bearing revisions unless separately proved for every request path. Until a strategy is approved, percentage traffic deployment is HOLD.

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

Cloud Run authentication and application health use two coexisting contracts. The
canonical identity-token audience is the exact, pathless HTTPS `run.app`
`status.url` returned by `gcloud run services describe` for the stable service.
The stable health request uses `<stable status.url>/health`. A private-tag health
request uses `<private tag URL>/health`, but its audience remains the stable
service `status.url`; a tag URL, revision URL, custom domain, or a value containing
`/health` is never an audience. Obtain the token only with
`gcloud auth print-identity-token --audiences=<memory-only stable service status.url> --quiet`,
compare its decoded `aud` claim exactly in memory before HTTP, and persist neither
the URL, token, decoded claim, nor response body. The Cloud Run boundary must
return 403 without authentication and 200 with authentication. The authenticated
JSON schema is exact: string field `status` equals `ok` and string field `service`
equals `qlib-skillup-runtime`.

This exact Cloud Run response-schema contract does not replace the legacy
observer and rollback normalized-health evidence contract below. They coexist;
the latter continues to use case and surrounding-whitespace normalization only
within its established legacy scope.

Using an authorized identity token without printing it, call only the candidate tag/revision URL:

- `GET /health`: HTTP 200 and a health value that becomes exactly `ok` after case-insensitive, surrounding-whitespace-trimmed normalization. `OK`, `ok`, and surrounding whitespace are equivalent; every other value fails.
- `GET /`: beginner screen loads and static asset requests succeed.
- `POST /api/f13/bridge/skillup/bridge-answer`: one approved packaged safe-summary question returns an answer with Evidence/Trace; one unsupported natural question returns additional review/HOLD.
- An unauthenticated request is rejected at the Cloud Run IAM boundary.
- No raw answer/query retention, internal path, secret, standard raw text, Production write, or cross-state contamination is observed.

Remain at 0% until the immutable maxScale gate and every smoke check pass. Any failure triggers the stop procedure; no service traffic is assigned.

## 5. Staged traffic and observations

At each stage, preserve the exact candidate/stable split, execute the health/auth/Evidence/Trace checks, and record error, latency, instance, CPU, memory and cost signals.

Every 10/15-minute window must use the canonical detached controller on an approved Linux host. On Windows, only the tracked PowerShell-to-existing-WSL2 entrypoint is supported; direct native-Windows invocation fails closed with `WINDOWS_NATIVE_OBSERVER_NOT_APPROVED` before creating an artifact or process. The launcher must return promptly, and every later poll must run from a new shell using the same observation id. PowerShell background jobs and a foreground sleep/observe loop are forbidden because their lifetime is coupled to the launching shell. The task-owned production sampler is passed only through the process environment, obtains identity material in its own memory, and emits sanitized JSON; its argv, identity material, raw URL, request text and raw output must never be persisted by the controller.

Before any percentage traffic mutation, run a short production-mode readiness observation at 0% candidate traffic through the same detached controller, Python/module path, working directory inheritance, sampler argv encoding, sanitized child environment and auth handoff used by the stage. It must complete and emit at least one valid health sample plus `import_status`, `dependency_status`, `auth_handoff_status`, `target_construction_status`, `health_sample_status` and `readiness_status` equal to `PASS`. Raw-response and identity-material persistence flags must be false. A missing field or mismatch is incomplete/STOP.

The sampler target contract is explicit:

- `HEALTH_ONLY_SERVICE`: service-wide health, latency and aggregate error only;
- `REVISION_FUNCTIONAL`: candidate revision/tag UI, ANSWERED, HOLD, Evidence and Trace only;
- `SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL`: both paths are independently constructed and reported; aggregate probes use the service target while functional probes use the candidate revision target.

Never issue candidate-only functional calls to the service-wide split target unless a separately validated compatibility matrix proves every traffic-bearing revision implements those surfaces. Production stages require `SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL`. Smoke requires `REVISION_FUNCTIONAL`.

The following legacy deployment-controller example is not a D42 operator
entrypoint. Its Python process must be the approved Linux runtime. A Windows
operator uses the tracked WSL2 fallback instead of entering these commands.

```powershell
$OBSERVER_ROOT = '<task-owned-proofpack-root>/observer'
$OBSERVATION_ID = 'stage-005-<approved-run-id>'
$env:QLIB_OBSERVER_PRODUCTION_SAMPLER = '["powershell","-NoProfile","-File","<approved-task-owned-sanitized-sampler>"]'
python -B tools/qlib_traffic_observer.py start --artifact-root $OBSERVER_ROOT --observation-id $OBSERVATION_ID --mode production --required-target-contract SPLIT_AGGREGATE_AND_REVISION_FUNCTIONAL --duration-seconds 600 --sample-interval-seconds 50 --max-gap-seconds 58 --stale-after-seconds 75
$env:QLIB_OBSERVER_PRODUCTION_SAMPLER = $null

# Run from a new shell; each invocation returns immediately.
python -B tools/qlib_traffic_observer.py poll --artifact-root $OBSERVER_ROOT --observation-id $OBSERVATION_ID
```

Use 600 seconds for the 5% and 20% stages and 900 seconds for the 50% and 100% stages. A stage is PASS only when `final.json` exists, `completion_marker=true`, `process_exit_code=0`, `monotonic_elapsed_seconds` is at least the requested duration, and the recorded maximum gap does not exceed the configured limit. Missing final markers, stale heartbeats, excess sample gaps, duplicate observation ids and child-process loss are `INCOMPLETE/NOT_VERIFIED` and trigger the stop procedure. Keep `start.json`, `events.ndjson`, `heartbeat.json`, `state.json`, and either `final.json` or `incomplete.json` in the ProofPack.

Sampler failures must not collapse to generic `SAMPLER_FAILURE`. The incomplete artifact records only the first failure phase, one of HTTP 403, HTTP 404, HTTP 5xx, timeout, JSON parse, auth, target routing, import/environment, argument/serialization, dependency/subprocess, functional HTTP, or verified external transient categories, the canonical source file/function/line, dependency class, exit category and retryable flag. Raw exception messages, stderr, URLs, response bodies and identity material are forbidden. Non-retryable 4xx, configuration, import, auth and dependency failures stop immediately. Only a sampler-declared `VERIFIED_EXTERNAL_TRANSIENT` may retry, with `MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRIES=1` and `MAX_VERIFIED_EXTERNAL_TRANSIENT_RETRY_SECONDS=10`.

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

### Pretraffic audit and internal-path evidence contract

SetIamPolicy evidence must use one immutable query contract. Resolve the active project in memory, target the exact Cloud Run service resource and `asia-northeast1`, use exact method equality for `google.cloud.run.v1.Services.SetIamPolicy`, and use the exact closed start/end window of the deployment task. Deduplicate by `insertId`, or by a SHA256 canonical event identifier only when `insertId` is absent. Project-wide events, registry/tag/service updates and regex method matches are excluded.

Run the identical closed-window query three times with at least 30 seconds between completed and next-start timestamps. Every result requires `completion_marker=true`, `partial_result=false`, one identical filter-contract hash and one identical event-id hash set. A partial result, missing marker, filter drift or a non-monotonic sequence such as 0/1/0 is `NOT_VERIFIED`; it can never be promoted to IAM PASS. Sanitized event evidence may retain only timestamp, exact method, resource and event-id hashes, principal category, policy-delta status and normalized IAM hashes. Raw principals and policies are forbidden.

Internal-path detection operates on parsed response values only. It must not scan serialized field names or count a slash merely because it appears in a route, JSON pointer or schema selector. True filesystem matches require a path segment after a Windows drive root, a UNC server/share, or one of `/workspace/`, `/root/`, `/home/`, `/tmp/` and `/var/`. Escaped drive-root schema literals without a path segment, `/health`, `/assets/...`, application routes, URL paths, JSON pointers, Evidence identifiers, field names and normalized health values are safe unless separate filesystem evidence exists.

Privacy-safe match artifacts contain only detector rule id, response-surface category, selector SHA256, matched-value SHA256, value length, match type, classification and `raw_fragment_persisted=false`. Identity material and raw response fragments must remain memory-only. A repaired detector must prove all required true fixtures are detected, all safe route/schema fixtures have zero true-path false positives, and a 0-percent candidate diagnostic has `true_internal_path_count=0` before any traffic decision. R480 candidate `qlib-skillup-runtime-00013-mad` is diagnostic-only and must not be reused for deployment. R482 deployment remains not granted without a separate exact owner decision.

## 6. Mandatory stop and rollback

Stop on any functional/authentication failure, Evidence/Trace failure, raw/internal-path/secret exposure, Production write, synthetic timeout, unexpected synthetic 5xx, two consecutive qualifying latency or aggregate-5xx windows above the exact limits, a failed low-volume fallback, fresh-candidate capacity failure, cost-proxy excess, observer incomplete/final-marker failure, rollback-target loss, IAM/public-member/SetIamPolicy drift, traffic mismatch, or owner stop instruction.

```powershell
$ROLLBACK_COMMAND = @('gcloud','run','services','update-traffic',$SERVICE,'--project',$PROJECT_ID,'--region',$REGION,'--to-revisions',"$STABLE_REVISION=100",'--quiet')
$ROLLBACK_EXECUTABLE = $ROLLBACK_COMMAND[0]
$ROLLBACK_ARGUMENTS = $ROLLBACK_COMMAND[1..($ROLLBACK_COMMAND.Count-1)]
& $ROLLBACK_EXECUTABLE @ROLLBACK_ARGUMENTS
if ($LASTEXITCODE -ne 0) { throw 'ROLLBACK_MUTATION_COMMAND_FAILED' }
```

The mutation command is executed and recorded separately from every read-only verification dependency. Its exit category must not be overwritten by a later describe, IAM, token, health or artifact-write failure. After mutation success, independently verify stable traffic 100%, every candidate 0%, no other positive traffic, stable Ready, unauthenticated/authenticated health 403/200 with normalized `ok`, IAM hash unchanged, public members 0 and authoritative SetIamPolicy count 0. A post-verification failure remains STOP but is reported as `ROLLBACK_POST_VERIFICATION_FAILED`, not as an ambiguous mutation failure. Do not delete either revision during emergency rollback.

The R482 helper coupled the mutation and all verification dependencies behind one generic exception and therefore returned exit 1 without proving which phase failed or whether traffic changed. The direct stable-100 command succeeded because it used the deterministic primary mutation path alone. Future rollback tooling must use the isolated command construction and post-verification contract above; a monolithic helper is not the primary path.

## 7. Completion evidence and cleanup

The R474 ProofPack must contain redacted pre/post configuration, stable and candidate revisions, immutable image digest and labels, traffic commands/results, stage observation timestamps, health/auth/Evidence/Trace results, leak/write zero counts, cost/capacity observations, rollback evidence if used, final repository state, manifest, and SHA256 register. Remove only task-owned local containers, networks and temporary artifacts. Create the single R474 Completion Report. Deployment remains incomplete until the owner provides exact new-digest R474 execution authorization and the post-deploy evidence passes.

## D42 Linux native observer operator surface

The D42 operator surface validates the platform-native Linux observer. It does not deploy, change traffic, access Production data, grant IAM, install WSL, or authorize the R474/R484 operations above. The authoritative remote validation workflow is `.github/workflows/qlib-linux-observer-acceptance.yml`. Its automatic trigger is restricted to the exact branch `r9znw-488d42-linux-observer-validation` and it runs only on GitHub-hosted `ubuntu-latest`.

The workflow exposes only these operator states: `READY`, `RUNNING`, `PASS`, `HOLD`, `FAIL`, and `ROLLED_BACK`. `ROLLED_BACK` means that a task-owned mutation was safely reversed; for the validation-only workflow its expected value is `ROLLED_BACK_NOT_REQUIRED_NO_MUTATION`. A job or step failure is never converted to `PASS`. There is no automatic rerun, sleep-only repair, deploy step, environment, OIDC permission, service credential, package installation, Docker operation, or external service call.

The workflow runs the fixed acceptance contract before the exact Linux/shared
observer regression tier:

- deterministic contract fixtures: 30 cases times 3, exactly 90 executions;
- actual Linux process campaign: 12 cases times 10, exactly 120 executions;
- fixed-seed stress campaign: 100 seeds, exactly 100 executions;
- required total: exactly 310 executions, with zero automatic retries.

The tracked `tools/qlib_linux_acceptance_test_tiers.json` manifest fixes the
native-Win32 diagnostic IDs, dual-path legacy-Windows expectation IDs, and
Linux-only IDs. It also binds the required public-Windows fail-closed test by
exact ID and seals the complete discovered-test count and ID digest. The
stdlib-only `tools/qlib_linux_acceptance_test_tiers.py` runner loads exactly the
four approved observer test modules, proves that every excluded ID still
exists, proves that every preserved Windows-native excluded class and method
has no skip or
expected-failure decorator or runtime flag, and executes the remaining
Linux/shared suite once. Any failure,
error, executed skip, expected failure, unexpected success, missing ID, or
manifest drift fails the workflow. Pattern-based filtering and result-driven
reruns are forbidden.

The fixed 310 campaign artifact contains only `acceptance-result.json` and
`manifest.json`. The separate platform-tier artifact contains only
`linux-shared-regression.json` and `windows-native-support.json`. The latter
records `WINDOWS_NATIVE_OBSERVER_NOT_APPROVED`; it is not evidence that the
native Windows observer passed. Both bundles are limited to counts, allowlisted
enums, and digests. The supervisor's `verify-artifact` command must reject any
extra file, raw field, missing count, digest mismatch, residual task-owned
process, unresolved wait, zombie, unrelated signal, timeout leak, Wrong-PASS,
or inconsistent terminal/seal. Raw PID, command line, environment, identity,
URL, token, response body, and raw log are forbidden in persistent evidence.

The workflow declares `workflow_dispatch`, but a workflow that exists only on the validation branch does not provide an active default-branch manual button. Until the exact workflow is adopted on the default branch, record:

```text
OPERATOR_MANUAL_DISPATCH_STATE=DECLARED_NOT_DEFAULT_BRANCH_ACTIVE
```

Do not claim manual operator readiness from the declaration alone. The validation-branch push run at the exact commit is the authoritative D42 remote execution evidence.

### D42C platform gate classification

The pre-repair boundary is classified
`MIXED_PLATFORM_BOUNDARY_GUARD_REQUIRED`. The native Win32 process-tree tests
are legacy diagnostics, but the public observer API and CLI also fell through
to the same unsupported launcher on Windows. The D42C guard now rejects native
Windows at every public start, poll, stop, and CLI dispatch before artifact or
process creation. POSIX dispatch continues only to the Linux supervisor.

The native Win32 test bodies and their expectations remain unchanged and
executable in a separately manifested diagnostic tier; they are not deleted,
skipped, xfail-marked, or weakened. Dual-path tests whose Windows branch asserts
the retired launcher remain required on Linux and are separated by exact ID
only from the Windows portable tier. Their support verdict remains
`WINDOWS_NATIVE_OBSERVER_NOT_APPROVED`. The platform artifact records both
groups separately. Do not report this split as an all-platform PASS.

The supported operator states are fixed as follows:

| Operator route | Support state | Selection contract |
|---|---|---|
| GitHub-hosted `ubuntu-latest` | Required Linux primary | Exact validation branch and exact-SHA workflow only |
| Windows PowerShell to existing WSL2 | Fallback; local result remains non-authoritative | Existing non-Docker WSL2 distribution only |
| Windows PowerShell without WSL2 | `HOLD` | No installation; direct operator to the GitHub workflow |
| Native Windows observer | `NOT_APPROVED` | Never selected automatically and never treated as Linux evidence |

No operator entrypoint may fall through from WSL2 absence to native Win32
execution. Native Win32 support requires a separate future approval and cannot
be inferred from the portable shared regression.

### Windows fallback without Linux commands

A Windows operator may run the tracked `tools/qlib_linux_observer_operator.ps1` entrypoint; the operator does not enter Linux commands. The entrypoint checks only for an already installed non-Docker WSL2 distribution and explicitly excludes `docker-desktop` and `docker-desktop-data`. It never enables or installs WSL2, a distribution, Docker, a VM, a package, or a credential.

If WSL2 or a distribution is absent, the entrypoint returns one sanitized line with `HOLD`, `NO_INSTALL_ATTEMPTED`, and the single next action `RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW`. If WSL2 is present, the entrypoint invokes the same fixed 90/120/100 supervisor acceptance and local-non-authoritative closed-artifact verifier while suppressing raw process output, then removes exactly that nonce-bound two-file temporary bundle only after semantic verification passes. A failed semantic verification preserves the sanitized bundle and returns `FAIL` without retry. A verified local WSL result remains `HOLD` with `LOCAL_WSL_ACCEPTANCE_VERIFIED_NOT_GITHUB` and evidence scope `LOCAL_WSL_FALLBACK_NOT_GITHUB`; it never substitutes for the exact-SHA GitHub-hosted acceptance result. Cleanup failure is `FAIL` and can never be promoted to local success.

The deterministic absence-path smoke is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/qlib_linux_observer_operator.ps1 -SmokeAbsencePath
```

Expected single-line state is `HOLD` with `cause=WSL2_NOT_AVAILABLE`, `safe_action=NO_INSTALL_ATTEMPTED`, `next_action=RUN_GITHUB_ACTIONS_OPERATOR_WORKFLOW`, and `evidence_scope=ABSENCE_PATH_SMOKE`. Exit zero in this explicit smoke mode means only that the safe absence behavior was verified; it is not Linux acceptance evidence.

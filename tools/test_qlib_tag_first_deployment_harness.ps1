Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'qlib_tag_first_deployment_harness.ps1')

$script:TestResults = @()
$script:Evidence = [ordered]@{}

function Assert-TestCondition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw [System.InvalidOperationException]::new($Message)
    }
}

function Assert-TestEqual {
    param(
        [AllowNull()]
        $Expected,

        [AllowNull()]
        $Actual,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if ($Expected -cne $Actual) {
        throw [System.InvalidOperationException]::new("$Message Expected=[$Expected] Actual=[$Actual]")
    }
}

function Invoke-HarnessTestCase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Body
    )

    try {
        & $Body
        $script:TestResults += [ordered]@{
            name   = $Name
            status = 'PASS'
        }
    }
    catch {
        $script:TestResults += [ordered]@{
            name            = $Name
            status          = 'FAIL'
            exception_type  = $_.Exception.GetType().FullName
            sanitized_error = $_.Exception.Message
        }
    }
}

function New-MockCommandResult {
    param(
        [int]$ExitCode = 0,
        [string]$StdOut = '',
        [bool]$PartialResult = $false
    )

    return [pscustomobject]@{
        ExitCode     = $ExitCode
        StdOut       = $StdOut
        PartialResult = $PartialResult
    }
}

$project = 'synthetic-project'
$region = 'asia-northeast1'
$service = 'synthetic-skillup-runtime'
$tag = 'r486-private'
$image = 'asia-northeast1-docker.pkg.dev/synthetic-project/synthetic-repository/synthetic-runtime@sha256:' + ('a' * 64)
$beforeRevision = 'synthetic-runtime-00001-aaa'
$candidateRevision = 'synthetic-runtime-00002-bbb'

$beforeJson = [ordered]@{
    status = [ordered]@{
        latestCreatedRevisionName = $beforeRevision
        traffic = @(
            [ordered]@{
                revisionName = $beforeRevision
                percent = 100
            }
        )
    }
} | ConvertTo-Json -Depth 8 -Compress

$afterJson = [ordered]@{
    status = [ordered]@{
        latestCreatedRevisionName = $candidateRevision
        traffic = @(
            [ordered]@{
                revisionName = $beforeRevision
                percent = 100
            },
            [ordered]@{
                revisionName = $candidateRevision
                tag = $tag
            }
        )
    }
} | ConvertTo-Json -Depth 8 -Compress

Invoke-HarnessTestCase -Name 'success_branch_and_argument_contract' -Body {
    $state = [ordered]@{
        invocation_count = 0
        deploy_invocation_count = 0
        deploy_arguments = @()
    }

    $runner = {
        param([string[]]$Arguments, [string]$Operation)

        $state.invocation_count++
        switch ($Operation) {
            'resolve active project' {
                return New-MockCommandResult -StdOut $project
            }
            'describe service before deploy' {
                return New-MockCommandResult -StdOut $beforeJson
            }
            'deploy no-traffic tagged candidate' {
                $state.deploy_invocation_count++
                $state.deploy_arguments = @($Arguments)
                return New-MockCommandResult
            }
            'describe service after deploy' {
                return New-MockCommandResult -StdOut $afterJson
            }
            default {
                return New-MockCommandResult -ExitCode 99
            }
        }
    }

    $result = Invoke-QlibTagFirstDeployment `
        -Project $project `
        -Region $region `
        -Service $service `
        -Image $image `
        -Tag $tag `
        -CommandRunner $runner `
        -RunnerKind 'MOCK'

    $expectedArguments = @(
        'run', 'deploy', $service,
        '--project', $project,
        '--region', $region,
        '--image', $image,
        '--no-traffic',
        '--port=8080',
        '--min=0',
        '--max=2',
        '--min-instances=0',
        '--max-instances=2',
        '--cpu=1',
        '--memory=512Mi',
        '--concurrency=80',
        '--timeout=300',
        '--tag', $tag,
        '--quiet'
    )

    $separator = [string][char]31
    Assert-TestEqual -Expected 1 -Actual $state.deploy_invocation_count -Message 'Mock deploy invocation count mismatch.'
    Assert-TestEqual -Expected ($expectedArguments -join $separator) -Actual ($state.deploy_arguments -join $separator) -Message 'Deploy argument array mismatch.'
    Assert-TestEqual -Expected 'PASS' -Actual $result.status -Message 'Success result was not PASS.'
    Assert-TestEqual -Expected 'MOCK' -Actual $result.runner_kind -Message 'Runner kind mismatch.'
    Assert-TestEqual -Expected 'NOT_GRANTED_MOCK_ONLY' -Actual $result.candidate_creation_fact -Message 'Mock candidate fact was promoted.'
    Assert-TestEqual -Expected 'NOT_GRANTED_MOCK_ONLY' -Actual $result.tag_creation_fact -Message 'Mock tag fact was promoted.'
    Assert-TestEqual -Expected 0 -Actual $result.existing_candidate_reuse_count -Message 'Existing candidate reuse was recorded.'
    Assert-TestEqual -Expected 0 -Actual $result.candidate_traffic_percent -Message 'Candidate received percentage traffic.'
    Assert-TestEqual -Expected 0 -Actual $result.raw_url_persistence_count -Message 'Raw URL persistence was recorded.'
    Assert-TestEqual -Expected 0 -Actual $result.raw_token_persistence_count -Message 'Raw token persistence was recorded.'
    Assert-TestEqual -Expected 0 -Actual $result.identity_persistence_count -Message 'Identity persistence was recorded.'
    Assert-TestEqual -Expected 0 -Actual $result.raw_response_persistence_count -Message 'Raw response persistence was recorded.'
    Assert-TestCondition -Condition $result.deploy_statement_reached -Message 'Deploy statement was not reached in mock success path.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--no-traffic') -Message 'Missing --no-traffic.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--min=0') -Message 'Missing service min contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--max=2') -Message 'Missing service max contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--min-instances=0') -Message 'Missing revision min contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--max-instances=2') -Message 'Missing immutable maxScale contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--cpu=1') -Message 'Missing CPU contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--memory=512Mi') -Message 'Missing memory contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--concurrency=80') -Message 'Missing concurrency contract.'
    Assert-TestCondition -Condition ($state.deploy_arguments -contains '--timeout=300') -Message 'Missing timeout contract.'
    Assert-TestCondition -Condition (-not ($state.deploy_arguments -contains '--allow-unauthenticated')) -Message 'Public access flag was present.'
    Assert-TestCondition -Condition (-not ($state.deploy_arguments -contains '--no-allow-unauthenticated')) -Message 'IAM-mutating private flag was present.'

    $tagIndex = [array]::IndexOf([object[]]$state.deploy_arguments, '--tag')
    Assert-TestCondition -Condition ($tagIndex -ge 0) -Message 'Private tag flag was absent.'
    Assert-TestEqual -Expected $tag -Actual $state.deploy_arguments[$tagIndex + 1] -Message 'Private tag value mismatch.'

    $script:Evidence.success = [ordered]@{
        helper_semantic_execution = 'PASS'
        dependency_context = $result.dependency_context
        deploy_statement_reached = $result.deploy_statement_reached
        mock_deploy_invocation_count = $state.deploy_invocation_count
        candidate_tag_argument_contract = 'PASS'
        no_traffic_contract = $result.no_traffic_contract
        private_tag_contract = $result.private_tag_contract
        existing_candidate_reuse_count = $result.existing_candidate_reuse_count
        candidate_creation_fact = $result.candidate_creation_fact
        tag_creation_fact = $result.tag_creation_fact
        synthetic_deploy_arguments = $state.deploy_arguments
    }
}

Invoke-HarnessTestCase -Name 'dependency_failure_stops_before_deploy' -Body {
    $state = [ordered]@{ deploy_invocation_count = 0 }
    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        if ($Operation -ceq 'deploy no-traffic tagged candidate') {
            $state.deploy_invocation_count++
        }
        return New-MockCommandResult -ExitCode 7
    }

    $caught = $null
    try {
        $null = Invoke-QlibTagFirstDeployment -Project $project -Region $region -Service $service -Image $image -Tag $tag -CommandRunner $runner -RunnerKind 'MOCK'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Dependency failure did not throw.'
    Assert-TestEqual -Expected 'System.InvalidOperationException' -Actual $caught.GetType().FullName -Message 'Dependency failure exception type mismatch.'
    Assert-TestEqual -Expected 0 -Actual $state.deploy_invocation_count -Message 'Dependency failure reached deploy.'
    $script:Evidence.dependency_failure = [ordered]@{ deploy_invocation_count = $state.deploy_invocation_count; exception_type = $caught.GetType().FullName; status = 'FAIL_CLOSED' }
}

Invoke-HarnessTestCase -Name 'error_branch_return_semantics' -Body {
    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        return New-MockCommandResult -ExitCode 11
    }

    $caught = $null
    try {
        $null = Invoke-QlibInjectedCommand -CommandRunner $runner -Arguments @('synthetic', 'dependency') -Operation 'synthetic dependency failure'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Injected command error branch did not throw.'
    Assert-TestEqual -Expected 'System.InvalidOperationException' -Actual $caught.GetType().FullName -Message 'Error branch return semantics regressed.'
    Assert-TestCondition -Condition ($caught.GetType().FullName -cne 'System.Management.Automation.CommandNotFoundException') -Message 'Error branch was tokenized as a command.'
    $script:Evidence.error_branch_return_semantics = [ordered]@{ exception_type = $caught.GetType().FullName; command_not_found = $false; status = 'PASS' }
}

Invoke-HarnessTestCase -Name 'invalid_argument_stops_before_runner' -Body {
    $state = [ordered]@{ invocation_count = 0; deploy_invocation_count = 0 }
    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        $state.invocation_count++
        if ($Operation -ceq 'deploy no-traffic tagged candidate') {
            $state.deploy_invocation_count++
        }
        return New-MockCommandResult
    }

    $caught = $null
    try {
        $null = Invoke-QlibTagFirstDeployment -Project $project -Region $region -Service $service -Image $image -Tag ' ' -CommandRunner $runner -RunnerKind 'MOCK'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Invalid argument did not throw.'
    Assert-TestEqual -Expected 'System.ArgumentException' -Actual $caught.GetType().FullName -Message 'Invalid argument exception type mismatch.'
    Assert-TestEqual -Expected 0 -Actual $state.invocation_count -Message 'Invalid argument reached command runner.'
    Assert-TestEqual -Expected 0 -Actual $state.deploy_invocation_count -Message 'Invalid argument reached deploy.'
    $script:Evidence.invalid_argument = [ordered]@{ runner_invocation_count = $state.invocation_count; deploy_invocation_count = $state.deploy_invocation_count; status = 'FAIL_CLOSED' }
}

Invoke-HarnessTestCase -Name 'existing_tag_stops_candidate_reuse' -Body {
    $state = [ordered]@{ deploy_invocation_count = 0 }
    $taggedBeforeJson = [ordered]@{
        status = [ordered]@{
            latestCreatedRevisionName = $beforeRevision
            traffic = @(
                [ordered]@{
                    revisionName = $beforeRevision
                    percent = 100
                    tag = $tag
                }
            )
        }
    } | ConvertTo-Json -Depth 8 -Compress

    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        switch ($Operation) {
            'resolve active project' { return New-MockCommandResult -StdOut $project }
            'describe service before deploy' { return New-MockCommandResult -StdOut $taggedBeforeJson }
            'deploy no-traffic tagged candidate' {
                $state.deploy_invocation_count++
                return New-MockCommandResult
            }
        }
    }

    $caught = $null
    try {
        $null = Invoke-QlibTagFirstDeployment -Project $project -Region $region -Service $service -Image $image -Tag $tag -CommandRunner $runner -RunnerKind 'MOCK'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Existing tag did not stop the harness.'
    Assert-TestEqual -Expected 'System.InvalidOperationException' -Actual $caught.GetType().FullName -Message 'Existing tag exception type mismatch.'
    Assert-TestEqual -Expected 0 -Actual $state.deploy_invocation_count -Message 'Existing tag reached deploy.'
    $script:Evidence.existing_tag = [ordered]@{ deploy_invocation_count = $state.deploy_invocation_count; existing_candidate_reuse_count = 0; status = 'FAIL_CLOSED' }
}

Invoke-HarnessTestCase -Name 'deploy_nonzero_is_not_success' -Body {
    $state = [ordered]@{ deploy_invocation_count = 0; after_readback_count = 0 }
    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        switch ($Operation) {
            'resolve active project' { return New-MockCommandResult -StdOut $project }
            'describe service before deploy' { return New-MockCommandResult -StdOut $beforeJson }
            'deploy no-traffic tagged candidate' {
                $state.deploy_invocation_count++
                return New-MockCommandResult -ExitCode 9
            }
            'describe service after deploy' {
                $state.after_readback_count++
                return New-MockCommandResult -StdOut $afterJson
            }
        }
    }

    $caught = $null
    try {
        $null = Invoke-QlibTagFirstDeployment -Project $project -Region $region -Service $service -Image $image -Tag $tag -CommandRunner $runner -RunnerKind 'MOCK'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Nonzero deploy did not throw.'
    Assert-TestEqual -Expected 'System.InvalidOperationException' -Actual $caught.GetType().FullName -Message 'Nonzero deploy exception type mismatch.'
    Assert-TestEqual -Expected 1 -Actual $state.deploy_invocation_count -Message 'Nonzero deploy count mismatch.'
    Assert-TestEqual -Expected 0 -Actual $state.after_readback_count -Message 'Nonzero deploy continued to readback.'
    $script:Evidence.deploy_failure = [ordered]@{ deploy_invocation_count = $state.deploy_invocation_count; after_readback_count = $state.after_readback_count; status = 'FAIL_CLOSED' }
}

Invoke-HarnessTestCase -Name 'readback_failure_is_not_success' -Body {
    $state = [ordered]@{ deploy_invocation_count = 0; after_readback_count = 0 }
    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        switch ($Operation) {
            'resolve active project' { return New-MockCommandResult -StdOut $project }
            'describe service before deploy' { return New-MockCommandResult -StdOut $beforeJson }
            'deploy no-traffic tagged candidate' {
                $state.deploy_invocation_count++
                return New-MockCommandResult
            }
            'describe service after deploy' {
                $state.after_readback_count++
                return New-MockCommandResult -ExitCode 8
            }
        }
    }

    $caught = $null
    try {
        $null = Invoke-QlibTagFirstDeployment -Project $project -Region $region -Service $service -Image $image -Tag $tag -CommandRunner $runner -RunnerKind 'MOCK'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Readback failure did not throw.'
    Assert-TestEqual -Expected 'System.InvalidOperationException' -Actual $caught.GetType().FullName -Message 'Readback exception type mismatch.'
    Assert-TestEqual -Expected 1 -Actual $state.deploy_invocation_count -Message 'Readback failure deploy count mismatch.'
    Assert-TestEqual -Expected 1 -Actual $state.after_readback_count -Message 'Readback failure count mismatch.'
    $script:Evidence.readback_failure = [ordered]@{ deploy_invocation_count = $state.deploy_invocation_count; after_readback_count = $state.after_readback_count; status = 'FAIL_CLOSED_NOT_PASS' }
}

Invoke-HarnessTestCase -Name 'partial_readback_is_not_success' -Body {
    $state = [ordered]@{ deploy_invocation_count = 0; partial_readback_count = 0 }
    $runner = {
        param([string[]]$Arguments, [string]$Operation)
        switch ($Operation) {
            'resolve active project' { return New-MockCommandResult -StdOut $project }
            'describe service before deploy' { return New-MockCommandResult -StdOut $beforeJson }
            'deploy no-traffic tagged candidate' {
                $state.deploy_invocation_count++
                return New-MockCommandResult
            }
            'describe service after deploy' {
                $state.partial_readback_count++
                return New-MockCommandResult -StdOut $afterJson -PartialResult $true
            }
        }
    }

    $caught = $null
    try {
        $null = Invoke-QlibTagFirstDeployment -Project $project -Region $region -Service $service -Image $image -Tag $tag -CommandRunner $runner -RunnerKind 'MOCK'
    }
    catch {
        $caught = $_.Exception
    }

    Assert-TestCondition -Condition ($null -ne $caught) -Message 'Partial readback did not throw.'
    Assert-TestEqual -Expected 'System.InvalidOperationException' -Actual $caught.GetType().FullName -Message 'Partial result exception type mismatch.'
    Assert-TestEqual -Expected 1 -Actual $state.deploy_invocation_count -Message 'Partial readback deploy count mismatch.'
    Assert-TestEqual -Expected 1 -Actual $state.partial_readback_count -Message 'Partial readback count mismatch.'
    $script:Evidence.partial_readback = [ordered]@{ deploy_invocation_count = $state.deploy_invocation_count; partial_readback_count = $state.partial_readback_count; status = 'FAIL_CLOSED_NOT_PASS' }
}

$failedTests = @($script:TestResults | Where-Object { $_.status -cne 'PASS' })
$summary = [ordered]@{
    task_id = 'R9ZNW-486'
    test_kind = 'LOCAL_SEMANTIC_MOCK_ONLY'
    status = if ($failedTests.Count -eq 0) { 'PASS' } else { 'FAIL' }
    test_count = $script:TestResults.Count
    failure_count = $failedTests.Count
    mock_deploy_invocation_count_success_path = if ($script:Evidence.Contains('success')) { $script:Evidence.success.mock_deploy_invocation_count } else { 0 }
    real_cloud_command_count = 0
    cloud_mutation_count = 0
    existing_candidate_reuse_count = 0
    raw_url_token_identity_response_persistence_count = 0
    candidate_creation_fact = 'NOT_GRANTED_MOCK_ONLY'
    tag_creation_fact = 'NOT_GRANTED_MOCK_ONLY'
    results = $script:TestResults
    evidence = $script:Evidence
}

$summary | ConvertTo-Json -Depth 20
if ($failedTests.Count -ne 0) {
    exit 1
}

Set-StrictMode -Version Latest

function Assert-QlibDeploymentArgument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Value,

        [Parameter(Mandatory = $true)]
        [string]$Pattern
    )

    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -cnotmatch $Pattern) {
        throw [System.ArgumentException]::new("Invalid deployment argument: $Name")
    }
}

function Invoke-QlibInjectedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$CommandRunner,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Operation
    )

    $response = & $CommandRunner -Arguments $Arguments -Operation $Operation
    if ($null -eq $response) {
        throw [System.InvalidOperationException]::new("Command runner returned no result: $Operation")
    }

    $exitCodeProperty = $response.PSObject.Properties['ExitCode']
    if ($null -eq $exitCodeProperty) {
        throw [System.InvalidOperationException]::new("Command runner omitted ExitCode: $Operation")
    }

    $partialResultProperty = $response.PSObject.Properties['PartialResult']
    if ($null -ne $partialResultProperty -and [bool]$partialResultProperty.Value) {
        throw [System.InvalidOperationException]::new("Command runner returned a partial result: $Operation")
    }

    $exitCode = [int]$exitCodeProperty.Value
    if ($exitCode -ne 0) {
        throw [System.InvalidOperationException]::new("Command runner failed: $Operation")
    }

    $standardOutputProperty = $response.PSObject.Properties['StdOut']
    $text = if ($null -eq $standardOutputProperty) {
        ''
    }
    else {
        [string]$standardOutputProperty.Value
    }

    return $text.Trim()
}

function ConvertFrom-QlibInjectedJson {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$CommandRunner,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Operation
    )

    $jsonText = Invoke-QlibInjectedCommand `
        -CommandRunner $CommandRunner `
        -Arguments $Arguments `
        -Operation $Operation

    if ([string]::IsNullOrWhiteSpace($jsonText)) {
        throw [System.InvalidOperationException]::new("Command runner returned empty JSON: $Operation")
    }

    try {
        $value = $jsonText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw [System.InvalidOperationException]::new("Command runner returned invalid JSON: $Operation")
    }

    return $value
}

function New-QlibTagFirstDeployArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Project,

        [Parameter(Mandatory = $true)]
        [string]$Region,

        [Parameter(Mandatory = $true)]
        [string]$Service,

        [Parameter(Mandatory = $true)]
        [string]$Image,

        [Parameter(Mandatory = $true)]
        [string]$Tag
    )

    Assert-QlibDeploymentArgument -Name 'Project' -Value $Project -Pattern '^[a-z][a-z0-9-]{4,62}[a-z0-9]$'
    Assert-QlibDeploymentArgument -Name 'Region' -Value $Region -Pattern '^[a-z]+-[a-z]+[0-9]+$'
    Assert-QlibDeploymentArgument -Name 'Service' -Value $Service -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'
    Assert-QlibDeploymentArgument -Name 'Image' -Value $Image -Pattern '^[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$'
    Assert-QlibDeploymentArgument -Name 'Tag' -Value $Tag -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'

    $arguments = @(
        'run',
        'deploy',
        $Service,
        '--project',
        $Project,
        '--region',
        $Region,
        '--image',
        $Image,
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
        '--tag',
        $Tag,
        '--quiet'
    )

    return $arguments
}

function Get-QlibRevisionTrafficPercent {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Traffic,

        [Parameter(Mandatory = $true)]
        [string]$Revision
    )

    $total = 0
    foreach ($item in $Traffic) {
        if ([string]$item.revisionName -ceq $Revision) {
            $percentProperty = $item.PSObject.Properties['percent']
            if ($null -ne $percentProperty -and $null -ne $percentProperty.Value) {
                $total += [int]$percentProperty.Value
            }
        }
    }

    return $total
}

function New-QlibCommandCounter {
    return [pscustomobject][ordered]@{
        ReadOnlyCommandCount = 0
        MutationCommandCount = 0
    }
}

function Invoke-QlibCountedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$CommandRunner,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Operation,

        [Parameter(Mandatory = $true)]
        [ValidateSet('READ_ONLY', 'MUTATION')]
        [string]$Classification,

        [Parameter(Mandatory = $true)]
        [psobject]$CommandCounter
    )

    if ($Classification -ceq 'MUTATION') {
        if ([int]$CommandCounter.MutationCommandCount -ne 0) {
            throw [System.InvalidOperationException]::new('A second mutation command is forbidden in one rollback operation.')
        }
        $CommandCounter.MutationCommandCount = [int]$CommandCounter.MutationCommandCount + 1
    }
    else {
        $CommandCounter.ReadOnlyCommandCount = [int]$CommandCounter.ReadOnlyCommandCount + 1
    }

    return Invoke-QlibInjectedCommand `
        -CommandRunner $CommandRunner `
        -Arguments $Arguments `
        -Operation $Operation
}

function ConvertFrom-QlibCountedJson {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$CommandRunner,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Operation,

        [Parameter(Mandatory = $true)]
        [psobject]$CommandCounter
    )

    $jsonText = Invoke-QlibCountedCommand `
        -CommandRunner $CommandRunner `
        -Arguments $Arguments `
        -Operation $Operation `
        -Classification 'READ_ONLY' `
        -CommandCounter $CommandCounter

    if ([string]::IsNullOrWhiteSpace($jsonText)) {
        throw [System.InvalidOperationException]::new("Command runner returned empty JSON: $Operation")
    }

    try {
        return $jsonText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw [System.InvalidOperationException]::new("Command runner returned invalid JSON: $Operation")
    }
}

function New-QlibTagRollbackArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Project,

        [Parameter(Mandatory = $true)]
        [string]$Region,

        [Parameter(Mandatory = $true)]
        [string]$Service,

        [Parameter(Mandatory = $true)]
        [string]$Tag
    )

    Assert-QlibDeploymentArgument -Name 'Project' -Value $Project -Pattern '^[a-z][a-z0-9-]{4,62}[a-z0-9]$'
    Assert-QlibDeploymentArgument -Name 'Region' -Value $Region -Pattern '^[a-z]+-[a-z]+[0-9]+$'
    Assert-QlibDeploymentArgument -Name 'Service' -Value $Service -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'
    Assert-QlibDeploymentArgument -Name 'Tag' -Value $Tag -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'

    return @(
        'run', 'services', 'update-traffic', $Service,
        '--remove-tags', $Tag,
        '--project', $Project,
        '--region', $Region,
        '--quiet'
    )
}

function Test-QlibRollbackReadback {
    param(
        [Parameter(Mandatory = $true)]
        [object]$ServiceState,

        [Parameter(Mandatory = $true)]
        [string]$Tag,

        [Parameter(Mandatory = $true)]
        [string]$CandidateRevision,

        [Parameter(Mandatory = $true)]
        [string]$StableRevision,

        [string[]]$PreservedRevisions = @(),

        [Parameter(Mandatory = $true)]
        [bool]$ExpectTag
    )

    $traffic = @($ServiceState.status.traffic)
    $tagTargets = @($traffic | Where-Object {
        $tagProperty = $_.PSObject.Properties['tag']
        $null -ne $tagProperty -and [string]$tagProperty.Value -ceq $Tag
    })

    if ($ExpectTag) {
        if ($tagTargets.Count -ne 1 -or [string]$tagTargets[0].revisionName -cne $CandidateRevision) {
            throw [System.InvalidOperationException]::new('Prerollback readback did not bind the tag to the expected candidate.')
        }
    }
    elseif ($tagTargets.Count -ne 0) {
        throw [System.InvalidOperationException]::new('Postrollback readback still contains the removed tag.')
    }

    if ((Get-QlibRevisionTrafficPercent -Traffic $traffic -Revision $StableRevision) -ne 100) {
        throw [System.InvalidOperationException]::new('Stable revision is not at 100 percent traffic.')
    }
    if ((Get-QlibRevisionTrafficPercent -Traffic $traffic -Revision $CandidateRevision) -ne 0) {
        throw [System.InvalidOperationException]::new('Candidate revision received percentage traffic.')
    }
    foreach ($revision in $PreservedRevisions) {
        if ((Get-QlibRevisionTrafficPercent -Traffic $traffic -Revision $revision) -ne 0) {
            throw [System.InvalidOperationException]::new('A preserved nonstable revision received percentage traffic.')
        }
    }

    $knownRevisions = @($StableRevision, $CandidateRevision) + @($PreservedRevisions)
    $unexpectedPositive = @($traffic | Where-Object {
        $percentProperty = $_.PSObject.Properties['percent']
        $percent = if ($null -eq $percentProperty -or $null -eq $percentProperty.Value) { 0 } else { [int]$percentProperty.Value }
        $percent -gt 0 -and $knownRevisions -cnotcontains [string]$_.revisionName
    }).Count
    if ($unexpectedPositive -ne 0) {
        throw [System.InvalidOperationException]::new('Unexpected revision received positive traffic.')
    }

    return [pscustomobject][ordered]@{
        tag_target_count = $tagTargets.Count
        stable_traffic_percent = 100
        candidate_traffic_percent = 0
        unexpected_positive_traffic_count = 0
    }
}

function Invoke-QlibTagRollback {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Project,

        [Parameter(Mandatory = $true)]
        [string]$Region,

        [Parameter(Mandatory = $true)]
        [string]$Service,

        [Parameter(Mandatory = $true)]
        [string]$Tag,

        [Parameter(Mandatory = $true)]
        [string]$CandidateRevision,

        [Parameter(Mandatory = $true)]
        [string]$StableRevision,

        [string[]]$PreservedRevisions = @(),

        [Parameter(Mandatory = $true)]
        [scriptblock]$CommandRunner,

        [Parameter(Mandatory = $true)]
        [ValidateSet('MOCK', 'EXTERNAL')]
        [string]$RunnerKind,

        [psobject]$CommandCounter = (New-QlibCommandCounter),

        [scriptblock]$Postprocessor
    )

    $rollbackArguments = @(New-QlibTagRollbackArguments -Project $Project -Region $Region -Service $Service -Tag $Tag)
    Assert-QlibDeploymentArgument -Name 'CandidateRevision' -Value $CandidateRevision -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'
    Assert-QlibDeploymentArgument -Name 'StableRevision' -Value $StableRevision -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'
    foreach ($revision in $PreservedRevisions) {
        Assert-QlibDeploymentArgument -Name 'PreservedRevision' -Value $revision -Pattern '^[a-z][a-z0-9-]{0,61}[a-z0-9]$'
    }

    $before = ConvertFrom-QlibCountedJson `
        -CommandRunner $CommandRunner `
        -Arguments @('run', 'services', 'describe', $Service, '--project', $Project, '--region', $Region, '--format=json', '--quiet') `
        -Operation 'describe service before tag rollback' `
        -CommandCounter $CommandCounter
    $null = Test-QlibRollbackReadback -ServiceState $before -Tag $Tag -CandidateRevision $CandidateRevision -StableRevision $StableRevision -PreservedRevisions $PreservedRevisions -ExpectTag $true

    $null = Invoke-QlibCountedCommand `
        -CommandRunner $CommandRunner `
        -Arguments $rollbackArguments `
        -Operation 'remove private tag' `
        -Classification 'MUTATION' `
        -CommandCounter $CommandCounter

    $after = ConvertFrom-QlibCountedJson `
        -CommandRunner $CommandRunner `
        -Arguments @('run', 'services', 'describe', $Service, '--project', $Project, '--region', $Region, '--format=json', '--quiet') `
        -Operation 'describe service after tag rollback' `
        -CommandCounter $CommandCounter
    $readback = Test-QlibRollbackReadback -ServiceState $after -Tag $Tag -CandidateRevision $CandidateRevision -StableRevision $StableRevision -PreservedRevisions $PreservedRevisions -ExpectTag $false

    $facts = [pscustomobject][ordered]@{
        runner_kind = $RunnerKind
        mutation_status = 'PASS'
        independent_readback_status = 'PASS'
        tag_removal_status = 'PASS'
        readback = $readback
    }

    $postprocessingStatus = 'PASS'
    $sanitizedErrorClass = 'NONE'
    try {
        if ($null -ne $Postprocessor) {
            $processed = & $Postprocessor -Facts $facts
            if ($null -eq $processed) {
                throw [System.InvalidOperationException]::new('Postprocessor returned no result.')
            }
        }
        else {
            $null = $facts | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json -ErrorAction Stop
        }
    }
    catch {
        $postprocessingStatus = 'FAIL'
        $sanitizedErrorClass = $_.Exception.GetType().Name
    }

    return [pscustomobject][ordered]@{
        status = if ($postprocessingStatus -ceq 'PASS') { 'PASS' } else { 'FAIL_POSTPROCESSING' }
        runner_kind = $RunnerKind
        mutation_status = 'PASS'
        independent_readback_status = 'PASS'
        postprocessing_status = $postprocessingStatus
        sanitized_postprocessing_error_class = $sanitizedErrorClass
        mutation_command_count = [int]$CommandCounter.MutationCommandCount
        read_only_command_count = [int]$CommandCounter.ReadOnlyCommandCount
        mutation_retry_count = 0
        stable_traffic_percent = $readback.stable_traffic_percent
        candidate_traffic_percent = $readback.candidate_traffic_percent
        tag_target_count = $readback.tag_target_count
        unexpected_positive_traffic_count = $readback.unexpected_positive_traffic_count
        raw_url_persistence_count = 0
        raw_token_persistence_count = 0
        identity_persistence_count = 0
        raw_response_persistence_count = 0
    }
}

function Invoke-QlibTagFirstDeployment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Project,

        [Parameter(Mandatory = $true)]
        [string]$Region,

        [Parameter(Mandatory = $true)]
        [string]$Service,

        [Parameter(Mandatory = $true)]
        [string]$Image,

        [Parameter(Mandatory = $true)]
        [string]$Tag,

        [Parameter(Mandatory = $true)]
        [scriptblock]$CommandRunner,

        [Parameter(Mandatory = $true)]
        [ValidateSet('MOCK', 'EXTERNAL')]
        [string]$RunnerKind
    )

    $deployArguments = @(New-QlibTagFirstDeployArguments `
        -Project $Project `
        -Region $Region `
        -Service $Service `
        -Image $Image `
        -Tag $Tag)

    $activeProject = Invoke-QlibInjectedCommand `
        -CommandRunner $CommandRunner `
        -Arguments @('config', 'get-value', 'project', '--quiet') `
        -Operation 'resolve active project'

    if ($activeProject -cne $Project) {
        throw [System.InvalidOperationException]::new('Active project does not match the validated project argument.')
    }

    $before = ConvertFrom-QlibInjectedJson `
        -CommandRunner $CommandRunner `
        -Arguments @('run', 'services', 'describe', $Service, '--project', $Project, '--region', $Region, '--format=json', '--quiet') `
        -Operation 'describe service before deploy'

    $beforeLatestRevision = [string]$before.status.latestCreatedRevisionName
    if ([string]::IsNullOrWhiteSpace($beforeLatestRevision)) {
        throw [System.InvalidOperationException]::new('Predeploy readback omitted the latest revision.')
    }

    $beforeTraffic = @($before.status.traffic)
    $existingTagCount = @($beforeTraffic | Where-Object {
        $tagProperty = $_.PSObject.Properties['tag']
        $null -ne $tagProperty -and [string]$tagProperty.Value -ceq $Tag
    }).Count
    if ($existingTagCount -ne 0) {
        throw [System.InvalidOperationException]::new('The requested private tag already exists; candidate reuse is forbidden.')
    }

    $null = Invoke-QlibInjectedCommand `
        -CommandRunner $CommandRunner `
        -Arguments $deployArguments `
        -Operation 'deploy no-traffic tagged candidate'

    $after = ConvertFrom-QlibInjectedJson `
        -CommandRunner $CommandRunner `
        -Arguments @('run', 'services', 'describe', $Service, '--project', $Project, '--region', $Region, '--format=json', '--quiet') `
        -Operation 'describe service after deploy'

    $candidateRevision = [string]$after.status.latestCreatedRevisionName
    if ([string]::IsNullOrWhiteSpace($candidateRevision) -or $candidateRevision -ceq $beforeLatestRevision) {
        throw [System.InvalidOperationException]::new('Postdeploy readback did not identify a fresh candidate revision.')
    }

    $afterTraffic = @($after.status.traffic)
    $tagTargets = @($afterTraffic | Where-Object {
        $tagProperty = $_.PSObject.Properties['tag']
        $null -ne $tagProperty -and [string]$tagProperty.Value -ceq $Tag
    })
    if ($tagTargets.Count -ne 1 -or [string]$tagTargets[0].revisionName -cne $candidateRevision) {
        throw [System.InvalidOperationException]::new('Postdeploy readback did not bind the private tag to the fresh candidate.')
    }

    $candidateTrafficPercent = Get-QlibRevisionTrafficPercent `
        -Traffic $afterTraffic `
        -Revision $candidateRevision

    if ($candidateTrafficPercent -ne 0) {
        throw [System.InvalidOperationException]::new('The fresh tagged candidate received percentage traffic.')
    }

    $candidateCreationFact = if ($RunnerKind -ceq 'MOCK') {
        'NOT_GRANTED_MOCK_ONLY'
    }
    else {
        'READBACK_VERIFIED'
    }

    return [pscustomobject][ordered]@{
        status                           = 'PASS'
        runner_kind                      = $RunnerKind
        dependency_context               = 'PASS'
        deploy_statement_reached         = $true
        no_traffic_contract              = 'PASS'
        private_tag_contract             = 'PASS'
        candidate_revision_readback      = $candidateRevision
        candidate_tag_readback           = $Tag
        candidate_traffic_percent        = $candidateTrafficPercent
        existing_candidate_reuse_count   = 0
        candidate_creation_fact          = $candidateCreationFact
        tag_creation_fact                = $candidateCreationFact
        raw_url_persistence_count         = 0
        raw_token_persistence_count       = 0
        identity_persistence_count        = 0
        raw_response_persistence_count    = 0
    }
}

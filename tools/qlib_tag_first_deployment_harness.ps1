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

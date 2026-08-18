param(
    [Parameter(Mandatory = $true)]
    [datetime]$RunAt
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$analysisScript = Join-Path $PSScriptRoot "resume_remaining_2023_agents.ps1"
$stateDirectory = Join-Path $projectRoot "artifacts\logs"
$statePath = Join-Path $stateDirectory "resume_remaining_2023_agents_schedule.json"

New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null

function Write-ScheduleState {
    param(
        [string]$State,
        [Nullable[int]]$ExitCode = $null,
        [string]$ErrorMessage = $null
    )

    [ordered]@{
        state = $State
        scheduled_at = $RunAt.ToString("o")
        updated_at = (Get-Date).ToString("o")
        process_id = $PID
        exit_code = $ExitCode
        error = $ErrorMessage
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
}

Write-ScheduleState -State "waiting"

while ((Get-Date) -lt $RunAt) {
    $remainingSeconds = [math]::Ceiling(($RunAt - (Get-Date)).TotalSeconds)
    Start-Sleep -Seconds ([math]::Min(60, [math]::Max(1, $remainingSeconds)))
}

try {
    Write-ScheduleState -State "running"
    & $analysisScript
    $exitCode = $LASTEXITCODE
    Write-ScheduleState -State "completed" -ExitCode $exitCode
    exit $exitCode
}
catch {
    Write-ScheduleState -State "failed" -ExitCode 1 -ErrorMessage $_.Exception.Message
    exit 1
}

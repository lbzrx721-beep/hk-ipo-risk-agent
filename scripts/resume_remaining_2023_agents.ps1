param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = "D:\AnacondaDate\envs\hk-ipo-risk\python.exe"
$experimentName = "agent_remaining_train_20260809T092506"
$logDirectory = Join-Path $projectRoot "artifacts\logs"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project Python environment not found: $pythonExe"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

$arguments = @(
    "scripts\run_unattended_agent_experiment.py",
    "--project-root", $projectRoot,
    "--split", "train",
    "--limit", "19",
    "--experiment-name", $experimentName,
    "--company-concurrency", "2"
)
if ($DryRun) {
    $arguments += "--dry-run"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "resume_remaining_2023_agents_$timestamp.log"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting remaining 2023 Agent analyses." |
    Tee-Object -FilePath $logPath
& $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
$analysisExitCode = $LASTEXITCODE

if (-not $DryRun) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Rebuilding consolidated training data." |
        Tee-Object -FilePath $logPath -Append
    & $pythonExe "scripts\build_agent_augmented_dataset.py" `
        "--project-root" $projectRoot 2>&1 |
        Tee-Object -FilePath $logPath -Append
    $buildExitCode = $LASTEXITCODE
    if ($buildExitCode -ne 0) {
        exit $buildExitCode
    }
}

exit $analysisExitCode

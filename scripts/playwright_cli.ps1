param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PlaywrightArgs
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$cachePath = Join-Path $projectRoot "tmp\npm-cache-playwright"

New-Item -ItemType Directory -Path $cachePath -Force | Out-Null

# Do not use the machine-wide npm cache: its current ACL can make npx hang on
# Windows. The CLI package is resolved from this project-local cache instead.
$env:npm_config_cache = $cachePath
$env:npm_config_registry = "https://registry.npmjs.org/"

& npx.cmd --offline --yes --package "@playwright/cli" playwright-cli @PlaywrightArgs
exit $LASTEXITCODE

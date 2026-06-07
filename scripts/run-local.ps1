[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$RebuildFrontend,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Run scripts/setup-local.ps1 first."
}

if ($RebuildFrontend -or -not (Test-Path "frontend/dist/index.html")) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm is not installed. Install Node.js 22 LTS or newer."
    }
    Push-Location "frontend"
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    } finally {
        Pop-Location
    }
}

uv run python scripts/verify_local_setup.py --require-frontend
if ($LASTEXITCODE -ne 0) {
    throw "Local setup verification failed. Run scripts/setup-local.ps1."
}

$uvicornArgs = @(
    "run", "uvicorn", "src.app.main:app",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
if ($Reload) {
    $uvicornArgs += "--reload"
}

Write-Host "Starting SkyWatch at http://127.0.0.1:$Port" -ForegroundColor Green
uv @uvicornArgs

[CmdletBinding()]
param(
    [switch]$SkipDvcPull,
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Require-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing command '$Name'. $InstallHint"
    }
}

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Description)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }
    foreach ($rawLine in Get-Content $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

Require-Command "uv" "Install it from https://docs.astral.sh/uv/getting-started/installation/."

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add DagsHub and HuggingFace credentials before continuing." -ForegroundColor Yellow
    if (-not $SkipDvcPull) {
        throw "Edit .env, add DagsHub credentials, and run this script again."
    }
}
Import-DotEnv ".env"

Write-Host "Installing locked Python dependencies..." -ForegroundColor Cyan
Invoke-Checked { uv sync --python 3.11 --locked } "Python dependency installation"

if (-not $SkipDvcPull) {
    $hasDagsHubCredentials = $env:DAGSHUB_ACCESS_KEY_ID -and $env:DAGSHUB_SECRET_ACCESS_KEY
    if ($hasDagsHubCredentials) {
        $endpoint = if ($env:DAGSHUB_S3_ENDPOINT_URL) {
            $env:DAGSHUB_S3_ENDPOINT_URL
        } else {
            "https://dagshub.com/HarisBeg26/OpenSky-IIS_Projekt.s3"
        }
        Invoke-Checked { uv run dvc remote add -d origin s3://dvc --force } "DVC remote configuration"
        Invoke-Checked { uv run dvc remote modify origin endpointurl $endpoint } "DVC endpoint configuration"
        Invoke-Checked { uv run dvc remote modify origin --local access_key_id $env:DAGSHUB_ACCESS_KEY_ID } "DVC access key configuration"
        Invoke-Checked { uv run dvc remote modify origin --local secret_access_key $env:DAGSHUB_SECRET_ACCESS_KEY } "DVC secret key configuration"
    } elseif (-not (Test-Path ".dvc/config.local")) {
        throw "DagsHub credentials are required for the first DVC pull. Fill DAGSHUB_ACCESS_KEY_ID and DAGSHUB_SECRET_ACCESS_KEY in .env."
    }

    Write-Host "Pulling versioned data, models, and reports from DVC..." -ForegroundColor Cyan
    Invoke-Checked { uv run dvc pull --allow-missing --force } "DVC pull"
}

if (-not $SkipFrontendBuild) {
    Require-Command "npm" "Install Node.js 22 LTS or newer from https://nodejs.org/."
    Push-Location "frontend"
    try {
        Write-Host "Installing and building the React frontend..." -ForegroundColor Cyan
        Invoke-Checked { npm ci } "Frontend dependency installation"
        Invoke-Checked { npm run build } "Frontend production build"
    } finally {
        Pop-Location
    }
}

$verifyArgs = @("run", "python", "scripts/verify_local_setup.py")
if (-not $SkipFrontendBuild) {
    $verifyArgs += "--require-frontend"
}
Invoke-Checked { uv @verifyArgs } "Local setup verification"

Write-Host ""
Write-Host "Setup complete. Start the application with:" -ForegroundColor Green
Write-Host "  .\scripts\run-local.ps1"
Write-Host "Then open http://127.0.0.1:8000"

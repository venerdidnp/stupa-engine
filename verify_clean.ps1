# verify_clean.ps1 -- verifies clean repository clone and setup
[CmdletBinding()]
param(
    [string]$Source = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "=== VERIFIKASI CLEAN CLONE STUPA-ENGINE ===" -ForegroundColor Cyan
Write-Host "Source: $Source"

$tempRoot = [System.IO.Path]::GetTempPath()
$randName = "stupa_clean_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
$cleanDir = Join-Path $tempRoot $randName

try {
    Write-Host "`n[1/6] Cloning repository to $cleanDir..." -ForegroundColor Yellow
    git clone $Source $cleanDir
    if ($LASTEXITCODE -ne 0) { throw "Git clone failed with code $LASTEXITCODE" }

    Push-Location $cleanDir

    Write-Host "`n[2/6] Creating isolated virtualenv..." -ForegroundColor Yellow
    python -m venv .venv
    $venvPython = Join-Path $cleanDir ".venv\Scripts\python.exe"
    $venvPip = Join-Path $cleanDir ".venv\Scripts\pip.exe"

    Write-Host "`n[3/6] Installing requirements..." -ForegroundColor Yellow
    & $venvPip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Pip install failed" }

    Write-Host "`n[4/6] Running make_model.py..." -ForegroundColor Yellow
    & $venvPython make_model.py
    if ($LASTEXITCODE -ne 0) { throw "make_model.py failed" }

    Write-Host "`n[5/6] Running tests.py..." -ForegroundColor Yellow
    $testOutput = & $venvPython tests.py
    $testOutput | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "tests.py failed" }

    Write-Host "`n[6/6] Testing web server (npx serve web)..." -ForegroundColor Yellow
    $port = 39821
    $serveProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c npx -y serve web -p $port" -PassThru -NoNewWindow
    Start-Sleep -Seconds 4

    try {
        $uri = "http://localhost:$port/index.html"
        $resp = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 10
        Write-Host "HTTP Response Code: $($resp.StatusCode)" -ForegroundColor Green
        if ($resp.StatusCode -ne 200) {
            throw "Web server returned status $($resp.StatusCode), expected 200"
        }
        Write-Host "Web server check PASSED (HTTP 200 OK)" -ForegroundColor Green
    } finally {
        Write-Host "Stopping web server process (PID $($serveProcess.Id))..."
        taskkill /PID $serveProcess.Id /F /T 2>&1 | Out-Null
    }

    $sw.Stop()
    Write-Host "`n=== CLEAN CLONE VERIFICATION SUCCESSFUL ($([math]::Round($sw.Elapsed.TotalSeconds, 2))s) ===" -ForegroundColor Green
    $cleanSuccess = $true
} catch {
    Write-Host "`nERROR during clean verification: $_" -ForegroundColor Red
    $cleanSuccess = $false
} finally {
    Pop-Location
    if (Test-Path $cleanDir) {
        Write-Host "Cleaning up temporary directory $cleanDir..." -ForegroundColor Gray
        Start-Sleep -Seconds 1
        Remove-Item -Path $cleanDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $cleanSuccess) {
    exit 1
}

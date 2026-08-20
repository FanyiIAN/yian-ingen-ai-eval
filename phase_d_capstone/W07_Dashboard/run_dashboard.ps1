Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$dashboardDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $dashboardDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv $venvDir
}

& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $dashboardDir "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard dependency installation failed with exit code $LASTEXITCODE."
}

$streamlitExitCode = 1
Push-Location -LiteralPath $dashboardDir
try {
    & $venvPython -m streamlit run (Join-Path $dashboardDir "app.py") @args
    $streamlitExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $streamlitExitCode

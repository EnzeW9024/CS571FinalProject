# PowerShell script to find Python installation
Write-Host "Searching for Python installations..." -ForegroundColor Green

# Check common Python locations
$pythonPaths = @(
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:ProgramFiles\Python*",
    "$env:ProgramFiles(x86)\Python*",
    "$env:USERPROFILE\AppData\Local\Programs\Python",
    "C:\Python*",
    "C:\Program Files\Python*"
)

$found = $false

foreach ($path in $pythonPaths) {
    $dirs = Get-ChildItem -Path $path -ErrorAction SilentlyContinue
    foreach ($dir in $dirs) {
        $pythonExe = Join-Path $dir.FullName "python.exe"
        if (Test-Path $pythonExe) {
            Write-Host "Found Python at: $pythonExe" -ForegroundColor Yellow
            $version = & $pythonExe --version 2>&1
            Write-Host "  Version: $version" -ForegroundColor Cyan
            $found = $true
        }
    }
}

# Check if python is in PATH
$pythonInPath = Get-Command python -ErrorAction SilentlyContinue
if ($pythonInPath) {
    Write-Host "Python found in PATH: $($pythonInPath.Source)" -ForegroundColor Green
    & python --version
    $found = $true
}

if (-not $found) {
    Write-Host "No Python installation found." -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
}


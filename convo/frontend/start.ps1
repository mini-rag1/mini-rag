# PowerShell script to start the AI Travel Assistant
Write-Host "🌍 Starting AI Travel Assistant..." -ForegroundColor Green
Write-Host ""

# Ensure we're in the frontend directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "📁 Working directory: $scriptDir" -ForegroundColor Cyan

# Check if streamlit is installed
try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "📦 Installing required packages..." -ForegroundColor Yellow
        pip install -r requirements_frontend.txt
    }
} catch {
    Write-Host "❌ Python not found. Please ensure Python is installed and in PATH." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting the Travel Assistant frontend..." -ForegroundColor Cyan
Write-Host "🌐 Open your browser and go to: http://localhost:8503" -ForegroundColor Blue
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Start Streamlit with full path to avoid caching issues
$appPath = Join-Path $scriptDir "app.py"
python -m streamlit run $appPath --server.port 8503

# Activate Mini-RAG Conda Environment
Write-Host "Activating Mini-RAG environment with Python 3.11.7..." -ForegroundColor Green

# Add conda to PATH if not already there
if (-not ($env:PATH -like "*miniconda3*")) {
    $env:PATH = "$env:USERPROFILE\miniconda3\Scripts;$env:USERPROFILE\miniconda3\condabin;" + $env:PATH
}

# Initialize conda for this session
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell --no-user 2>$null

# Activate the environment
try {
    conda activate mini-rag
    Write-Host "Environment activated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Current Python version:" -ForegroundColor Yellow
    python --version
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  python src\main.py     - Run your main application"
    Write-Host "  pip install <package> - Install additional packages"
    Write-Host "  conda deactivate      - Exit the environment"
} catch {
    Write-Host "Using conda run instead (PowerShell session needs restart for full conda support)" -ForegroundColor Yellow
    Write-Host "Use: conda run -n mini-rag python src\main.py" -ForegroundColor Cyan
}

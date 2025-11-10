#!/usr/bin/env pwsh
# Start all MCP services and Streamlit UI
# Usage: .\start_services.ps1

Write-Host "🚀 Starting RAG+MCP+Agent Framework..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found. Please run:" -ForegroundColor Red
    Write-Host "   python -m venv .venv" -ForegroundColor Yellow
    Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Get current directory
$rootDir = Get-Location

# Start plot-service
Write-Host "🎨 Starting plot-service on port 8000..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$rootDir\tools\plot-service'; & '$rootDir\.venv\Scripts\python.exe' -m uvicorn main:app --host 0.0.0.0 --port 8000"

# Start calculator service
Write-Host "🔢 Starting calculator on port 8001..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$rootDir\tools\calculator'; & '$rootDir\.venv\Scripts\python.exe' -m uvicorn main:app --host 0.0.0.0 --port 8001"

# Start pdf-parser service
Write-Host "📄 Starting pdf-parser on port 8002..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$rootDir\tools\pdf-parser'; & '$rootDir\.venv\Scripts\python.exe' -m uvicorn main:app --host 0.0.0.0 --port 8002"

# Wait for services to start
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start Streamlit UI
Write-Host "🌐 Starting Streamlit UI on port 9000..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$rootDir\frontend'; & '$rootDir\.venv\Scripts\streamlit.exe' run app.py --server.port 9000"

Write-Host ""
Write-Host "✅ All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access the UI at: http://localhost:9000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C in each terminal window to stop services." -ForegroundColor Gray

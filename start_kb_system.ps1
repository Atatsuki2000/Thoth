# Start Complete RAG Agent System with Knowledge Base
# Runs: MCP tools + KB API + Enhanced Frontend

Write-Host "🚀 Starting RAG Agent System with Knowledge Base..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Green
& ".venv\Scripts\Activate.ps1"

# Check if required packages are installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor Green
$packages = @("fastapi", "uvicorn", "streamlit", "chromadb", "langchain")
$missing = @()

foreach ($package in $packages) {
    $installed = python -c "import $package" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $package
    }
}

if ($missing.Count -gt 0) {
    Write-Host "❌ Missing packages: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Create kb_storage directory if it doesn't exist
if (-not (Test-Path "kb_storage")) {
    New-Item -ItemType Directory -Path "kb_storage" | Out-Null
    Write-Host "✅ Created kb_storage directory" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎯 Starting services..." -ForegroundColor Cyan
Write-Host ""

# Array to hold all started processes
$processes = @()

# Function to start a service in a new window
function Start-Service {
    param($Name, $Command, $WorkingDir, $Color)
    
    Write-Host "Starting $Name..." -ForegroundColor $Color
    
    # Use direct Python path to avoid activation issues
    $pythonPath = "$WorkingDir\.venv\Scripts\python.exe"
    
    $process = Start-Process powershell -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-NoExit",
        "-Command",
        "& {
            `$host.UI.RawUI.WindowTitle = '$Name'
            Write-Host '═══════════════════════════════════════' -ForegroundColor $Color
            Write-Host '  $Name' -ForegroundColor $Color
            Write-Host '═══════════════════════════════════════' -ForegroundColor $Color
            Write-Host ''
            cd '$WorkingDir'
            $Command
        }"
    ) -PassThru -WindowStyle Normal
    
    return $process
}

# Start MCP Tools
$plotProcess = Start-Service `
    -Name "Plot Service (Port 8000)" `
    -Command "cd tools\plot-service; ..\..`\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000" `
    -WorkingDir $PWD `
    -Color "Magenta"
$processes += $plotProcess

Start-Sleep -Seconds 2

$calcProcess = Start-Service `
    -Name "Calculator Service (Port 8001)" `
    -Command "cd tools\calculator; ..\..`\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001" `
    -WorkingDir $PWD `
    -Color "Blue"
$processes += $calcProcess

Start-Sleep -Seconds 2

$pdfProcess = Start-Service `
    -Name "PDF Parser Service (Port 8002)" `
    -Command "cd tools\pdf-parser; ..\..`\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8002" `
    -WorkingDir $PWD `
    -Color "Yellow"
$processes += $pdfProcess

Start-Sleep -Seconds 2

# Start Web Search Service
$webSearchProcess = Start-Service `
    -Name "Web Search Service (Port 8003)" `
    -Command "cd tools\web-search; ..\..`\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8003" `
    -WorkingDir $PWD `
    -Color "DarkCyan"
$processes += $webSearchProcess

Start-Sleep -Seconds 2

# Start File Operations Service
$fileOpsProcess = Start-Service `
    -Name "File Operations Service (Port 8004)" `
    -Command "`$env:WORKSPACE_DIR='$PWD'; cd tools\file-ops; ..\..`\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8004" `
    -WorkingDir $PWD `
    -Color "DarkYellow"
$processes += $fileOpsProcess

Start-Sleep -Seconds 2

# Start Knowledge Base API
$kbProcess = Start-Service `
    -Name "Knowledge Base API (Port 8100)" `
    -Command ".venv\Scripts\python.exe kb_api.py" `
    -WorkingDir $PWD `
    -Color "Green"
$processes += $kbProcess

Start-Sleep -Seconds 3

# Start Enhanced Streamlit Frontend
$frontendProcess = Start-Service `
    -Name "Enhanced Frontend (Port 9001)" `
    -Command "cd frontend; ..`\.venv\Scripts\streamlit.exe run app_kb.py --server.port 9001 --server.headless true" `
    -WorkingDir $PWD `
    -Color "Cyan"
$processes += $frontendProcess

# Wait for services to start
Write-Host ""
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Display service status
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✅ ALL SERVICES STARTED" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor White
Write-Host "  🎨 Plot Service:      http://localhost:8000" -ForegroundColor Magenta
Write-Host "  🔢 Calculator:        http://localhost:8001" -ForegroundColor Blue
Write-Host "  📄 PDF Parser:        http://localhost:8002" -ForegroundColor Yellow
Write-Host "  � Web Search:        http://localhost:8003" -ForegroundColor DarkCyan
Write-Host "  📁 File Operations:   http://localhost:8004" -ForegroundColor DarkYellow
Write-Host "  �📚 Knowledge Base:    http://localhost:8100" -ForegroundColor Green
Write-Host "  🌐 Web Interface:     http://localhost:9001" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Open browser to frontend
Write-Host "🌐 Opening web interface in browser..." -ForegroundColor Green
Start-Sleep -Seconds 3
Start-Process "http://localhost:9001"

Write-Host ""
Write-Host "💡 Quick Start Guide:" -ForegroundColor Yellow
Write-Host "  1. Go to 'Upload Documents' tab" -ForegroundColor White
Write-Host "  2. Create a new collection (e.g., 'my_docs')" -ForegroundColor White
Write-Host "  3. Upload PDF, TXT, DOCX, or MD files" -ForegroundColor White
Write-Host "  4. Switch to 'Chat' tab to query your documents" -ForegroundColor White
Write-Host "  5. Initialize the agent in sidebar to use tools" -ForegroundColor White
Write-Host ""
Write-Host "📖 Full documentation: docs/kb-upload-guide.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to view cleanup options..." -ForegroundColor Gray
Write-Host ""

# Wait and handle cleanup
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host ""
    Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
    
    foreach ($process in $processes) {
        if ($process -and !$process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    
    Write-Host "✅ All services stopped" -ForegroundColor Green
}

#!/bin/bash
# Start all MCP services and Streamlit UI
# Usage: ./start_services.sh

echo "🚀 Starting RAG+MCP+Agent Framework..."
echo ""

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Start plot-service in background
echo "🎨 Starting plot-service on port 8000..."
cd tools/plot-service
uvicorn main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
PLOT_PID=$!
cd ../..

# Start calculator service in background
echo "🔢 Starting calculator on port 8001..."
cd tools/calculator
uvicorn main:app --host 0.0.0.0 --port 8001 > /dev/null 2>&1 &
CALC_PID=$!
cd ../..

# Start pdf-parser service in background
echo "📄 Starting pdf-parser on port 8002..."
cd tools/pdf-parser
uvicorn main:app --host 0.0.0.0 --port 8002 > /dev/null 2>&1 &
PDF_PID=$!
cd ../..

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 3

# Start Streamlit UI
echo "🌐 Starting Streamlit UI on port 9000..."
cd frontend
streamlit run app.py --server.port 9000 &
STREAMLIT_PID=$!
cd ..

echo ""
echo "✅ All services started!"
echo ""
echo "PIDs:"
echo "  plot-service: $PLOT_PID"
echo "  calculator: $CALC_PID"
echo "  pdf-parser: $PDF_PID"
echo "  streamlit: $STREAMLIT_PID"
echo ""
echo "Access the UI at: http://localhost:9000"
echo ""
echo "To stop all services, run:"
echo "  kill $PLOT_PID $CALC_PID $PDF_PID $STREAMLIT_PID"

# Trap Ctrl+C to clean up
trap "echo ''; echo 'Stopping all services...'; kill $PLOT_PID $CALC_PID $PDF_PID $STREAMLIT_PID; exit" INT

# Wait for user to stop
wait

# Retrieval-Aware Tool-Using Agent Framework with MCP Integration

A production-ready system combining **Retrieval-Augmented Generation (RAG)**, **Model Context Protocol (MCP)** tools, and **autonomous agent orchestration** to create an intelligent assistant that retrieves context and invokes specialized tools.

## 🎯 Features

- **RAG System**: Retrieves relevant documents using HuggingFace embeddings and Chroma vector store
- **MCP Tools** (deployed to Google Cloud Run):
  - 🎨 `plot-service`: Mathematical functions (sin, cos, tan, etc.) + categorical visualizations
  - 🔢 `calculator`: Safe mathematical expression evaluation
  - 📄 `pdf-parser`: Extract text from PDF documents
- **Dual-Mode Agent Orchestration**:
  - ⚡ **Keyword-based** (517ms avg): Fast, deterministic, zero cost
  - 🧠 **Local LLM with TinyLlama** (1.9s avg): Optimized inference, no API needed
  - 🤖 **OpenAI GPT-3.5** (optional): Highest accuracy, requires paid API
- **Interactive UI**: Streamlit frontend for real-time interaction
- **Production-Ready**: Deployed to Cloud Run, 100% free tier compatible
- **Optimized Performance**: 13.7x LLM speedup (26s → 1.9s)
- **Error Handling**: Robust retry logic and graceful error recovery
- **CI/CD**: Automated testing with GitHub Actions

## 📋 Architecture

```
[User UI (Streamlit)]
		  ↓
[Agent Orchestrator]
	↓          ↓           ↓
[RAG]    [Reasoner]  [MCP Tools]
	↓                      ↑
[Chroma DB]      [FastAPI Services]
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Atatsuki2000/Retrieval-Aware-Tool-Using-Agent-Framework-with-MCP-Integration.git
cd Retrieval-Aware-Tool-Using-Agent-Framework-with-MCP-Integration

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**Option 1: Docker Compose (Easiest)**
```bash
# Start all services with Docker
docker-compose up -d --build

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f
```

Services available at:
- plot-service: http://localhost:8000
- calculator: http://localhost:8001
- pdf-parser: http://localhost:8002

See [DOCKER.md](DOCKER.md) for full Docker guide.

**Option 2: PowerShell/Bash Scripts**
```bash
# Windows
.\start_services.ps1

# Linux/Mac
chmod +x start_services.sh
./start_services.sh
```

This will automatically start all MCP services and the Streamlit UI.

**Option 3: Manual Startup**

1. **Start MCP Tool Services** (in separate terminals):

```bash
# Terminal 1: Plot Service
cd tools/plot-service
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Calculator Service
cd tools/calculator
uvicorn main:app --host 0.0.0.0 --port 8001

# Terminal 3: PDF Parser Service
cd tools/pdf-parser
uvicorn main:app --host 0.0.0.0 --port 8002
```

2. **Launch Streamlit UI**:

```bash
cd frontend
streamlit run app.py --server.port 9000
```

3. **Configure Endpoints** in the Streamlit sidebar:
	- plot-service URL: `http://127.0.0.1:8000/mcp/plot`
	- calculator URL: `http://127.0.0.1:8001/mcp/calculate`
	- pdf-parser URL: `http://127.0.0.1:8002/mcp/parse`

4. **Try Example Queries**:
	- "Plot sin(x) from 0 to 10" (mathematical visualization)
	- "Calculate 25 * 17 + 89" (calculator tool)
	- "Show me a bar chart" (categorical visualization)
	- "What is machine learning?" (RAG retrieval only)

## 🧪 Testing

### Unit & Integration Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agent --cov-report=html
```

### Performance Benchmarks

```bash
# Run all benchmarks (retrieval + agent modes comparison)
python benchmark.py --mode all --save

# Test only retrieval performance
python benchmark.py --mode retrieval

# Compare agent modes (keyword vs local LLM vs OpenAI)
python benchmark.py --mode comparison --save
```

**Benchmark Metrics:**
- 📊 **Retrieval Precision@k**: Accuracy of document retrieval
- ⏱️ **Latency Breakdown**: Retrieval, LLM, tool invocation timings
- 🎯 **Tool Selection Accuracy**: Correctness of agent's tool choice
- ✅ **Tool Success Rate**: Percentage of successful MCP calls

See [Benchmarking Guide](docs/benchmarking.md) for detailed usage and interpretation.

## 📁 Project Structure

```
.
├── agent/              # Agent orchestration & retriever
│   ├── agent.py        # Main agent logic
│   ├── retriever.py    # RAG retrieval system
│   └── test_corpus.txt # Sample corpus
├── tools/              # MCP tool services
│   ├── plot-service/   # Visualization tool
│   ├── calculator/     # Math evaluation tool
│   └── pdf-parser/     # PDF text extraction
├── frontend/           # Streamlit UI
│   └── app.py
├── tests/              # Integration tests
├── .github/workflows/  # CI/CD pipelines
└── requirements.txt    # Python dependencies
```

## 📚 Documentation

- **[Architecture Guide](docs/architecture.md)**: System design and data flow diagrams
- **[Deployment Guide](docs/deployment.md)**: Local and Cloud Run deployment instructions
- **[Testing Guide](docs/testing.md)**: Test execution and coverage reporting
- **[Benchmarking Guide](docs/benchmarking.md)**: 📊 Performance evaluation and metrics
- **[Usage Examples](docs/usage.md)**: API examples and common patterns
- **[Local LLM Setup](docs/local-llm-setup.md)**: 🆓 Use free HuggingFace models for tool selection (no API costs!)
- **[LLM Tool Selection](docs/llm-tool-selection.md)**: Guide to using OpenAI GPT-3.5 (paid API)

## �🛠️ Development

### Environment Variables

Set these to avoid manual configuration:

```bash
export PLOT_SERVICE_URL=http://127.0.0.1:8000/mcp/plot
export CALCULATOR_URL=http://127.0.0.1:8001/mcp/calculate
export PDF_PARSER_URL=http://127.0.0.1:8002/mcp/parse
```

### Adding New MCP Tools

1. Create a new directory under `tools/`
2. Implement FastAPI endpoint with MCP schema
3. Update agent keyword detection in `agent/agent.py`
4. Add endpoint to Streamlit sidebar configuration

## 📊 Metrics & Performance

### Benchmarked Performance (Latest Results - 2025)

| Metric | Keyword Mode | Local LLM Mode | Target | Status |
|--------|-------------|----------------|--------|--------|
| **Tool Selection Accuracy** | 100% | 100% | >85% | ✅ Excellent |
| **Tool Success Rate** | 100% | 100% | >90% | ✅ Excellent |
| **Avg Retrieval Latency** | 66ms | 70ms | <100ms | ✅ Excellent |
| **Avg End-to-End Latency** | **517ms** | **1.9s** | <2s | ✅ Excellent |
| **Cost per Query** | $0 | $0 | Free | ✅ Zero Cost |

**Performance Optimization:** TinyLlama optimized from 26s → 1.9s (13.7x speedup) through:
- Simplified prompt design (JSON → direct keyword format)
- Reduced token generation (max_new_tokens: 50 → 10)
- Greedy decoding for faster inference
- Efficient keyword extraction from generated text

### Mode Comparison

| Mode | Accuracy | Latency | Cost | Best For |
|------|----------|---------|------|----------|
| **Keyword** ⭐ | 100% | 517ms | $0 | Production, ultra-low latency |
| **Local LLM (TinyLlama)** 🚀 | 100% | 1.9s | $0 | Zero-cost inference, portfolio demos |
| **OpenAI GPT-3.5** | 95-98% | ~800ms | ~$0.0004 | Highest flexibility |

Run `python benchmark.py --mode comparison --save` for detailed analysis.

## ☁️ Cloud Deployment

### Google Cloud Run (Production)

All 3 MCP tools are deployed to Google Cloud Run (us-central1):
- **plot-service**: https://plot-service-347876502362.us-central1.run.app
- **calculator**: https://calculator-h7whjphxza-uc.a.run.app
- **pdf-parser**: https://pdf-parser-h7whjphxza-uc.a.run.app

**Deployment Cost:** $0/month (100% within free tier)
- 2M requests/month free
- 360k GB-seconds compute free
- 0.5GB container storage free

**To use Cloud Run endpoints in Streamlit:**
```bash
# Set environment variables
export PLOT_SERVICE_URL=https://plot-service-347876502362.us-central1.run.app/mcp/plot
export CALCULATOR_URL=https://calculator-h7whjphxza-uc.a.run.app/mcp/calculate
export PDF_PARSER_URL=https://pdf-parser-h7whjphxza-uc.a.run.app/mcp/parse

# Or configure directly in Streamlit sidebar
```

**Deploy your own:**
```bash
cd tools/plot-service
gcloud run deploy plot-service --source . --region us-central1 --allow-unauthenticated
```

See [Deployment Guide](docs/deployment.md) for detailed instructions.

## 🔒 Security

- No hardcoded credentials (environment variables only)
- Safe expression evaluation using `numexpr`
- Input validation on all MCP endpoints
- HTTPS recommended for production deployments

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 📜 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)

## 📧 Contact

- **GitHub Issues**: For bug reports and feature requests
- **Repository**: https://github.com/Atatsuki2000/Retrieval-Aware-Tool-Using-Agent-Framework-with-MCP-Integration

## ⭐ Show Your Support

If you find this project helpful, please consider giving it a star on GitHub!

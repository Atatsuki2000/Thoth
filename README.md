# Retrieval-Aware Tool-Using Agent Framework with MCP Integration

A production-ready system combining **Retrieval-Augmented Generation (RAG)**, **Model Context Protocol (MCP)** tools, and **autonomous agent orchestration** to create an intelligent assistant that retrieves context and invokes specialized tools.

## 🎯 Features

- **RAG System**: Retrieves relevant documents using HuggingFace embeddings and Chroma vector store
- **MCP Tools**: 
  - 🎨 `plot-service`: Generate visualizations with matplotlib
  - 🔢 `calculator`: Safe mathematical expression evaluation
  - 📄 `pdf-parser`: Extract text from PDF documents
- **Agent Orchestration**: Triple-mode tool selection
  - 📝 **Keyword-based** (default): Fast, deterministic, zero cost
  - 🆓 **Local LLM** (recommended): Free HuggingFace models, no API needed
  - 🤖 **OpenAI GPT-3.5** (optional): Highest accuracy, requires paid API
- **Interactive UI**: Streamlit frontend for real-time interaction
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

**Option 1: One-Command Startup (Recommended)**
```bash
# Windows
.\start_services.ps1

# Linux/Mac
chmod +x start_services.sh
./start_services.sh
```

This will automatically start all MCP services and the Streamlit UI.

**Option 2: Manual Startup**

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
	- "What tool can I use to plot data?"
	- "Calculate 5 + 3 * 2"
	- "Show me a histogram"

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agent --cov-report=html
```

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

- **Retrieval Precision**: Evaluated using test corpus
- **End-to-End Latency**: Measured from query to response
- **Tool Success Rate**: Tracked via MCP response status

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

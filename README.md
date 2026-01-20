# 📚 Thoth - A Powerful Multi-Tool RAG Agent Framework

> **Ancient wisdom meets modern AI: Intelligent assistant combining knowledge retrieval, LLM reasoning, and specialized MCP tools**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Named after the Egyptian god of knowledge, writing, and wisdom, **Thoth** is your AI companion that remembers, reasons, and acts.

## ✨ Highlights

- 📚 **Smart Knowledge Base**: Upload PDFs, DOCX, TXT, MD - AI instantly learns from your documents
- 🛠️ **5 Powerful Tools**: Calculator, Web Search, File Ops, Plotting, PDF Parser
- 🤖 **Flexible AI Models**: Free local LLM (TinyLlama) or premium OpenAI GPT-3.5/4
- 🌐 **Web Interface**: Beautiful Streamlit UI with real-time chat
- 🔌 **n8n Integration**: Pre-built workflows for automation
- ⚡ **Production Ready**: Error handling, retry logic, comprehensive logging

## 🎯 What Can It Do?

**Ask natural language questions:**
- ❓ "What is machine learning?" → Searches your knowledge base + web
- 🧮 "Calculate 25 * 17" → Returns `425`
- 📊 "Plot a sine wave" → Generates beautiful matplotlib chart
- 📄 "Read file README.md" → Shows file contents
- 🌍 "What's the latest on Python 3.12?" → Web search + AI summary

## 📋 Architecture

```
User Interface (Streamlit) → http://localhost:9001
         ↓
   Agent Orchestrator
    ↙    ↓    ↘
  RAG  Tools  Web Search
   ↓     ↓      ↓
ChromaDB MCP   DuckDuckGo
         Services
```

**Components:**
- **KB API** (port 8100): Document upload, vector storage, retrieval
- **Agent**: Intelligent tool selection + execution
- **MCP Tools**: Calculator (8001), Plot (8000), PDF (8002), Web Search (8003), File Ops (8004)
- **Frontend**: Streamlit chat interface (9001)

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone & setup
git clone https://github.com/Atatsuki2000/Thoth.git
cd Retrieval-Aware-Tool-Using-Agent-Framework-with-MCP-Integration

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch (One Command!)

```powershell
# Windows
.\start_kb_system.ps1

# The script starts:
# ✅ KB API (port 8100)
# ✅ 5 MCP tool services (8000-8004)
# ✅ Web UI (port 9001) - Auto-opens in browser
```

### 3. Use It!

1. **Upload Documents** (Upload tab)
   - Drag & drop PDF/DOCX/TXT/MD files
   - Select collection (or create new)
   - Click "Upload to Knowledge Base"

2. **Initialize Agent** (Sidebar)
   - Choose LLM: `local` (free TinyLlama) or `openai`
   - Click "Initialize Agent"
   - See ✅ "Agent is ready!"

3. **Ask Questions** (Chat tab)
   - ✅ Enable MCP Tools
   - Type: "What is machine learning?" or "Calculate 25*17" or "Plot sine wave"
   - Get AI-powered answers with tool execution!

## 🔌 n8n Automation

Automate your knowledge base with pre-built n8n workflows:

**Setup:**
```bash
npm install -g n8n
n8n start  # Opens http://localhost:5678
```

**Available Workflows** (in `n8n-nodes/` folder):

1. **`rag-query-workflow.json`** - Query knowledge base programmatically
   - Manual trigger → HTTP Request to KB API → Format response
   - Use for batch queries or API integration

2. **`kb-upload-workflow.json`** - Upload documents automatically
   - Read files → Upload to KB → Process response
   - Supports PDF, TXT, MD, DOCX

3. **`automated-rag-workflow.json`** - Scheduled knowledge reports
   - Cron trigger (daily 9am) → Batch queries → Generate report
   - Perfect for daily summaries or monitoring

**How to use:**
1. In n8n, click "Import from File"
2. Select a workflow JSON file
3. Update parameters (collection name, query, schedule)
4. Click "Execute Workflow" to test
5. Set "Active: ON" for scheduled workflows

**Note:** Ensure KB API is running at `http://localhost:8100` before using workflows.

## 🧪 Testing

### Quick Health Check

```bash
# Ensure all 7 services are responding
curl http://localhost:8100/health  # KB API
curl http://localhost:8000/mcp/health  # Plot
curl http://localhost:8001/mcp/health  # Calculator
curl http://localhost:8002/mcp/health  # PDF Parser
curl http://localhost:8003/mcp/health  # Web Search
curl http://localhost:8004/mcp/health  # File Ops
# UI at http://localhost:9001
```

### Try Example Queries

1. **Calculator**: "What is 25 times 17 plus 100?"
2. **Plotting**: "Plot a sine wave from 0 to 10"
3. **Web Search**: "What is Christmas?"
4. **File Operations**: "Read the README.md file"
5. **Knowledge Base**: Upload a document first, then ask about it!

## 📚 Getting Help

**Documentation:** Check the `demo_kb/` folder for sample documents and usage examples. Upload these files to see Thoth in action!

**OpenAI Setup (Optional):** To use GPT-3.5/4 instead of local LLM, set environment variable: `OPENAI_API_KEY=sk-your-key`

**Local LLM (Free):** Thoth uses TinyLlama by default. It downloads automatically on first run (~500MB). For better accuracy, use OpenAI.

## 🤝 Contributing & License

Contributions are welcome! Feel free to:
- Report bugs or request features via [GitHub Issues](https://github.com/Atatsuki2000/Thoth/issues)
- Submit pull requests with improvements
- Share your use cases in [Discussions](https://github.com/Atatsuki2000/Thoth/discussions)

This project is licensed under **MIT License** - free to use, modify, and distribute.

---

**Built with ❤️ using FastAPI, Streamlit, ChromaDB, and n8n**

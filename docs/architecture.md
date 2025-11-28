# System Architecture

## High-Level Overview

```
┌────────────────────────────────────────────────────────────────┐
│                       User Interface Layer                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              Streamlit Frontend (Port 9000)             │   │
│  │  - Query Input                                          │   │
│  │  - Endpoint Configuration                               │   │
│  │  - Result Visualization                                 │   │
│  └────────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                      Agent Orchestration Layer                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                   SimpleAgent                           │   │
│  │  - plan_and_execute(query)                             │   │
│  │  - Rule-based tool selection                           │   │
│  │  - Retry logic with exponential backoff                │   │
│  └────────────────────────────────────────────────────────┘   │
└───────┬────────────────────────────────────┬───────────────────┘
        │                                    │
        ▼                                    ▼
┌───────────────────┐              ┌──────────────────────────┐
│  Retrieval Layer  │              │   MCP Tools Layer        │
│                   │              │                          │
│ ┌───────────────┐ │              │  ┌─────────────────┐   │
│ │   Retriever   │ │              │  │  plot-service   │   │
│ │               │ │              │  │  (Port 8000)    │   │
│ │ - get_top_k() │ │              │  │  - Bar charts   │   │
│ │               │ │              │  │  - Base64 PNG   │   │
│ └───────┬───────┘ │              │  └─────────────────┘   │
│         │         │              │                          │
│         ▼         │              │  ┌─────────────────┐   │
│ ┌───────────────┐ │              │  │  calculator     │   │
│ │  Chroma DB    │ │              │  │  (Port 8001)    │   │
│ │               │ │              │  │  - numexpr eval │   │
│ │ - Vector      │ │              │  │  - Safe math    │   │
│ │   Store       │ │              │  └─────────────────┘   │
│ │               │ │              │                          │
│ │ - HuggingFace │ │              │  ┌─────────────────┐   │
│ │   Embeddings  │ │              │  │  pdf-parser     │   │
│ └───────────────┘ │              │  │  (Port 8002)    │   │
│                   │              │  │  - pypdf extract│   │
└───────────────────┘              │  │  - Base64 input │   │
                                   │  └─────────────────┘   │
                                   └──────────────────────────┘
```

## Data Flow Sequence

### 1. Query Processing Flow

```
User → Streamlit UI → Agent.plan_and_execute()
                            ↓
                    [Step 1: Retrieval]
                    retriever.get_top_k(query, k=3)
                            ↓
                    Chroma DB similarity search
                            ↓
                    Return top 3 documents (deduplicated)
                            ↓
                    [Step 2: Tool Selection]
                    Keyword detection:
                    - "calculate|compute|math" → calculator
                    - "plot|chart|histogram" → plot-service
                            ↓
                    [Step 3: Tool Invocation]
                    HTTP POST to MCP endpoint
                            ↓
                    Retry logic (3 attempts, 1s delay)
                            ↓
                    Parse MCP response
                            ↓
                    Return to UI for visualization
```

### 2. MCP Request/Response Format

#### Request Schema
```json
{
  "method": "invoke",
  "params": {
    "tool_name": "plot",
    "payload": {
      "data_reference": "..."
    }
  }
}
```

#### Response Schema
```json
{
  "status": "success",
  "result": {
    "type": "image",
    "data": "base64_encoded_string"
  }
}
```

## Agent Decision Tree

```
User Query
    ↓
Retriever.get_top_k(query, k=3)
    ↓
Extract context
    ↓
Keyword Detection?
    ├── "calculate/compute/math"
    │   ↓
    │   POST to calculator
    │   ↓
    │   Return numeric result
    │
    ├── "plot/chart/histogram"
    │   ↓
    │   POST to plot-service
    │   ↓
    │   Return base64 image
    │
    └── No keywords
        ↓
        Return retrieval context only
```

## Error Handling Flow

```
Tool Invocation
    ↓
Try HTTP POST
    ├── Success → Return result
    │
    ├── ConnectionError/Timeout
    │   ↓
    │   Retry (attempt 1/3)
    │   ↓
    │   Wait 1s × attempt
    │   ↓
    │   Retry (attempt 2/3)
    │   ↓
    │   Wait 2s
    │   ↓
    │   Retry (attempt 3/3)
    │   ↓
    │   Return error message
    │
    └── HTTPError (4xx/5xx)
        ↓
        Return error immediately
        (no retry)
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Interactive UI |
| Agent | Python | Orchestration logic |
| Retrieval | LangChain + Chroma | RAG system |
| Embeddings | HuggingFace | Free vector embeddings |
| MCP Tools | FastAPI | REST endpoints |
| Visualization | Matplotlib | Plot generation |
| Math Eval | numexpr | Safe calculator |
| PDF Parse | pypdf | Text extraction |
| Testing | pytest | Integration tests |
| CI/CD | GitHub Actions | Automated workflows |

## Deployment Architectures

### Local Development Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Developer Workstation                     │
│                                                              │
│  ┌──────────────┐                                           │
│  │  Streamlit   │  Port 9000                                │
│  │   Frontend   │                                           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐        ┌─────────────────┐              │
│  │ SimpleAgent  │◄──────►│   Chroma DB     │              │
│  │              │        │   (SQLite)      │              │
│  └──────┬───────┘        └─────────────────┘              │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │
          │ HTTPS (Public Internet)
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud Run (us-central1)                  │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  plot-service    │  │   calculator     │                │
│  │  Port 8080       │  │   Port 8080      │                │
│  │  512Mi / 1 CPU   │  │   512Mi / 1 CPU  │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  ┌──────────────────┐                                       │
│  │  pdf-parser      │                                       │
│  │  Port 8080       │                                       │
│  │  512Mi / 1 CPU   │                                       │
│  └──────────────────┘                                       │
│                                                              │
│  Service URLs:                                              │
│  - https://plot-service-xxx.us-central1.run.app/mcp/plot   │
│  - https://calculator-xxx.us-central1.run.app/mcp/calculate│
│  - https://pdf-parser-xxx.us-central1.run.app/mcp/parse    │
└─────────────────────────────────────────────────────────────┘
```

### Production Deployment (Future)

```
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud Run (us-central1)                  │
│                                                              │
│  ┌──────────────────┐                                       │
│  │  Frontend        │  (Streamlit on Cloud Run)             │
│  │  Port 8080       │                                       │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  Agent API       │◄──────►│  Vector DB       │          │
│  │  (FastAPI)       │        │  (Pinecone/      │          │
│  │  Port 8080       │        │   Weaviate)      │          │
│  └────────┬─────────┘        └──────────────────┘          │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────┐          │
│  │        MCP Tool Services                      │          │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │          │
│  │  │  plot    │ │   calc   │ │   pdf    │     │          │
│  │  └──────────┘ └──────────┘ └──────────┘     │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  Load Balancer   │        │  Cloud Armor     │          │
│  │  (HTTPS)         │        │  (WAF/DDoS)      │          │
│  └──────────────────┘        └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Agent Operation Modes

### Keyword Mode (Fast - 208ms avg)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Query: "Calculate 15 * 23 + 7"                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Retrieval: get_top_k(query, k=3)                         │
│    - Latency: 69ms                                          │
│    - Returns: calculator tool docs                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Keyword Detection: "calculate" → calculator tool         │
│    - Regex match: r'(calculate|compute|math)'               │
│    - Decision time: <1ms                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MCP Tool Call: POST /mcp/calculate                       │
│    Payload: {"expression": "15 * 23 + 7"}                   │
│    - Network latency: 139ms                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Response: {"value": 352.0, "status": "success"}          │
│    - Total time: 208ms                                      │
│    - Accuracy: 100%                                         │
└─────────────────────────────────────────────────────────────┘
```

### Local LLM Mode (Accurate - ~26s avg)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Query: "Calculate 15 * 23 + 7"                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Retrieval: get_top_k(query, k=3)                         │
│    - Latency: 85ms                                          │
│    - Returns: calculator tool docs                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM Reasoning: Phi-3.5-MoE-instruct (16.9B params)       │
│    - Prompt: "Query + Context → Plan tool call"             │
│    - Generation time: ~25s                                  │
│    - Output: "Use calculator with expression '15*23+7'"     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Parse LLM Output: Extract tool name & parameters         │
│    - Parsing time: <1ms                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. MCP Tool Call: POST /mcp/calculate                       │
│    - Network latency: 1206ms                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Response: {"value": 352.0, "status": "success"}          │
│    - Total time: ~26s                                       │
│    - Accuracy: 100%                                         │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

### Latency Breakdown

**Keyword Mode (Average: 208ms)**
```
Retrieval:       69ms  ████████████████████████████████ (33%)
Tool Selection:  <1ms  (negligible)
Network RTT:    139ms  ████████████████████████████████████████████████████████████████ (67%)
Processing:      <1ms  (negligible)
```

**Local LLM Mode (Average: 26,378ms)**
```
Retrieval:          85ms  ▏ (0.3%)
LLM Generation:  25,087ms  ████████████████████████████████████████████████████████████████████████████████████████████ (95%)
Network RTT:      1,206ms  ████▉ (4.6%)
Processing:          <1ms  (negligible)
```

### Resource Usage

**Local Components:**
- Chroma DB: 100MB disk, 50MB RAM
- Embedding Model: 90MB download, 200MB RAM
- Local LLM: 16GB RAM (Phi-3.5-MoE)
- Python Process: 500MB RAM baseline

**Cloud Run (per service):**
- Memory Limit: 512Mi
- CPU: 1 vCPU
- Cold Start: ~2s
- Warm Request: 100-200ms
- Auto-scaling: 0-10 instances
- Cost: ~$0.50/month per service (demo usage)

## Security Architecture

### Input Validation Layer

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│  Streamlit Input Sanitization       │
│  - XSS prevention                   │
│  - Length limits                    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Agent Input Validation              │
│  - Query length check                │
│  - Character encoding validation     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  MCP Tool Pydantic Validation        │
│  - Schema enforcement                │
│  - Type checking                     │
│  - Required fields                   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Safe Execution                      │
│  - Calculator: AST-only (no eval)    │
│  - PDF Parser: Type validation       │
│  - Plot: Data sanitization           │
└─────────────────────────────────────┘
```

### Security Checklist

✅ **Authentication**: Environment variables for secrets
✅ **Authorization**: Cloud Run IAM (can add API keys)
✅ **Input Validation**: Pydantic models on all endpoints
✅ **Code Injection**: AST-based calculator (no `eval()`)
✅ **HTTPS**: All Cloud Run endpoints use TLS
✅ **Secrets Management**: `.env` files (local), Secret Manager (prod)
⚠️ **Rate Limiting**: Not implemented (use Cloud Armor in prod)
⚠️ **Authentication**: Currently allow-unauthenticated (demo mode)

## Cost Analysis

### Current Deployment (Demo Mode)

**Google Cloud Run Costs:**
```
plot-service:   $0.50/month  (1000 requests/day estimate)
calculator:     $0.50/month  (1000 requests/day estimate)
pdf-parser:     $0.50/month  (1000 requests/day estimate)
─────────────────────────────
Total:          ~$1.50/month

Free Tier:
- 2M requests/month free
- 360,000 GiB-seconds/month free
- 180,000 vCPU-seconds/month free

Demo usage << Free tier limits ✓
```

**Local Development:**
- Electricity: Negligible
- Embeddings: Free (HuggingFace model)
- LLM: Free (local Phi-3.5-MoE)
- Chroma DB: Free

### Scaling Cost Estimates

**At 10,000 requests/day:**
```
Cloud Run: ~$5/month
Still within free tier
```

**At 100,000 requests/day:**
```
Cloud Run: ~$50/month
Exceeds free tier
Consider:
- Caching layer (Redis)
- Request batching
- Minimum instances for warm starts
```

## Technology Stack Details

### Core Dependencies

```python
# Agent & RAG
langchain==0.1.0
langchain-chroma==0.1.0
langchain-community==0.1.0
langchain-huggingface==0.1.0
chromadb==0.4.0

# LLM & Embeddings
transformers>=4.35.0
sentence-transformers==2.2.2
torch>=2.0.0  # CPU-only for CI

# MCP Tools
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
matplotlib==3.8.2
pypdf==4.0.0

# Frontend
streamlit==1.30.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0
```

### System Requirements

**Minimum (Keyword Mode only):**
- RAM: 4GB
- Disk: 2GB
- CPU: 2 cores
- Network: Broadband internet

**Recommended (Local LLM Mode):**
- RAM: 32GB
- Disk: 20GB SSD
- CPU: 8 cores / 16 threads
- GPU: Optional (4GB VRAM for faster LLM)
- Network: Broadband internet

## Observability & Monitoring

### Logging Strategy

```python
# Request logging format
{
  "timestamp": "2025-01-28T10:30:00Z",
  "request_id": "uuid-1234",
  "agent_mode": "keyword",
  "query": "calculate 2+2",
  "retrieval_latency_ms": 69,
  "tool_selected": "calculator",
  "tool_latency_ms": 139,
  "total_latency_ms": 208,
  "status": "success",
  "accuracy": 1.0
}
```

### Metrics Dashboard (Conceptual)

```
┌─────────────────────────────────────────────────────────────┐
│                    System Metrics                            │
├─────────────────────────────────────────────────────────────┤
│  Request Rate:          ▁▂▃▅▆█▇▅▃▂▁  (10 req/min avg)      │
│  Avg Latency (Keyword): 208ms        ████░░░░░░ (Target)   │
│  Avg Latency (LLM):     26s          ██████████ (Expected) │
│  Tool Success Rate:     100%         ██████████ (Perfect)  │
│  Retrieval Accuracy:    100%         ██████████ (Perfect)  │
│  Error Rate:            0%           ░░░░░░░░░░ (None)     │
│                                                              │
│  Active Cloud Run Instances:                                │
│  - plot-service:   0 (scaled to zero)                       │
│  - calculator:     0 (scaled to zero)                       │
│  - pdf-parser:     0 (scaled to zero)                       │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

### Phase 1: Performance
- [ ] Add request caching (Redis/Memcached)
- [ ] Implement connection pooling
- [ ] Add streaming LLM responses
- [ ] GPU acceleration for local LLM

### Phase 2: Features
- [ ] Multi-turn conversations with memory
- [ ] Additional MCP tools (image-gen, web-search)
- [ ] Multi-agent collaboration
- [ ] Batch request processing

### Phase 3: Production Readiness
- [ ] Add authentication & API keys
- [ ] Implement rate limiting
- [ ] Add Cloud Armor for DDoS protection
- [ ] Migrate to Kubernetes for more control
- [ ] Add comprehensive monitoring (Prometheus/Grafana)
- [ ] Implement A/B testing framework

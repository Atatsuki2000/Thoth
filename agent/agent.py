import os
import re
import requests
import json
import time
from typing import Optional, Callable
from .retriever import get_top_k

# Optional: Local LLM support
pipeline: Optional[Callable] = None
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_pipeline = None  # type: ignore

class SimpleAgent:
    def __init__(self, endpoints: dict | None = None, use_llm: bool = True, llm_api_key: str | None = None, llm_model: str = "local"):
        """
        endpoints: dict keys can include
          - 'plot': URL to plot-service MCP endpoint
          - 'calculator': URL to calculator MCP endpoint
          - 'pdf': URL to pdf-parser MCP endpoint
        use_llm: If True, use LLM to determine which tool to call instead of keywords (default: True)
        llm_api_key: OpenAI API key for LLM-based tool selection (only if llm_model="openai")
        llm_model: "local" for HuggingFace (free), "openai" for GPT-3.5 (paid)
        """
        # Allow environment-driven configuration; no hardcoded defaults
        env_endpoints = {
            'plot': os.getenv('PLOT_SERVICE_URL', ''),
            'calculator': os.getenv('CALCULATOR_URL', ''),
            'pdf': os.getenv('PDF_PARSER_URL', ''),
            'web_search': os.getenv('WEB_SEARCH_URL', ''),
            'file_ops': os.getenv('FILE_OPS_URL', ''),
        }
        self.endpoints = {**env_endpoints, **(endpoints or {})}
        self.use_llm = use_llm
        self.llm_model = llm_model
        # Use None for empty API key to make checking easier
        api_key = llm_api_key or os.getenv('OPENAI_API_KEY', '')
        self.llm_api_key = api_key if api_key else None
        
        # Initialize local LLM pipeline if needed
        self.local_llm = None
        if self.use_llm and self.llm_model == "local":
            if not HF_AVAILABLE:
                print("⚠️ transformers not installed. Install with: pip install transformers torch")
                print("Falling back to keyword mode")
                self.use_llm = False
            else:
                try:
                    print("Loading local LLM model (this may take a moment on first run)...")
                    # Use a small, fast model for tool selection
                    # Ensure pipeline is available for type checkers
                    assert hf_pipeline is not None
                    self.local_llm = hf_pipeline(
                        "text-generation",
                        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params, ~2GB
                        device_map="auto"
                    )
                    print("✅ Local LLM loaded successfully!")
                except Exception as e:
                    print(f"⚠️ Failed to load local LLM: {e}")
                    print("Falling back to keyword mode")
                    self.use_llm = False
        
        # Tool descriptions for LLM
        self.tool_descriptions = {
            'plot': 'Generate visualizations, charts, graphs, histograms, or any kind of plots',
            'calculator': 'Perform mathematical calculations, evaluate expressions, compute numbers',
            'pdf': 'Parse PDF documents, extract text from PDFs',
            'web_search': 'Search the web for current information, news, or facts not in the knowledge base',
            'file_ops': 'Read or write files in the workspace, list directory contents',
            'none': 'No tool needed, just retrieve information from knowledge base'
        }

    def _post(self, url: str, payload: dict, timeout: int = 30, retries: int = 3):
        """Post to MCP endpoint with retry logic and proper error handling."""
        last_error = None
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection failed (attempt {attempt + 1}/{retries}): {str(e)}"
                if attempt < retries - 1:
                    import time
                    time.sleep(1)  # Wait 1 second before retry
            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout (attempt {attempt + 1}/{retries}): {str(e)}"
                if attempt < retries - 1:
                    import time
                    time.sleep(1)
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {str(e)}"
                break  # Don't retry on HTTP errors (4xx, 5xx)
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                break
        
        return {
            "status": "error",
            "result": {"error": last_error or "Request failed"},
            "logs": ["request failed after retries"]
        }

    def _select_tool_with_llm(self, user_query: str, context: str) -> dict:
        """Use LLM to intelligently select which tool to call."""
        # Build available tools list
        available_tools = []
        for tool, desc in self.tool_descriptions.items():
            if tool == 'none' or self.endpoints.get(tool):
                available_tools.append(f"- {tool}: {desc}")

        prompt = f"""You are a deterministic tool selection module.
Analyze the user query and retrieved context and choose EXACTLY one tool.

User Query:
{user_query}

Context (truncated):
{context[:1500]}

Available Tools:
{chr(10).join(available_tools)}

Return ONLY valid JSON matching this schema (no extra text):
{{"tool": "calculator|plot|pdf|web_search|file_ops|none", "reasoning": "short justification"}}

CRITICAL RULES:
- 'calculator' ONLY for explicit math expressions (e.g., "25*17", "calculate 5+3", "square root of 144")
- 'plot' ONLY for explicit visualization requests (e.g., "plot a chart", "draw a graph", "create histogram")
- 'web_search' for questions requiring current/external information (e.g., "What is", "latest news", "search for")
- 'file_ops' for reading/writing files (e.g., "read file", "show contents of")
- 'pdf' ONLY if parsing a PDF document is explicitly mentioned
- 'none' for general knowledge questions that can be answered from context
- DEFAULT to 'none' if unsure - knowledge base will handle it

Examples:
- "What is Christmas?" → {{"tool": "none", "reasoning": "General knowledge question"}}
- "Calculate 25*17" → {{"tool": "calculator", "reasoning": "Math calculation"}}
- "Search for Python news" → {{"tool": "web_search", "reasoning": "Current information needed"}}
- "Read file README.md" → {{"tool": "file_ops", "reasoning": "File operation"}}

Respond with ONLY JSON."""

        # Use local HuggingFace model
        if self.llm_model == "local" and self.local_llm:
            try:
                # TinyLlama is too small for reliable tool selection
                # Fall back to keyword-based selection for better accuracy
                print("⚠️ Local LLM (TinyLlama) has limited tool selection accuracy. Using keyword-based selection instead.")
                return {"tool": "none", "reasoning": "Skipping TinyLlama, using keywords"}
                        
            except Exception as e:
                print(f"Local LLM tool selection failed: {e}")
                return {"tool": "none", "reasoning": f"Local LLM error: {str(e)}"}
            # Default fallback if no return occurred above
            return {"tool": "none", "reasoning": "Unable to parse local LLM output"}
        
        # Use OpenAI API
        elif self.llm_model == "openai":
            if not self.llm_api_key:
                print("⚠️  OpenAI API key not configured. Falling back to keyword-based tool selection.")
                print("   Tip: Set OPENAI_API_KEY environment variable or use 'local' LLM model.")
                return {"tool": "none", "reasoning": "No OpenAI API key - using keywords"}
            
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a tool selection assistant. Analyze queries and select the appropriate tool."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 200
                    },
                    timeout=10
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response (robust against extra text)
                json_match = re.search(r'\{\s*"tool"[^}]+\}', content)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    parsed = json.loads(content)
                tool = parsed.get("tool", "none").strip().lower()
                if tool not in {"plot", "calculator", "pdf", "web_search", "file_ops", "none"}:
                    tool = "none"
                reasoning = parsed.get("reasoning", "OpenAI JSON parsed")[:200]
                return {"tool": tool, "reasoning": reasoning}
                
            except requests.exceptions.HTTPError as e:
                error_msg = str(e)
                if "401" in error_msg or "authentication" in error_msg.lower():
                    print("❌ OpenAI API authentication failed. Please check your API key.")
                    print("   Tip: Switch to 'local' LLM model in sidebar to use free HuggingFace models.")
                    return {"tool": "none", "reasoning": "Invalid OpenAI API key - using keywords"}
                else:
                    print(f"OpenAI API HTTP error: {e}")
                    return {"tool": "none", "reasoning": f"OpenAI HTTP error - using keywords"}
            except Exception as e:
                print(f"OpenAI tool selection failed: {e}")
                return {"tool": "none", "reasoning": f"OpenAI error - using keywords"}
        
        else:
            return {"tool": "none", "reasoning": "Invalid LLM model configuration"}

    def _execute_calculator(self, user_query: str, reasoning: str = ""):
        """Execute calculator tool."""
        # Extract / normalize expression from query
        q = user_query.lower()
        # Common phrase to operator replacements
        replacements = [
            ("divided by", "/"),
            ("multiply by", "*"),
            ("multiplied by", "*"),
            ("times", "*"),
            ("x", "*"),
            ("plus", "+"),
            ("minus", "-"),
            ("to the power of", "**"),
            ("power of", "**"),
        ]
        for a, b in replacements:
            q = q.replace(a, b)
        # square root of N -> (N)**0.5
        sqrt_match = re.search(r'square\s+root\s+of\s+(\d+(?:\.\d+)?)', q)
        if sqrt_match:
            n = sqrt_match.group(1)
            expression = f"({n})**0.5"
        else:
            # Fallback: pull arithmetic expression - must start with digit or parenthesis
            expr_match = re.search(r'[\(\d][\d\s\+\-\*\/\^\(\)\.]*', q)
            if expr_match:
                expression = expr_match.group(0).strip().replace("^", "**")
            else:
                expression = "2 + 2"

        if not self.endpoints.get('calculator'):
            return {
                "plan": "Calculator requested but endpoint not configured",
                "tool_result": {
                    "status": "info",
                    "result": {"message": "Set CALCULATOR_URL or pass endpoints to SimpleAgent."},
                    "logs": ["calculator endpoint missing"]
                }
            }

        payload = {
            "mcp_version": "1.0",
            "tool": "calculator",
            "input": {"expression": expression},
            "metadata": {"request_id": "req-calc", "agent_id": "agent-v1"}
        }
        r = self._post(self.endpoints['calculator'], payload)
        return {"plan": f"Call calculator: {reasoning}" if reasoning else "Call calculator", "tool_result": r}

    def _execute_plot(self, user_query: str, reasoning: str = ""):
        """Execute plot service tool."""
        if not self.endpoints.get('plot'):
            return {
                "plan": "Plot requested but endpoint not configured",
                "tool_result": {
                    "status": "info",
                    "result": {"message": "Set PLOT_SERVICE_URL or pass endpoints to SimpleAgent."},
                    "logs": ["plot endpoint missing"]
                }
            }

        # Parse query to generate plotting code
        q = user_query.lower()

        # Extract function name - support both full names and abbreviations
        func_mapping = {
            'sine': 'sin', 'sin': 'sin',
            'cosine': 'cos', 'cos': 'cos',
            'tangent': 'tan', 'tan': 'tan',
            'logarithm': 'log', 'log': 'log',
            'exponential': 'exp', 'exp': 'exp',
            'square root': 'sqrt', 'sqrt': 'sqrt'
        }
        
        func_name = None
        for keyword, np_func in func_mapping.items():
            if keyword in q:
                func_name = np_func
                break

        if func_name:

            # Extract range (e.g., "from 0 to 10")
            range_match = re.search(r'from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)', q)
            if range_match:
                start, end = range_match.group(1), range_match.group(2)
            else:
                # Better defaults for trig functions
                if func_name in ['sin', 'cos', 'tan']:
                    start, end = "0", "6.28"  # 0 to 2π
                else:
                    start, end = "0", "10"

            # Generate plotting code
            code = f"""x = np.linspace({start}, {end}, 100)
y = np.{func_name}(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('{func_name}(x)')
plt.title('{func_name}(x) from {start} to {end}')
plt.grid(True)"""

            payload = {
                "mcp_version": "1.0",
                "tool": "plot-service",
                "input": {"code": code},
                "metadata": {"request_id": "req-plot", "agent_id": "agent-v1"}
            }
        else:
            # Fallback: use bar chart with dummy data
            payload = {
                "mcp_version": "1.0",
                "tool": "plot-service",
                "input": {
                    "data_reference": {
                        "type": "inline",
                        "payload": {
                            "columns": ["category", "value"],
                            "rows": [["A", 10], ["B", 15], ["C", 7]]
                        }
                    }
                },
                "metadata": {"request_id": "req-plot", "agent_id": "agent-v1"}
            }

        r = self._post(self.endpoints['plot'], payload)
        return {"plan": f"Call plot-service: {reasoning}" if reasoning else "Call plot-service", "tool_result": r}

    def _execute_pdf(self, user_query: str, reasoning: str = ""):
        """Execute PDF parser tool."""
        return {
            "plan": f"PDF parser: {reasoning}" if reasoning else "Call pdf-parser (requires PDF input)",
            "tool_result": {
                "status": "info",
                "result": {"message": "PDF parser available. Please provide PDF content in base64 format."},
                "logs": ["pdf-parser ready"]
            }
        }
    
    def _execute_web_search(self, user_query: str, reasoning: str = ""):
        """Execute web search tool and generate answer from results."""
        if not self.endpoints.get('web_search'):
            return {"plan": "Web search unavailable", "tool_result": {"status": "error", "result": {"error": "Web search endpoint not configured"}}}
        
        # Extract search query from user query
        query = user_query
        
        payload = {
            "query": query,
            "max_results": 5
        }
        
        r = self._post(self.endpoints['web_search'], payload)
        
        # Generate natural language answer from search results if using OpenAI
        if self.llm_model == "openai" and self.llm_api_key and isinstance(r, dict):
            try:
                # Extract search results
                results = r.get('results', []) if 'results' in r else r.get('result', {}).get('results', [])
                if results:
                    # Build context from top 3 results
                    context = "\n\n".join([f"Source: {res['title']}\n{res['snippet']}" for res in results[:3]])
                    
                    # Generate answer using OpenAI
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.llm_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-3.5-turbo",
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided web search results. Be concise and informative."},
                                {"role": "user", "content": f"Question: {user_query}\n\nWeb Search Results:\n{context}\n\nAnswer:"}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 300
                        },
                        timeout=15
                    )
                    response.raise_for_status()
                    answer = response.json()['choices'][0]['message']['content']
                    
                    # Return answer with sources
                    return {
                        "plan": f"Search web and generate answer for: {query}",
                        "tool_result": {
                            "status": "success",
                            "result": {
                                "answer": answer,
                                "sources": results[:3]
                            }
                        }
                    }
            except Exception as e:
                print(f"Failed to generate answer from search results: {e}")
                # Fall back to returning raw results
        
        # Wrap response in MCP format if successful (fallback)
        if isinstance(r, dict) and 'status' not in r:
            # FastAPI response - wrap it
            return {"plan": f"Search web for: {query}", "tool_result": {"status": "success", "result": r}}
        
        return {"plan": f"Search web: {reasoning}" if reasoning else f"Search web for: {query}", "tool_result": r}
    
    def _execute_file_read(self, user_query: str, reasoning: str = ""):
        """Execute file read tool."""
        if not self.endpoints.get('file_ops'):
            return {"plan": "File operations unavailable", "tool_result": {"status": "error", "result": {"error": "File operations endpoint not configured"}}}
        
        # Try to extract file path from query
        import re
        path_match = re.search(r'[\w/\\.-]+\.\w+', user_query)
        file_path = path_match.group(0) if path_match else "README.md"
        
        payload = {
            "path": file_path
        }
        
        r = self._post(self.endpoints['file_ops'], payload)
        
        # Wrap response in MCP format if successful
        if isinstance(r, dict) and 'status' not in r:
            # FastAPI response - wrap it
            return {"plan": f"Read file: {file_path}", "tool_result": {"status": "success", "result": r}}
        
        return {"plan": f"Read file: {reasoning}" if reasoning else f"Read file: {file_path}", "tool_result": r}


    def plan_and_execute(self, user_query):
        start_total = time.time()
        # Step 1: Retrieve top-k documents
        start_retrieval = time.time()
        docs = get_top_k(user_query, k=3)
        retrieval_time_ms = (time.time() - start_retrieval) * 1000
        context = "\n\n".join([d.page_content for d in docs])
        print("Context for planning:", context)

        # Step 2: Determine which tool to use
        if self.use_llm:
            start_llm = time.time()
            tool_selection = self._select_tool_with_llm(user_query, context)
            llm_time_ms = (time.time() - start_llm) * 1000
            selected_tool = tool_selection.get('tool', 'none')
            reasoning = tool_selection.get('reasoning', '')
            print(f"LLM selected tool: {selected_tool} - {reasoning}")
            # Execute selected tool
            if selected_tool == 'calculator':
                exe = self._execute_calculator(user_query, reasoning)
            elif selected_tool == 'plot':
                exe = self._execute_plot(user_query, reasoning)
            elif selected_tool == 'pdf':
                exe = self._execute_pdf(user_query, reasoning)
            elif selected_tool == 'web_search':
                exe = self._execute_web_search(user_query, reasoning)
            elif selected_tool == 'file_ops':
                exe = self._execute_file_read(user_query, reasoning)
            else:
                # Hybrid fallback: if LLM returns 'none', try keyword heuristics
                ql = user_query.lower()
                # More specific keywords to avoid false positives
                calc_keywords = ["calculate", "compute", "calc ", "evaluate", " math", "square root", "multiply", "divide"]
                calc_patterns = [r'\d+\s*[\+\-\*\/\^x]\s*\d+', r'\bsum of\b', r'\btimes\b', r'\bplus\b', r'\bminus\b']
                plot_keywords = ["plot ", "histogram", "chart", "graph", "visualiz", "draw chart", "generate plot", "bar chart", "line graph", "scatter plot"]
                search_keywords = ["search for", "find online", "look up online", "google", "current", "latest", "news about", "what is"]
                file_keywords = ["read file", "open file", "show file", "file content", "read the file"]
                
                # Check patterns and keywords
                has_calc = any(k in ql for k in calc_keywords) or any(re.search(p, ql) for p in calc_patterns)
                has_plot = any(k in ql for k in plot_keywords)
                has_search = any(k in ql for k in search_keywords)
                has_file = any(k in ql for k in file_keywords)
                
                if has_calc:
                    selected_tool = 'calculator'
                    exe = self._execute_calculator(user_query, "Keyword fallback: calculation detected")
                elif has_plot:
                    selected_tool = 'plot'
                    exe = self._execute_plot(user_query, "Keyword fallback: visualization detected")
                elif has_search:
                    selected_tool = 'web_search'
                    exe = self._execute_web_search(user_query, "Keyword fallback: web search detected")
                elif has_file:
                    selected_tool = 'file_ops'
                    exe = self._execute_file_read(user_query, "Keyword fallback: file operation detected")
                else:
                    exe = {"plan": f"Query knowledge base. {reasoning}", "tool_result": None}
            exe["selected_tool"] = selected_tool
            exe["retrieval_time_ms"] = retrieval_time_ms
            exe["llm_time_ms"] = llm_time_ms
            exe["end_to_end_ms"] = (time.time() - start_total) * 1000
            return exe
        
        # Fallback to keyword-based selection
        query_lower = user_query.lower()
        # More precise keywords to avoid false positives
        calc_keywords = ["calculate", "compute", "calc ", "evaluate", " math", "square root", "multiply", "divide"]
        calc_patterns = [r'\d+\s*[\+\-\*\/\^x]\s*\d+', r'\bsum of\b', r'\btimes\b', r'\bplus\b', r'\bminus\b']
        plot_keywords = ["plot ", "histogram", "chart", "graph", "visualiz", "draw chart", "generate plot", "bar chart", "line graph"]
        pdf_keywords = ["pdf", "parse pdf", "extract from", "document"]
        search_keywords = ["search for", "find online", "look up online", "what is", "who is", "latest", "current news"]
        file_keywords = ["read file", "open file", "show file"]

        selected_tool = "none"
        # Check calculator with patterns and keywords
        if any(word in query_lower for word in calc_keywords) or any(re.search(p, query_lower) for p in calc_patterns):
            exe = self._execute_calculator(user_query, "Keyword match: calculation detected")
            selected_tool = "calculator"
        elif any(word in query_lower for word in plot_keywords):
            exe = self._execute_plot(user_query, "Keyword match: visualization detected")
            selected_tool = "plot"
        elif any(word in query_lower for word in search_keywords):
            exe = self._execute_web_search(user_query, "Keyword match: web search detected")
            selected_tool = "web_search"
        elif any(word in query_lower for word in file_keywords):
            exe = self._execute_file_read(user_query, "Keyword match: file operation detected")
            selected_tool = "file_ops"
        elif any(word in query_lower for word in pdf_keywords):
            exe = self._execute_pdf(user_query, "Keyword match: PDF operation detected")
            selected_tool = "pdf"
        else:
            exe = {"plan": "Query knowledge base", "tool_result": None}

        exe["selected_tool"] = selected_tool
        exe["retrieval_time_ms"] = retrieval_time_ms
        exe["end_to_end_ms"] = (time.time() - start_total) * 1000
        return exe

if __name__ == "__main__":
    # Allow testing from CLI with environment-configured endpoints
    agent = SimpleAgent()
    query = input("Enter your query: ")
    result = agent.plan_and_execute(query)
    print(result)

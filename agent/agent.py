import os
import re
import requests
import json
import time
from typing import Optional, Callable
from retriever import get_top_k

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
        }
        self.endpoints = {**env_endpoints, **(endpoints or {})}
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key or os.getenv('OPENAI_API_KEY', '')
        
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
{{"tool": "plot|calculator|pdf|none", "reasoning": "short justification"}}
Rules:
- 'calculator' for arithmetic / numeric computation requests.
- 'plot' for visualization, chart, graph, histogram, bar, line, scatter, draw, diagram.
- 'pdf' only if parsing / extracting from a PDF/document mentioned.
- 'none' if no tool action required.
- Prefer calculator over plot if user wants a numeric answer.
- Prefer plot if explicit visualization requested.
Respond with ONLY JSON."""

        # Use local HuggingFace model
        if self.llm_model == "local" and self.local_llm:
            try:
                # Simplified prompt that works with TinyLlama
                chat_prompt = f"""Select tool for: {user_query}
Options: calculator, plot, pdf, none
Answer: """
                full_response = self.local_llm(
                    chat_prompt,
                    max_new_tokens=10,  # Reduced from 50 - only need one word
                    do_sample=False,     # Greedy decoding for speed
                    pad_token_id=self.local_llm.tokenizer.eos_token_id  # Proper padding
                )[0]['generated_text']

                # Extract only the generated part (after the prompt)
                generated_part = full_response[len(chat_prompt):].strip().lower()

                # Direct keyword detection in ONLY the generated part
                if 'calculator' in generated_part or 'calculate' in generated_part:
                    return {"tool": "calculator", "reasoning": f"TinyLlama selected calculator: {generated_part}"}
                elif 'plot' in generated_part or 'visual' in generated_part or 'chart' in generated_part or 'graph' in generated_part:
                    return {"tool": "plot", "reasoning": f"TinyLlama selected plot: {generated_part}"}
                elif 'pdf' in generated_part:
                    return {"tool": "pdf", "reasoning": f"TinyLlama selected PDF: {generated_part}"}
                else:
                    return {"tool": "none", "reasoning": f"TinyLlama selected none: {generated_part}"}
                        
            except Exception as e:
                print(f"Local LLM tool selection failed: {e}")
                return {"tool": "none", "reasoning": f"Local LLM error: {str(e)}"}
            # Default fallback if no return occurred above
            return {"tool": "none", "reasoning": "Unable to parse local LLM output"}
        
        # Use OpenAI API
        elif self.llm_model == "openai":
            if not self.llm_api_key:
                return {"tool": "none", "reasoning": "No OpenAI API key provided"}
            
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
                if tool not in {"plot", "calculator", "pdf", "none"}:
                    tool = "none"
                reasoning = parsed.get("reasoning", "OpenAI JSON parsed")[:200]
                return {"tool": tool, "reasoning": reasoning}
                
            except Exception as e:
                print(f"OpenAI tool selection failed: {e}")
                return {"tool": "none", "reasoning": f"OpenAI error: {str(e)}"}
        
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

        # Extract function name (sin, cos, tan, log, exp, etc.)
        func_match = re.search(r'(sin|cos|tan|log|exp|sqrt)\s*\(?\s*x\s*\)?', q)

        if func_match:
            func_name = func_match.group(1)

            # Extract range (e.g., "from 0 to 10")
            range_match = re.search(r'from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)', q)
            if range_match:
                start, end = range_match.group(1), range_match.group(2)
            else:
                start, end = "0", "10"  # Default range

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
            else:
                # Hybrid fallback: if LLM returns 'none', try keyword heuristics
                ql = user_query.lower()
                calc_keywords = ["calculate", "compute", "calc", "evaluate", "math", "+", "-", "*", "/", "^", "divide", "times", "plus", "minus", "square root"]
                plot_keywords = ["plot", "histogram", "chart", "graph", "visualiz", "visual", "draw", "generate", "bar", "line", "scatter", "diagram", "figure"]
                if any(k in ql for k in calc_keywords) or re.search(r'[\d\s]+[\+\-\*\/^][\d\s]+', ql):
                    selected_tool = 'calculator'
                    exe = self._execute_calculator(user_query, "Keyword fallback: calculation detected")
                elif any(k in ql for k in plot_keywords):
                    selected_tool = 'plot'
                    exe = self._execute_plot(user_query, "Keyword fallback: visualization detected")
                else:
                    exe = {"plan": f"No tool needed. {reasoning}", "tool_result": None}
            exe["selected_tool"] = selected_tool
            exe["retrieval_time_ms"] = retrieval_time_ms
            exe["llm_time_ms"] = llm_time_ms
            exe["end_to_end_ms"] = (time.time() - start_total) * 1000
            return exe
        
        # Fallback to keyword-based selection
        query_lower = user_query.lower()
        calc_keywords = ["calculate", "compute", "calc", "evaluate", "math", "sum", "add", "subtract", "multiply", "divide", "square root"]
        plot_keywords = ["plot", "histogram", "chart", "graph", "visualiz", "visual", "draw", "generate", "bar", "line", "scatter", "diagram", "figure"]
        pdf_keywords = ["pdf", "parse", "extract", "document", "file", "pages"]

        selected_tool = "none"
        if any(word in query_lower for word in calc_keywords) or re.search(r'[\d\s]+[\+\-\*\/^][\d\s]+', query_lower):
            exe = self._execute_calculator(user_query, "Keyword match: calculation detected")
            selected_tool = "calculator"
        elif any(word in query_lower for word in plot_keywords):
            exe = self._execute_plot(user_query, "Keyword match: visualization detected")
            selected_tool = "plot"
        elif any(word in query_lower for word in pdf_keywords):
            exe = self._execute_pdf(user_query, "Keyword match: PDF operation detected")
            selected_tool = "pdf"
        else:
            exe = {"plan": "No tool call needed", "tool_result": None}

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

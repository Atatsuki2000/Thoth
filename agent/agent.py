import os
import re
import requests
import json
from retriever import get_top_k

# Optional: Local LLM support
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class SimpleAgent:
    def __init__(self, endpoints: dict | None = None, use_llm: bool = False, llm_api_key: str | None = None, llm_model: str = "local"):
        """
        endpoints: dict keys can include
          - 'plot': URL to plot-service MCP endpoint
          - 'calculator': URL to calculator MCP endpoint
          - 'pdf': URL to pdf-parser MCP endpoint
        use_llm: If True, use LLM to determine which tool to call instead of keywords
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
                    self.local_llm = pipeline(
                        "text-generation",
                        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params, ~2GB
                        device_map="auto",
                        max_new_tokens=100
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
        
        prompt = f"""Given the user's query and retrieved context, determine which tool to use.

User Query: {user_query}

Retrieved Context:
{context}

Available Tools:
{chr(10).join(available_tools)}

Respond in JSON format with:
{{
    "tool": "plot|calculator|pdf|none",
    "reasoning": "brief explanation of why this tool was chosen"
}}

Choose the most appropriate tool. If no tool is needed, select "none"."""

        # Use local HuggingFace model
        if self.llm_model == "local" and self.local_llm:
            try:
                # Format prompt for TinyLlama chat model
                chat_prompt = f"""<|system|>
You are a tool selection assistant. Analyze the user's query and select the appropriate tool.</s>
<|user|>
{prompt}</s>
<|assistant|>
"""
                response = self.local_llm(chat_prompt, max_new_tokens=100, temperature=0.3)[0]['generated_text']
                
                # Extract JSON from response
                # Look for JSON between { and }
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    tool_selection = json.loads(json_match.group(0))
                    return tool_selection
                else:
                    # Fallback: simple keyword detection in response
                    response_lower = response.lower()
                    if 'calculator' in response_lower or 'calculate' in response_lower:
                        return {"tool": "calculator", "reasoning": "Local LLM detected calculation intent"}
                    elif 'plot' in response_lower or 'visual' in response_lower:
                        return {"tool": "plot", "reasoning": "Local LLM detected visualization intent"}
                    elif 'pdf' in response_lower:
                        return {"tool": "pdf", "reasoning": "Local LLM detected PDF intent"}
                    else:
                        return {"tool": "none", "reasoning": "Local LLM suggests no tool needed"}
                        
            except Exception as e:
                print(f"Local LLM tool selection failed: {e}")
                return {"tool": "none", "reasoning": f"Local LLM error: {str(e)}"}
        
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
                
                # Parse JSON response
                tool_selection = json.loads(content)
                return tool_selection
                
            except Exception as e:
                print(f"OpenAI tool selection failed: {e}")
                return {"tool": "none", "reasoning": f"OpenAI error: {str(e)}"}
        
        else:
            return {"tool": "none", "reasoning": "Invalid LLM model configuration"}

    def _execute_calculator(self, user_query: str, reasoning: str = ""):
        """Execute calculator tool."""
        # Extract expression from query
        expr_match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', user_query)
        expression = expr_match.group(0).strip() if expr_match else "2 + 2"

        if not self.endpoints.get('calculator'):
            return {
                "plan": f"Calculator: {reasoning}" if reasoning else "Calculator requested but endpoint not configured",
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
                "plan": f"Plot: {reasoning}" if reasoning else "Plot requested but endpoint not configured",
                "tool_result": {
                    "status": "info",
                    "result": {"message": "Set PLOT_SERVICE_URL or pass endpoints to SimpleAgent."},
                    "logs": ["plot endpoint missing"]
                }
            }
        
        payload = {
            "mcp_version": "1.0",
            "tool": "plot-service",
            "input": {
                "instructions": "Plot a histogram of the numeric column 'value' grouped by 'category'",
                "data_reference": {
                    "type": "inline",
                    "payload": {
                        "columns": ["category", "value"],
                        "rows": [["A", 10], ["B", 3], ["A", 2]]
                    }
                },
                "options": {"bins": 10, "title": "Value by Category"}
            },
            "metadata": {"request_id": "req-123", "agent_id": "agent-v1"}
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
        # Step 1: Retrieve top-k documents
        docs = get_top_k(user_query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        print("Context for planning:", context)

        # Step 2: Determine which tool to use
        if self.use_llm:
            # Use LLM for intelligent tool selection
            tool_selection = self._select_tool_with_llm(user_query, context)
            selected_tool = tool_selection.get('tool', 'none')
            reasoning = tool_selection.get('reasoning', '')
            print(f"LLM selected tool: {selected_tool} - {reasoning}")
            
            # Execute selected tool
            if selected_tool == 'calculator':
                return self._execute_calculator(user_query, reasoning)
            elif selected_tool == 'plot':
                return self._execute_plot(user_query, reasoning)
            elif selected_tool == 'pdf':
                return self._execute_pdf(user_query, reasoning)
            else:
                return {"plan": f"No tool needed. {reasoning}", "tool_result": None}
        
        # Fallback to keyword-based selection
        query_lower = user_query.lower()
        
        # Check for calculator keywords
        if any(word in query_lower for word in ["calculate", "compute", "calc", "evaluate", "math", "+", "-", "*", "/", "="]):
            return self._execute_calculator(user_query, "Keyword match: calculation detected")
        
        # Check for plot keywords
        elif any(word in query_lower for word in ["plot", "histogram", "chart", "graph", "visualiz", "visual", "draw", "generate"]):
            return self._execute_plot(user_query, "Keyword match: visualization detected")
        
        # Check for PDF keywords
        elif any(word in query_lower for word in ["pdf", "parse", "extract", "document"]):
            return self._execute_pdf(user_query, "Keyword match: PDF operation detected")
        
        else:
            return {"plan": "No tool call needed", "tool_result": None}

if __name__ == "__main__":
    # Allow testing from CLI with environment-configured endpoints
    agent = SimpleAgent()
    query = input("Enter your query: ")
    result = agent.plan_and_execute(query)
    print(result)

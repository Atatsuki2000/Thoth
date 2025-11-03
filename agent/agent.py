import os
import re
import requests
from retriever import get_top_k

class SimpleAgent:
    def __init__(self, endpoints: dict | None = None):
        """
        endpoints: dict keys can include
          - 'plot': URL to plot-service MCP endpoint
          - 'calculator': URL to calculator MCP endpoint
          - 'pdf': URL to pdf-parser MCP endpoint
        If not provided, the agent will not call that tool.
        """
        # Allow environment-driven configuration; no hardcoded defaults
        env_endpoints = {
            'plot': os.getenv('PLOT_SERVICE_URL', ''),
            'calculator': os.getenv('CALCULATOR_URL', ''),
            'pdf': os.getenv('PDF_PARSER_URL', ''),
        }
        self.endpoints = {**env_endpoints, **(endpoints or {})}

    def _post(self, url: str, payload: dict, timeout: int = 30):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "result": {"error": str(e)}, "logs": ["request failed"]}

    def plan_and_execute(self, user_query):
        # Step 1: Retrieve top-k documents
        docs = get_top_k(user_query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        print("Context for planning:", context)

        # Step 2: Simple rule-based plan (call appropriate tool based on keywords)
        query_lower = user_query.lower()
        
        # Check for calculator keywords
        if any(word in query_lower for word in ["calculate", "compute", "calc", "evaluate", "math", "+", "-", "*", "/", "="]):
            # Extract expression (simple approach - look for numbers and operators)
            expr_match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', user_query)
            expression = expr_match.group(0).strip() if expr_match else "2 + 2"

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
            return {"plan": "Call calculator", "tool_result": r}
        
        # Check for plot keywords
        elif any(word in query_lower for word in ["plot", "histogram", "chart", "graph"]):
            if not self.endpoints.get('plot'):
                return {
                    "plan": "Plot requested but endpoint not configured",
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
            return {"plan": "Call plot-service", "tool_result": r}
        
        # Check for PDF keywords
        elif any(word in query_lower for word in ["pdf", "parse", "extract", "document"]):
            return {
                "plan": "Call pdf-parser (requires PDF input)", 
                "tool_result": {
                    "status": "info",
                    "result": {"message": "PDF parser available. Please provide PDF content in base64 format."},
                    "logs": ["pdf-parser ready"]
                }
            }
        
        else:
            return {"plan": "No tool call needed", "tool_result": None}

if __name__ == "__main__":
    # Allow testing from CLI with environment-configured endpoints
    agent = SimpleAgent()
    query = input("Enter your query: ")
    result = agent.plan_and_execute(query)
    print(result)

from fastapi import FastAPI
from pydantic import BaseModel
import numexpr as ne
import traceback

app = FastAPI()

class MCPInput(BaseModel):
    mcp_version: str
    tool: str
    input: dict
    metadata: dict = {}

@app.post("/mcp/calculate")
def calculate_endpoint(req: MCPInput):
    """
    Safely evaluate mathematical expressions.
    Example input: {"expression": "2 + 2 * 3"}
    """
    try:
        expression = req.input.get('expression', '')
        if not expression:
            return {
                "status": "error",
                "result": {"error": "No expression provided"},
                "logs": ["error: missing expression"]
            }
        
        # Use numexpr for safe evaluation
        result = ne.evaluate(expression)
        
        return {
            "status": "success",
            "result": {
                "expression": expression,
                "value": float(result),
                "type": "numeric"
            },
            "logs": [f"evaluated: {expression} = {result}"]
        }
    except Exception as e:
        return {
            "status": "error",
            "result": {
                "error": str(e),
                "traceback": traceback.format_exc()
            },
            "logs": [f"error: {str(e)}"]
        }

@app.get("/health")
def health_check():
    return {"status": "healthy", "tool": "calculator"}

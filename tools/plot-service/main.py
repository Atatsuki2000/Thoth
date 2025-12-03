from fastapi import FastAPI
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io, base64

app = FastAPI()

class MCPInput(BaseModel):
    mcp_version: str
    tool: str
    input: dict
    metadata: dict = {}

@app.post("/mcp/plot")
def plot_endpoint(req: MCPInput):
    try:
        # Check if this is a code-based plot (mathematical function)
        if 'code' in req.input:
            code = req.input['code']
            # Execute Python plotting code safely with NumPy functions available
            import numpy as np
            allowed_globals = {
                'np': np,
                'plt': plt,
                '__builtins__': {
                    'range': range,
                    'len': len,
                    'list': list,
                    'float': float,
                    'int': int
                }
            }
            plt.figure(figsize=(8,5))
            exec(code, allowed_globals)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('utf-8')
            return {"status":"success", "result": {"artifact_type":"image/png", "artifact_base64": b64}, "logs": ["executed code plot"]}

        # Otherwise, use the bar chart mode (categorical data)
        rows = req.input.get('data_reference', {}).get('payload', {}).get('rows', [])
        if not rows:
            return {"status":"error", "logs": ["No data provided"]}

        categories = [r[0] for r in rows]
        values = [r[1] for r in rows]
        plt.figure(figsize=(6,4))
        plt.bar(categories, values)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return {"status":"success", "result": {"artifact_type":"image/png", "artifact_base64": b64}, "logs": ["bar chart plotted"]}
    except Exception as e:
        return {"status":"error", "logs": [f"error: {str(e)}"]}

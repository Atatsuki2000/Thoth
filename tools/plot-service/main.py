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
    rows = req.input.get('data_reference', {}).get('payload', {}).get('rows', [])
    categories = [r[0] for r in rows]
    values = [r[1] for r in rows]
    plt.figure(figsize=(6,4))
    plt.bar(categories, values)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"status":"success", "result": {"artifact_type":"image/png", "artifact_base64": b64}, "logs": ["plotted"]}

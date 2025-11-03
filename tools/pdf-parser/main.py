from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pypdf
import traceback
import io
import base64

app = FastAPI()

class MCPInput(BaseModel):
    mcp_version: str
    tool: str
    input: dict
    metadata: dict = {}

@app.post("/mcp/parse")
def parse_pdf_endpoint(req: MCPInput):
    """
    Extract text from a PDF file.
    Input can be:
    - {"pdf_base64": "base64_encoded_pdf_content"}
    - {"pdf_url": "http://..."}
    """
    try:
        pdf_base64 = req.input.get('pdf_base64', '')
        
        if not pdf_base64:
            return {
                "status": "error",
                "result": {"error": "No PDF content provided. Use 'pdf_base64' field."},
                "logs": ["error: missing pdf_base64"]
            }
        
        # Decode base64 PDF
        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Parse PDF
        reader = pypdf.PdfReader(pdf_file)
        num_pages = len(reader.pages)
        
        # Extract text from all pages
        text_content = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            text_content.append({
                "page": i + 1,
                "text": text
            })
        
        # Combine all text
        full_text = "\n\n".join([f"Page {p['page']}:\n{p['text']}" for p in text_content])
        
        return {
            "status": "success",
            "result": {
                "num_pages": num_pages,
                "pages": text_content,
                "full_text": full_text,
                "type": "pdf_extraction"
            },
            "logs": [f"parsed {num_pages} pages"]
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
    return {"status": "healthy", "tool": "pdf-parser"}

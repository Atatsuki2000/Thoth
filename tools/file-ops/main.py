"""
File Operations MCP Tool
Provides safe file read/write operations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from pathlib import Path
import json

app = FastAPI(
    title="File Operations MCP Tool",
    description="Safe file read and write operations",
    version="1.0.0"
)

# Safety: Only allow operations in workspace directory
WORKSPACE_DIR = Path(os.getenv('WORKSPACE_DIR', os.getcwd()))

def is_safe_path(path: str) -> bool:
    """Check if path is within workspace directory."""
    try:
        full_path = (WORKSPACE_DIR / path).resolve()
        return full_path.is_relative_to(WORKSPACE_DIR)
    except:
        return False

class ReadFileRequest(BaseModel):
    path: str

class ReadFileResponse(BaseModel):
    path: str
    content: str
    size: int
    encoding: str = "utf-8"

class WriteFileRequest(BaseModel):
    path: str
    content: str
    append: bool = False

class WriteFileResponse(BaseModel):
    path: str
    bytes_written: int
    status: str

class ListFilesRequest(BaseModel):
    path: str = "."
    pattern: str = "*"

class FileInfo(BaseModel):
    name: str
    path: str
    is_directory: bool
    size: Optional[int]

class ListFilesResponse(BaseModel):
    directory: str
    files: List[FileInfo]
    count: int

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "File Operations MCP Tool",
        "status": "running",
        "version": "1.0.0",
        "workspace": str(WORKSPACE_DIR),
        "capabilities": ["read_file", "write_file", "list_files"]
    }

@app.post("/mcp/read_file", response_model=ReadFileResponse)
async def read_file(request: ReadFileRequest):
    """
    Read a text file from the workspace.
    
    Args:
        path: Path to file (relative to workspace)
    
    Returns:
        File content and metadata
    """
    if not is_safe_path(request.path):
        raise HTTPException(
            status_code=403,
            detail="Access denied: Path is outside workspace"
        )
    
    file_path = WORKSPACE_DIR / request.path
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.path}"
        )
    
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Not a file: {request.path}"
        )
    
    try:
        content = file_path.read_text(encoding='utf-8')
        return ReadFileResponse(
            path=request.path,
            content=content,
            size=len(content),
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File is not a text file (binary content detected)"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read file: {str(e)}"
        )

@app.post("/mcp/write_file", response_model=WriteFileResponse)
async def write_file(request: WriteFileRequest):
    """
    Write content to a file in the workspace.
    
    Args:
        path: Path to file (relative to workspace)
        content: Content to write
        append: If True, append to existing file (default: False)
    
    Returns:
        Write status and bytes written
    """
    if not is_safe_path(request.path):
        raise HTTPException(
            status_code=403,
            detail="Access denied: Path is outside workspace"
        )
    
    file_path = WORKSPACE_DIR / request.path
    
    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write or append
        mode = 'a' if request.append else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            bytes_written = f.write(request.content)
        
        return WriteFileResponse(
            path=request.path,
            bytes_written=bytes_written,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write file: {str(e)}"
        )

@app.post("/mcp/list_files", response_model=ListFilesResponse)
async def list_files(request: ListFilesRequest):
    """
    List files in a directory.
    
    Args:
        path: Directory path (relative to workspace, default: ".")
        pattern: Glob pattern (default: "*")
    
    Returns:
        List of files and directories
    """
    if not is_safe_path(request.path):
        raise HTTPException(
            status_code=403,
            detail="Access denied: Path is outside workspace"
        )
    
    dir_path = WORKSPACE_DIR / request.path
    
    if not dir_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {request.path}"
        )
    
    if not dir_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Not a directory: {request.path}"
        )
    
    try:
        files = []
        for item in dir_path.glob(request.pattern):
            files.append(FileInfo(
                name=item.name,
                path=str(item.relative_to(WORKSPACE_DIR)),
                is_directory=item.is_dir(),
                size=item.stat().st_size if item.is_file() else None
            ))
        
        return ListFilesResponse(
            directory=request.path,
            files=files,
            count=len(files)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

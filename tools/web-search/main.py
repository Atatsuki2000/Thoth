"""
Web Search MCP Tool
Provides web search capabilities using DuckDuckGo (no API key needed)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup
import re

app = FastAPI(
    title="Web Search MCP Tool",
    description="Search the web and retrieve information",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Web Search MCP Tool",
        "status": "running",
        "version": "1.0.0",
        "capabilities": ["web_search", "get_page_content"]
    }

@app.post("/mcp/search", response_model=SearchResponse)
async def search_web(request: SearchRequest):
    """
    Search the web using DuckDuckGo HTML interface.
    
    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)
    
    Returns:
        List of search results with title, snippet, and URL
    """
    try:
        # Use DuckDuckGo HTML version (no API key needed)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": request.query},
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
        
        # Parse results
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result_div in soup.find_all('div', class_='result', limit=request.max_results):
            # Extract title
            title_elem = result_div.find('a', class_='result__a')
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            url = title_elem.get('href', '')
            
            # Extract snippet
            snippet_elem = result_div.find('a', class_='result__snippet')
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            # Clean URL (DuckDuckGo wraps URLs)
            if url.startswith('/l/?'):
                # Extract actual URL from redirect
                match = re.search(r'uddg=([^&]+)', url)
                if match:
                    from urllib.parse import unquote
                    url = unquote(match.group(1))
            
            results.append(SearchResult(
                title=title,
                snippet=snippet,
                url=url
            ))
        
        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

class PageContentRequest(BaseModel):
    url: str
    max_length: int = 5000

class PageContentResponse(BaseModel):
    url: str
    title: Optional[str]
    content: str
    length: int

@app.post("/mcp/get_page", response_model=PageContentResponse)
async def get_page_content(request: PageContentRequest):
    """
    Retrieve and extract text content from a web page.
    
    Args:
        url: URL of the page to retrieve
        max_length: Maximum content length (default: 5000 chars)
    
    Returns:
        Page title and text content
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(
                request.url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get title
        title = soup.title.string if soup.title else None
        
        # Get main text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = '\n'.join(lines)
        
        # Truncate if needed
        if len(content) > request.max_length:
            content = content[:request.max_length] + "... [truncated]"
        
        return PageContentResponse(
            url=request.url,
            title=title,
            content=content,
            length=len(content)
        )
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Failed to retrieve page: {e.response.status_code}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve page: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

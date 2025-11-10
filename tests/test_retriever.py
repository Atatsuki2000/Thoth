import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

from retriever import build_index, get_top_k

def test_retriever_basic():
    """Test basic retrieval functionality."""
    texts = [
        "Python is a programming language",
        "Matplotlib creates plots and visualizations",
        "FastAPI is a web framework"
    ]
    
    # Build index
    build_index(texts, persist_directory="db/test_chroma")
    
    # Query for plotting
    docs = get_top_k("plotting library", k=2, persist_directory="db/test_chroma")
    
    assert len(docs) > 0, "Should retrieve at least one document"
    assert "Matplotlib" in docs[0].page_content or "plot" in docs[0].page_content.lower()

def test_retriever_no_duplicates():
    """Test that retriever removes duplicate results."""
    texts = ["Document about Python"] * 5  # Same text repeated
    
    build_index(texts, persist_directory="db/test_chroma_dup")
    docs = get_top_k("Python", k=5, persist_directory="db/test_chroma_dup")
    
    # Should only return 1 unique document despite multiple copies
    assert len(docs) <= 1, "Should remove duplicate documents"

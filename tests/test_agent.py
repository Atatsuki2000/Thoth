import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

from agent import SimpleAgent

def test_agent_calculator_detection():
    """Test agent detects calculator keywords and plans accordingly."""
    agent = SimpleAgent({
        "calculator": "http://127.0.0.1:8001/mcp/calculate",
        "plot": "http://127.0.0.1:8000/mcp/plot"
    })
    
    result = agent.plan_and_execute("calculate 5 + 3")
    assert "calculator" in result["plan"].lower(), "Should detect calculator keyword"

def test_agent_plot_detection():
    """Test agent detects plot keywords."""
    agent = SimpleAgent({
        "calculator": "http://127.0.0.1:8001/mcp/calculate",
        "plot": "http://127.0.0.1:8000/mcp/plot"
    })
    
    result = agent.plan_and_execute("show me a histogram")
    assert "plot" in result["plan"].lower(), "Should detect plot keyword"

def test_agent_missing_endpoint():
    """Test agent handles missing endpoint gracefully."""
    agent = SimpleAgent({})  # No endpoints configured
    
    result = agent.plan_and_execute("calculate 2 + 2")
    assert result["tool_result"]["status"] == "info", "Should return info status"
    assert "endpoint not configured" in result["plan"].lower()

def test_agent_error_handling():
    """Test agent handles connection errors gracefully."""
    agent = SimpleAgent({
        "calculator": "http://invalid-url-that-does-not-exist:9999/mcp/calculate"
    })
    
    result = agent.plan_and_execute("calculate 1 + 1")
    # Should not crash, should return error status
    assert result["tool_result"]["status"] == "error"

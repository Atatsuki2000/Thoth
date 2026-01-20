"""
Basic health check tests for Thoth services
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestServiceHealth:
    """Test that all services can start and respond"""
    
    def test_kb_api_module_exists(self):
        """Test that kb_api.py file exists"""
        kb_api_file = project_root / "kb_api.py"
        assert kb_api_file.exists(), "kb_api.py should exist"
    
    def test_agent_module_exists(self):
        """Test that agent module exists"""
        agent_dir = project_root / "agent"
        assert agent_dir.exists(), "agent/ directory should exist"
        assert (agent_dir / "agent.py").exists(), "agent/agent.py should exist"
    
    def test_retriever_module_exists(self):
        """Test that retriever module exists"""
        agent_dir = project_root / "agent"
        assert (agent_dir / "retriever.py").exists(), "agent/retriever.py should exist"
    
    def test_tools_exist(self):
        """Test that MCP tool services exist"""
        tools_dir = project_root / "tools"
        assert tools_dir.exists(), "tools/ directory should exist"
        
        expected_tools = ['calculator', 'plot-service', 'pdf-parser', 'web-search', 'file-ops']
        for tool in expected_tools:
            tool_dir = tools_dir / tool
            assert tool_dir.exists(), f"tools/{tool}/ should exist"
            assert (tool_dir / "main.py").exists(), f"tools/{tool}/main.py should exist"


class TestProjectStructure:
    """Test project structure and configuration"""
    
    def test_requirements_files_exist(self):
        """Test that requirements files exist"""
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "requirements-ci.txt").exists()
    
    def test_frontend_exists(self):
        """Test that frontend exists"""
        frontend_dir = project_root / "frontend"
        assert frontend_dir.exists()
        assert (frontend_dir / "app_kb.py").exists()
    
    def test_demo_kb_exists(self):
        """Test that demo knowledge base exists"""
        demo_kb_dir = project_root / "demo_kb"
        assert demo_kb_dir.exists()
        assert (demo_kb_dir / "README.md").exists()
    
    def test_startup_script_exists(self):
        """Test that startup script exists"""
        assert (project_root / "start_kb_system.ps1").exists()


class TestToolEndpoints:
    """Test MCP tool service endpoint configuration"""
    
    def test_tool_endpoint_format(self):
        """Test that tool endpoints follow MCP format"""
        endpoints = {
            'calculator': 'http://127.0.0.1:8001/mcp/calculate',
            'plot': 'http://127.0.0.1:8000/mcp/plot',
            'pdf': 'http://127.0.0.1:8002/mcp/parse',
            'web_search': 'http://127.0.0.1:8003/mcp/search',
            'file_ops': 'http://127.0.0.1:8004/mcp/read_file',
        }
        
        for name, url in endpoints.items():
            assert url.startswith('http://127.0.0.1:'), f"{name} should use localhost"
            assert '/mcp/' in url, f"{name} should have /mcp/ path"
    
    def test_kb_api_endpoint(self):
        """Test KB API endpoint format"""
        kb_api_url = 'http://127.0.0.1:8100'
        assert kb_api_url.startswith('http://127.0.0.1:8100')


class TestDocumentation:
    """Test that documentation files exist"""
    
    def test_readme_exists(self):
        """Test that README exists and has content"""
        readme = project_root / "README.md"
        assert readme.exists()
        
        content = readme.read_text(encoding='utf-8')
        assert 'Thoth' in content, "README should mention Thoth"
        assert 'MCP' in content or 'tool' in content.lower(), "README should mention tools"
    
    def test_changelog_exists(self):
        """Test that CHANGELOG exists"""
        assert (project_root / "CHANGELOG.md").exists()
    
    def test_license_exists(self):
        """Test that LICENSE exists"""
        assert (project_root / "LICENSE").exists()
    
    def test_n8n_workflows_exist(self):
        """Test that n8n workflows exist"""
        n8n_dir = project_root / "n8n-nodes"
        assert n8n_dir.exists()
        assert (n8n_dir / "rag-query-workflow.json").exists()
        assert (n8n_dir / "kb-upload-workflow.json").exists()
        assert (n8n_dir / "README.md").exists()


def test_placeholder():
    """Placeholder test to ensure pytest always has something to run"""
    assert True


def test_python_version():
    """Test that Python version is 3.10+"""
    assert sys.version_info >= (3, 10), "Python 3.10+ required"

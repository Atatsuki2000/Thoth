"""
Quick setup verification for benchmark script
"""
import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...\n")
    
    required = [
        ("langchain_huggingface", "pip install langchain-huggingface"),
        ("langchain_chroma", "pip install langchain-chroma"),
        ("chromadb", "pip install chromadb"),
        ("transformers", "pip install transformers torch accelerate"),
    ]
    
    missing = []
    
    for module, install_cmd in required:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - Missing")
            missing.append(install_cmd)
    
    if missing:
        print("\n⚠️  Missing dependencies! Install with:")
        for cmd in set(missing):
            print(f"  {cmd}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_services():
    """Check if MCP services are running"""
    import requests
    
    print("\n🔍 Checking MCP services...\n")
    
    services = {
        "plot-service": "http://127.0.0.1:8000",
        "calculator": "http://127.0.0.1:8001",
        "pdf-parser": "http://127.0.0.1:8002"
    }
    
    all_running = True
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            print(f"  ✅ {name}: {url}")
        except:
            print(f"  ❌ {name}: {url} (not running)")
            all_running = False
    
    if not all_running:
        print("\n⚠️  Some services are not running!")
        print("  Start with: .\\start_services.ps1")
        return False
    
    print("\n✅ All services are running!")
    return True

def check_vector_db():
    """Check if vector database is initialized"""
    print("\n🔍 Checking vector database...\n")
    
    db_path = os.path.join(os.path.dirname(__file__), "agent", "db", "chroma")
    
    if os.path.exists(db_path):
        print(f"  ✅ Vector DB found at: {db_path}")
        return True
    else:
        print(f"  ⚠️  Vector DB not found at: {db_path}")
        print("  The benchmark will attempt to create it")
        return True  # Not critical for benchmarks

if __name__ == "__main__":
    print("="*60)
    print("📋 Benchmark Setup Verification")
    print("="*60 + "\n")
    
    deps_ok = check_dependencies()
    
    if deps_ok:
        services_ok = check_services()
        db_ok = check_vector_db()
        
        print("\n" + "="*60)
        if deps_ok and services_ok:
            print("✅ Ready to run benchmarks!")
            print("\nRun: python benchmark.py --mode all --save")
        elif deps_ok:
            print("⚠️  Dependencies OK, but services not running")
            print("\nStart services first: .\\start_services.ps1")
            print("Then run: python benchmark.py --mode all --save")
        print("="*60 + "\n")

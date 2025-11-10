"""
Performance Benchmark Script for RAG+MCP Agent Framework

Measures:
- Retrieval precision@k (accuracy of relevant documents)
- Latency breakdown (retrieval, LLM, tool invocation, end-to-end)
- Tool success rate (percentage of successful MCP calls)
- Mode comparison (keyword vs local LLM vs OpenAI)

Usage:
    python benchmark.py --mode all
    python benchmark.py --mode retrieval
    python benchmark.py --mode agent
    python benchmark.py --mode comparison
"""

import sys
import os
import time
import json
import argparse
from typing import List, Dict, Tuple
from statistics import mean, stdev
import requests

# Add agent directory to path
agent_dir = os.path.join(os.path.dirname(__file__), 'agent')
if agent_dir not in sys.path:
    sys.path.append(agent_dir)

from agent import SimpleAgent
from retriever import get_top_k

# Test queries with expected tool and ground truth
TEST_QUERIES = [
    {
        "query": "Calculate 25 * 17 + 89",
        "expected_tool": "calculator",
        "ground_truth": 514,
        "category": "calculation"
    },
    {
        "query": "Show me a bar chart of sales data",
        "expected_tool": "plot",
        "ground_truth": None,
        "category": "visualization"
    },
    {
        "query": "Generate a scatter plot",
        "expected_tool": "plot",
        "ground_truth": None,
        "category": "visualization"
    },
    {
        "query": "What is 100 divided by 4?",
        "expected_tool": "calculator",
        "ground_truth": 25,
        "category": "calculation"
    },
    {
        "query": "Create a histogram visualization",
        "expected_tool": "plot",
        "ground_truth": None,
        "category": "visualization"
    },
    {
        "query": "Compute 2^10",
        "expected_tool": "calculator",
        "ground_truth": 1024,
        "category": "calculation"
    },
    {
        "query": "Draw a line graph",
        "expected_tool": "plot",
        "ground_truth": None,
        "category": "visualization"
    },
    {
        "query": "What is the square root of 144?",
        "expected_tool": "calculator",
        "ground_truth": 12,
        "category": "calculation"
    },
    {
        "query": "Plot random data",
        "expected_tool": "plot",
        "ground_truth": None,
        "category": "visualization"
    },
    {
        "query": "Calculate (5 + 3) * (10 - 2)",
        "expected_tool": "calculator",
        "ground_truth": 64,
        "category": "calculation"
    }
]

# Retrieval test queries with relevance labels
RETRIEVAL_QUERIES = [
    {
        "query": "machine learning algorithms",
        "relevant_docs": ["ml", "algorithm", "learning", "model", "training"]
    },
    {
        "query": "data visualization techniques",
        "relevant_docs": ["plot", "chart", "visual", "graph", "data"]
    },
    {
        "query": "python programming",
        "relevant_docs": ["python", "code", "program", "script", "function"]
    }
]


class BenchmarkRunner:
    def __init__(self):
        self.results = {
            "retrieval": {},
            "agent_keyword": {},
            "agent_local_llm": {},
            "agent_openai": {},
            "comparison": {}
        }
        
        # Default endpoints
        self.endpoints = {
            "plot": os.getenv('PLOT_SERVICE_URL', 'http://127.0.0.1:8000/mcp/plot'),
            "calculator": os.getenv('CALCULATOR_URL', 'http://127.0.0.1:8001/mcp/calculate'),
            "pdf": os.getenv('PDF_PARSER_URL', 'http://127.0.0.1:8002/mcp/parse')
        }
    
    def check_services(self) -> Dict[str, bool]:
        """Check if MCP services are running"""
        print("\n🔍 Checking MCP Services...")
        service_status = {}
        
        for name, url in self.endpoints.items():
            try:
                # Try to access the base URL
                base_url = url.rsplit('/', 1)[0] if '/' in url else url
                response = requests.get(base_url, timeout=2)
                service_status[name] = response.status_code in [200, 404]  # 404 is ok, means server is up
                status = "✅" if service_status[name] else "❌"
                print(f"  {status} {name}: {url}")
            except:
                service_status[name] = False
                print(f"  ❌ {name}: {url} (not reachable)")
        
        return service_status
    
    def benchmark_retrieval(self, k: int = 5) -> Dict:
        """Benchmark retrieval system"""
        print(f"\n📊 Benchmarking Retrieval (k={k})...")
        
        results = {
            "queries_tested": len(RETRIEVAL_QUERIES),
            "precision_at_k": [],
            "latencies_ms": [],
            "avg_precision": 0,
            "avg_latency_ms": 0
        }
        
        for test in RETRIEVAL_QUERIES:
            query = test["query"]
            relevant_terms = test["relevant_docs"]
            
            # Measure retrieval latency
            start = time.time()
            try:
                docs = get_top_k(query, k=k)
                latency_ms = (time.time() - start) * 1000
                results["latencies_ms"].append(latency_ms)
                
                # Calculate precision@k (simple keyword matching)
                relevant_found = 0
                for doc in docs:
                    content = doc.page_content.lower()
                    if any(term in content for term in relevant_terms):
                        relevant_found += 1
                
                precision = relevant_found / k if k > 0 else 0
                results["precision_at_k"].append(precision)
                
                print(f"  ✓ '{query[:40]}...' - Precision: {precision:.2f}, Latency: {latency_ms:.1f}ms")
            
            except Exception as e:
                print(f"  ✗ '{query[:40]}...' - Error: {e}")
                results["precision_at_k"].append(0)
                results["latencies_ms"].append(0)
        
        # Calculate averages
        if results["precision_at_k"]:
            results["avg_precision"] = mean(results["precision_at_k"])
            results["avg_latency_ms"] = mean(results["latencies_ms"])
            results["std_latency_ms"] = stdev(results["latencies_ms"]) if len(results["latencies_ms"]) > 1 else 0
        
        return results
    
    def benchmark_agent(self, mode: str = "keyword", api_key: str = None) -> Dict:
        """Benchmark agent with specified mode"""
        mode_name = {"keyword": "Keyword-based", "local": "Local LLM", "openai": "OpenAI GPT-3.5"}
        print(f"\n🤖 Benchmarking Agent ({mode_name.get(mode, mode)})...")
        
        # Initialize agent based on mode
        if mode == "keyword":
            agent = SimpleAgent(self.endpoints, use_llm=False)
        elif mode == "local":
            agent = SimpleAgent(self.endpoints, use_llm=True, llm_model="local")
        elif mode == "openai":
            agent = SimpleAgent(self.endpoints, use_llm=True, llm_model="openai", llm_api_key=api_key)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        results = {
            "mode": mode,
            "queries_tested": len(TEST_QUERIES),
            "tool_selection_accuracy": 0,
            "correct_selections": 0,
            "tool_success_rate": 0,
            "successful_calls": 0,
            "latencies": {
                "retrieval_ms": [],
                "llm_ms": [],
                "tool_ms": [],
                "end_to_end_ms": []
            },
            "errors": []
        }
        
        for test in TEST_QUERIES:
            query = test["query"]
            expected_tool = test["expected_tool"]
            
            print(f"\n  Testing: '{query}'")
            print(f"    Expected tool: {expected_tool}")
            
            # Measure end-to-end latency
            start_time = time.time()
            
            try:
                # Run agent
                plan_result = agent.plan_and_execute(query)
                
                end_to_end_ms = (time.time() - start_time) * 1000
                results["latencies"]["end_to_end_ms"].append(end_to_end_ms)
                
                # Extract timing information if available
                if "retrieval_time_ms" in plan_result:
                    results["latencies"]["retrieval_ms"].append(plan_result["retrieval_time_ms"])
                
                # Check tool selection accuracy
                selected_tool = plan_result.get("selected_tool", "none")
                if selected_tool == expected_tool:
                    results["correct_selections"] += 1
                    print(f"    ✓ Tool selection: {selected_tool} (correct)")
                else:
                    print(f"    ✗ Tool selection: {selected_tool} (expected {expected_tool})")
                
                # Check tool execution success
                if plan_result.get("tool_result") and plan_result["tool_result"].get("status") == "success":
                    results["successful_calls"] += 1
                    tool_time = plan_result["tool_result"].get("metadata", {}).get("tool_time_ms", 0)
                    results["latencies"]["tool_ms"].append(tool_time)
                    print(f"    ✓ Tool execution: success ({tool_time:.1f}ms)")
                else:
                    print(f"    ✗ Tool execution: failed")
                
                print(f"    ⏱️  End-to-end: {end_to_end_ms:.1f}ms")
            
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results["errors"].append({"query": query, "error": str(e)})
                results["latencies"]["end_to_end_ms"].append(0)
        
        # Calculate metrics
        results["tool_selection_accuracy"] = results["correct_selections"] / results["queries_tested"] * 100
        results["tool_success_rate"] = results["successful_calls"] / results["queries_tested"] * 100
        
        # Calculate average latencies
        for key in results["latencies"]:
            if results["latencies"][key]:
                values = [v for v in results["latencies"][key] if v > 0]
                if values:
                    results[f"avg_{key}"] = mean(values)
                    results[f"std_{key}"] = stdev(values) if len(values) > 1 else 0
        
        return results
    
    def run_comparison(self, api_key: str = None) -> Dict:
        """Compare all three modes"""
        print("\n" + "="*60)
        print("🔬 Running Mode Comparison")
        print("="*60)
        
        comparison = {
            "keyword": self.benchmark_agent("keyword"),
            "local_llm": self.benchmark_agent("local"),
        }
        
        if api_key:
            comparison["openai"] = self.benchmark_agent("openai", api_key)
        
        return comparison
    
    def print_summary(self, results: Dict):
        """Print formatted benchmark summary"""
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        if "retrieval" in results and results["retrieval"]:
            print("\n🔍 Retrieval Performance:")
            r = results["retrieval"]
            print(f"  • Average Precision@k: {r['avg_precision']:.2%}")
            print(f"  • Average Latency: {r['avg_latency_ms']:.1f}ms (±{r.get('std_latency_ms', 0):.1f}ms)")
            print(f"  • Queries Tested: {r['queries_tested']}")
        
        if "comparison" in results and results["comparison"]:
            print("\n🤖 Agent Mode Comparison:")
            
            modes = ["keyword", "local_llm", "openai"]
            mode_names = {"keyword": "Keyword-based", "local_llm": "Local LLM", "openai": "OpenAI GPT-3.5"}
            
            print("\n  Tool Selection Accuracy:")
            for mode in modes:
                if mode in results["comparison"]:
                    acc = results["comparison"][mode]["tool_selection_accuracy"]
                    print(f"    • {mode_names[mode]}: {acc:.1f}%")
            
            print("\n  Tool Success Rate:")
            for mode in modes:
                if mode in results["comparison"]:
                    rate = results["comparison"][mode]["tool_success_rate"]
                    print(f"    • {mode_names[mode]}: {rate:.1f}%")
            
            print("\n  Average End-to-End Latency:")
            for mode in modes:
                if mode in results["comparison"]:
                    latency = results["comparison"][mode].get("avg_end_to_end_ms", 0)
                    std = results["comparison"][mode].get("std_end_to_end_ms", 0)
                    print(f"    • {mode_names[mode]}: {latency:.1f}ms (±{std:.1f}ms)")
        
        print("\n" + "="*60)
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save results to JSON file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG+MCP Agent Framework")
    parser.add_argument(
        "--mode",
        choices=["all", "retrieval", "agent", "comparison"],
        default="all",
        help="Benchmark mode to run"
    )
    parser.add_argument(
        "--agent-mode",
        choices=["keyword", "local", "openai"],
        default="local",
        help="Agent mode for single agent benchmark"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key for OpenAI mode benchmarking"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Check services
    service_status = runner.check_services()
    if not any(service_status.values()):
        print("\n⚠️  Warning: No MCP services are running!")
        print("   Start services with: .\\start_services.ps1")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run benchmarks based on mode
    all_results = {}
    
    if args.mode in ["all", "retrieval"]:
        all_results["retrieval"] = runner.benchmark_retrieval()
    
    if args.mode == "agent":
        all_results["agent"] = runner.benchmark_agent(args.agent_mode, args.openai_api_key)
    
    if args.mode in ["all", "comparison"]:
        all_results["comparison"] = runner.run_comparison(args.openai_api_key)
    
    # Print summary
    runner.print_summary(all_results)
    
    # Save results if requested
    if args.save:
        runner.save_results(all_results)
    
    print("\n✅ Benchmark complete!\n")


if __name__ == "__main__":
    main()

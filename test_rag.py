"""
Comprehensive Testing Script for RAG System
Tests all functionality locally and on Railway
"""

import requests
import time
from typing import List, Dict
import json

# Test queries covering different aspects
TEST_QUERIES = [
    {
        "query": "What are the required chapters of a Development FYP report?",
        "category": "Structure",
        "expected_keywords": ["chapter", "development", "report"]
    },
    {
        "query": "What headings, fonts, and sizes are required?",
        "category": "Formatting",
        "expected_keywords": ["font", "heading", "size"]
    },
    {
        "query": "What margins and spacing do we use?",
        "category": "Formatting",
        "expected_keywords": ["margin", "spacing"]
    },
    {
        "query": "How to use Ibid. and op. cit. in citations?",
        "category": "Citations",
        "expected_keywords": ["ibid", "citation"]
    },
    {
        "query": "What is the evaluation criteria for FYP?",
        "category": "Evaluation",
        "expected_keywords": ["evaluation", "criteria"]
    },
    {
        "query": "What goes into the Executive Summary?",
        "category": "Content",
        "expected_keywords": ["executive", "summary"]
    },
    {
        "query": "What are the submission deadlines?",
        "category": "Timeline",
        "expected_keywords": ["deadline", "submission"]
    }
]

# Edge case queries
EDGE_CASES = [
    {
        "query": "",
        "category": "Empty Query",
        "expected_behavior": "Should handle gracefully"
    },
    {
        "query": "What is quantum physics?",
        "category": "Out of Scope",
        "expected_behavior": "Should say not in handbook"
    },
    {
        "query": "a" * 500,
        "category": "Very Long Query",
        "expected_behavior": "Should truncate or handle"
    }
]


class RAGTester:
    def __init__(self, base_url: str = None, local: bool = True):
        """
        Initialize tester
        
        Args:
            base_url: Railway URL (e.g., "https://your-app.up.railway.app")
            local: If True, test local CLI; if False, test Railway deployment
        """
        self.base_url = base_url
        self.local = local
        self.results = []
        
    def test_local_cli(self, query: str) -> Dict:
        """Test local ask.py CLI"""
        import subprocess
        try:
            start_time = time.time()
            result = subprocess.run(
                ['python', 'ask.py', query],
                capture_output=True,
                text=True,
                timeout=30
            )
            elapsed = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "time": elapsed
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "time": 0
            }
    
    def test_railway_endpoint(self, query: str) -> Dict:
        """Test Railway deployed app (simulate Streamlit interaction)"""
        try:
            start_time = time.time()
            # Note: Streamlit doesn't expose REST API by default
            # This would require adding an API endpoint or using Streamlit's internal API
            # For now, we check if the app is accessible
            response = requests.get(self.base_url, timeout=10)
            elapsed = time.time() - start_time
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "time": elapsed,
                "note": "Manual testing required via web UI"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "time": 0
            }
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        query = test_case["query"]
        category = test_case["category"]
        
        print(f"\n{'='*60}")
        print(f"📝 Testing: {category}")
        print(f"Query: {query[:100]}...")
        print(f"{'='*60}")
        
        if self.local:
            result = self.test_local_cli(query)
        else:
            result = self.test_railway_endpoint(query)
        
        # Analyze result
        analysis = {
            "query": query,
            "category": category,
            "success": result["success"],
            "time": result["time"],
            "result": result
        }
        
        # Check for expected keywords (if applicable)
        if "expected_keywords" in test_case and result.get("output"):
            output_lower = result["output"].lower()
            found_keywords = [
                kw for kw in test_case["expected_keywords"]
                if kw.lower() in output_lower
            ]
            analysis["keywords_found"] = found_keywords
            analysis["keyword_match"] = len(found_keywords) / len(test_case["expected_keywords"])
        
        # Print results
        if result["success"]:
            print(f"✅ SUCCESS ({result['time']:.2f}s)")
            if "output" in result:
                print(f"\nAnswer preview:")
                print(result["output"][:200] + "..." if len(result["output"]) > 200 else result["output"])
        else:
            print(f"❌ FAILED")
            if "error" in result:
                print(f"Error: {result['error']}")
        
        self.results.append(analysis)
        return analysis
    
    def run_all_tests(self):
        """Run all test cases"""
        print("\n" + "="*70)
        print("🧪 RAG SYSTEM TEST SUITE")
        print("="*70)
        print(f"Mode: {'LOCAL CLI' if self.local else 'RAILWAY DEPLOYMENT'}")
        if not self.local:
            print(f"URL: {self.base_url}")
        print()
        
        # Run standard tests
        print("\n📋 STANDARD TEST CASES")
        print("-"*70)
        for test in TEST_QUERIES:
            self.run_single_test(test)
            time.sleep(1)  # Avoid rate limiting
        
        # Run edge cases
        print("\n\n⚠️ EDGE CASE TESTS")
        print("-"*70)
        for test in EDGE_CASES:
            self.run_single_test(test)
            time.sleep(1)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Average response time
        avg_time = sum(r["time"] for r in self.results) / total
        print(f"⏱️ Average Response Time: {avg_time:.2f}s")
        
        # Keyword matching (if applicable)
        keyword_results = [r for r in self.results if "keyword_match" in r]
        if keyword_results:
            avg_match = sum(r["keyword_match"] for r in keyword_results) / len(keyword_results)
            print(f"🎯 Average Keyword Match: {avg_match*100:.1f}%")
        
        # Failed tests detail
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for r in self.results:
                if not r["success"]:
                    print(f"  - {r['category']}: {r['result'].get('error', 'Unknown error')}")
        
        print("\n" + "="*70)
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("📄 Detailed results saved to: test_results.json")


def test_local():
    """Test local CLI implementation"""
    tester = RAGTester(local=True)
    tester.run_all_tests()


def test_railway(url: str):
    """Test Railway deployment"""
    tester = RAGTester(base_url=url, local=False)
    
    # First, check if app is accessible
    print(f"🔍 Checking Railway deployment at: {url}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ Railway app is accessible!")
            print(f"\n⚠️ Note: Streamlit apps require manual testing via web UI")
            print(f"Please test the following queries manually at: {url}\n")
            
            for i, test in enumerate(TEST_QUERIES + EDGE_CASES, 1):
                print(f"{i}. {test['query']}")
                if 'expected_keywords' in test:
                    print(f"   Expected: {', '.join(test['expected_keywords'])}")
                print()
        else:
            print(f"❌ Railway app returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to Railway: {e}")


if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║          RAG SYSTEM COMPREHENSIVE TEST SUITE                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Choose test mode
    print("Select test mode:")
    print("1. Test Local CLI (ask.py)")
    print("2. Test Railway Deployment")
    print("3. Run Both")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🖥️ Testing Local CLI...")
        test_local()
    
    elif choice == "2":
        if len(sys.argv) > 2:
            url = sys.argv[2]
        else:
            url = input("Enter Railway URL (e.g., https://your-app.up.railway.app): ").strip()
        
        print(f"\n☁️ Testing Railway at {url}...")
        test_railway(url)
    
    elif choice == "3":
        print("\n🖥️ Testing Local CLI...")
        test_local()
        
        if len(sys.argv) > 2:
            url = sys.argv[2]
        else:
            url = input("\nEnter Railway URL (e.g., https://your-app.up.railway.app): ").strip()
        
        print(f"\n☁️ Testing Railway at {url}...")
        test_railway(url)
    
    else:
        print("Invalid choice!")
        sys.exit(1)

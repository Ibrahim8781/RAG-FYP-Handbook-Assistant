"""
Setup Validation Script
Checks if all components are properly configured
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    required = ['groq', 'dotenv', 'faiss', 'streamlit', 'sentence_transformers']
    missing = []
    
    for package in required:
        try:
            if package == 'dotenv':
                import dotenv
            elif package == 'faiss':
                import faiss
            elif package == 'streamlit':
                import streamlit
            elif package == 'sentence_transformers':
                import sentence_transformers
            elif package == 'groq':
                import groq
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed\n")
    return True


def check_env_file():
    """Check if .env file exists and has API key"""
    print("Checking .env configuration...")
    
    if not os.path.exists('.env'):
        print("  ✗ .env file not found")
        print("\n❌ Please create .env file:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Groq API key")
        return False
    
    print("  ✓ .env file exists")
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY', '')
    
    if not api_key or api_key == 'your_groq_api_key_here':
        print("  ✗ GROQ_API_KEY not configured")
        print("\n❌ Please add your API key to .env file:")
        print("   GROQ_API_KEY=your_actual_key")
        return False
    
    print("  ✓ GROQ_API_KEY configured")
    
    # Test API key
    print("  Testing API connection...")
    try:
        from llm_utils import validate_api_key
        if validate_api_key(api_key):
            print("  ✓ API key is valid")
        else:
            print("  ✗ API key appears invalid")
            print("\n❌ Please check your API key at https://console.groq.com/keys")
            return False
    except Exception as e:
        print(f"  ✗ Error testing API key: {e}")
        return False
    
    print("✅ Environment configured correctly\n")
    return True


def check_rag_files():
    """Check if RAG system files exist"""
    print("Checking RAG system files...")
    
    required_files = {
        'FYP-Handbook-2023.pdf': 'Source PDF',
        'faiss_index.bin': 'FAISS index',
        'chunks_metadata.pkl': 'Chunk metadata',
        'config.json': 'Configuration'
    }
    
    pdf_exists = os.path.exists('FYP-Handbook-2023.pdf')
    rag_files_exist = all(os.path.exists(f) for f in ['faiss_index.bin', 'chunks_metadata.pkl', 'config.json'])
    
    if not pdf_exists:
        print("  ✗ FYP-Handbook-2023.pdf not found")
        print("\n❌ Please add the handbook PDF to the project directory")
        return False
    
    print("  ✓ FYP-Handbook-2023.pdf found")
    
    if not rag_files_exist:
        print("  ✗ RAG index files not found")
        print("\n⚠️  Please run: python ingest.py")
        return False
    
    print("  ✓ FAISS index files found")
    
    print("✅ All RAG files present\n")
    return True


def check_llm_utils():
    """Check if llm_utils.py is present and functional"""
    print("Checking LLM integration...")
    
    if not os.path.exists('llm_utils.py'):
        print("  ✗ llm_utils.py not found")
        return False
    
    print("  ✓ llm_utils.py exists")
    
    try:
        from llm_utils import GroqLLM, format_context_for_llm
        print("  ✓ LLM utilities importable")
    except Exception as e:
        print(f"  ✗ Error importing llm_utils: {e}")
        return False
    
    print("✅ LLM integration ready\n")
    return True


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("RAG Setup Validation")
    print("=" * 60)
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Environment", check_env_file),
        ("RAG Files", check_rag_files),
        ("LLM Integration", check_llm_utils)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Error in {name}: {e}\n")
            results.append(False)
    
    print("=" * 60)
    if all(results):
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print()
        print("Your RAG system is ready to use!")
        print()
        print("To run:")
        print("  • Web UI:  streamlit run app.py")
        print("  • CLI:     python ask.py")
        print()
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print()
        print("Please fix the issues above and run this script again.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
tests/test_imports.py
---------------------
Kiểm tra xem tất cả các module có thể import thành công mà không bị lỗi cú pháp.
"""

import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    print("Testing imports...")
    
    try:
        from src.rag.parser import TextChunk, DocumentParser
        print("✅ Import Parser OK")
    except Exception as e:
        print(f"❌ Import Parser FAILED: {e}")
        return False
        
    try:
        from src.rag.embedder import Embedder
        print("✅ Import Embedder OK")
    except Exception as e:
        print(f"❌ Import Embedder FAILED: {e}")
        return False
        
    try:
        from src.rag.vector_store import VectorStore, SearchResult
        print("✅ Import VectorStore OK")
    except Exception as e:
        print(f"❌ Import VectorStore FAILED: {e}")
        return False
        
    try:
        from src.rag.rag_pipeline import RAGPipeline
        print("✅ Import RAGPipeline OK")
    except Exception as e:
        print(f"❌ Import RAGPipeline FAILED: {e}")
        return False
        
    try:
        from src.agent.mcp_client import get_mcp_tools
        print("✅ Import MCP Client OK")
    except Exception as e:
        print(f"❌ Import MCP Client FAILED: {e}")
        return False
        
    try:
        from src.agent.graph import agent_executor
        print("✅ Import Agent Graph OK")
    except Exception as e:
        print(f"❌ Import Agent Graph FAILED: {e}")
        return False
        
    try:
        from src.api.main import app
        print("✅ Import FastAPI App OK")
    except Exception as e:
        print(f"❌ Import FastAPI App FAILED: {e}")
        return False
        
    print("\n🎉 Tất cả các module đã được import thành công mà không có lỗi cú pháp!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

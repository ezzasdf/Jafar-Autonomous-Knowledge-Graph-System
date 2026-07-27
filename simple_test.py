#!/usr/bin/env python3
"""
Simple Test for Offline AI Book Reader
Tests basic functionality without heavy dependencies
"""

import sys
import os
from pathlib import Path
import tempfile

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports that should work"""
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import sqlite3
        print("✅ SQLite3")
    except ImportError as e:
        print(f"❌ SQLite3: {e}")
        return False
    
    try:
        from pathlib import Path
        print("✅ Pathlib")
    except ImportError as e:
        print(f"❌ Pathlib: {e}")
        return False
    
    try:
        import json
        print("✅ JSON")
    except ImportError as e:
        print(f"❌ JSON: {e}")
        return False
    
    try:
        import logging
        print("✅ Logging")
    except ImportError as e:
        print(f"❌ Logging: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import get_config, USER_BOOKS_DIR
        config = get_config()
        print("✅ Configuration loaded")
        print(f"   Books directory: {USER_BOOKS_DIR}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n🧪 Testing utilities...")
    
    try:
        from utils import get_system_info, is_low_end_device, format_file_size
        system_info = get_system_info()
        is_low_end = is_low_end_device()
        formatted_size = format_file_size(1024)
        
        print("✅ Utilities working")
        print(f"   System: {system_info.get('cpu', {}).get('count', 'Unknown')} CPUs")
        print(f"   Low-end device: {is_low_end}")
        print(f"   Format test: {formatted_size}")
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False

def test_memory_manager():
    """Test memory manager"""
    print("\n🧪 Testing memory manager...")
    
    try:
        from memory_manager import MemoryManager
        memory_manager = MemoryManager()
        stats = memory_manager.get_memory_stats()
        
        print("✅ Memory manager working")
        print(f"   Memory usage: {stats.get('current_memory_mb', 0):.1f}MB")
        
        # Test compression
        test_text = "This is a test text for compression."
        compressed = memory_manager.compress_text(test_text)
        decompressed = memory_manager.decompress_text(compressed)
        
        if test_text == decompressed:
            print("✅ Text compression working")
        else:
            print("❌ Text compression failed")
            return False
        
        memory_manager.cleanup()
        return True
    except Exception as e:
        print(f"❌ Memory manager error: {e}")
        return False

def test_vector_db_basic():
    """Test vector database basic functionality"""
    print("\n🧪 Testing vector database...")
    
    try:
        from memory_manager import MemoryManager
        from vector_db import VectorDatabase
        
        memory_manager = MemoryManager()
        vector_db = VectorDatabase(memory_manager)
        
        # Test basic operations
        stats = vector_db.get_database_stats()
        print("✅ Vector database initialized")
        print(f"   Books in database: {stats.get('book_count', 0)}")
        
        vector_db.cleanup()
        memory_manager.cleanup()
        return True
    except Exception as e:
        print(f"❌ Vector database error: {e}")
        return False

def test_book_processor_basic():
    """Test book processor basic functionality"""
    print("\n🧪 Testing book processor...")
    
    try:
        from memory_manager import MemoryManager
        from book_processor import BookProcessor
        
        memory_manager = MemoryManager()
        book_processor = BookProcessor(memory_manager)
        
        # Test file type detection
        test_file = Path("test.txt")
        file_type = book_processor._get_file_type(test_file)
        
        print("✅ Book processor initialized")
        print(f"   Test file type: {file_type}")
        
        memory_manager.cleanup()
        return True
    except Exception as e:
        print(f"❌ Book processor error: {e}")
        return False

def test_text_processing():
    """Test text processing functionality"""
    print("\n🧪 Testing text processing...")
    
    try:
        from memory_manager import MemoryManager
        from book_processor import BookProcessor
        
        memory_manager = MemoryManager()
        book_processor = BookProcessor(memory_manager)
        
        # Test text cleaning and chunking
        test_text = "This is a test text. It has multiple sentences. We will test cleaning and chunking."
        cleaned_text = book_processor._clean_text(test_text)
        chunks = book_processor._chunk_text(cleaned_text, 1)
        
        print("✅ Text processing working")
        print(f"   Original length: {len(test_text)}")
        print(f"   Cleaned length: {len(cleaned_text)}")
        print(f"   Number of chunks: {len(chunks)}")
        
        memory_manager.cleanup()
        return True
    except Exception as e:
        print(f"❌ Text processing error: {e}")
        return False

def test_optional_dependencies():
    """Test optional dependencies"""
    print("\n🧪 Testing optional dependencies...")
    
    optional_deps = {
        "sentence_transformers": "Text embeddings",
        "torch": "PyTorch",
        "faiss": "Vector search",
        "fitz": "PDF processing (PyMuPDF)",
        "docx": "Word document processing",
        "cv2": "Image processing (OpenCV)",
        "easyocr": "OCR processing"
    }
    
    working_deps = 0
    for module, description in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {description} ({module})")
            working_deps += 1
        except ImportError:
            print(f"⚠️  {description} ({module}) - not installed")
    
    print(f"\n📊 Optional dependencies: {working_deps}/{len(optional_deps)} available")
    return True

def main():
    """Run all tests"""
    print("🚀 Simple Test for Offline AI Book Reader")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_config),
        ("Utilities", test_utils),
        ("Memory Manager", test_memory_manager),
        ("Vector Database", test_vector_db_basic),
        ("Book Processor", test_book_processor_basic),
        ("Text Processing", test_text_processing),
        ("Optional Dependencies", test_optional_dependencies),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n💡 Next steps:")
        print("1. Add books to the 'books' folder")
        print("2. Run: python main.py scan-books")
        print("3. Run: python main.py process-all")
        print("4. Start chatting: python main.py chat")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 To fix missing dependencies:")
        print("python install.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
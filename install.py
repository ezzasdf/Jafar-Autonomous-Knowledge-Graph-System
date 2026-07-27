#!/usr/bin/env python3
"""
Installation script for Offline AI Book Reader
Helps install all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install all required dependencies"""
    print("🚀 Installing Offline AI Book Reader dependencies...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Failed to upgrade pip, continuing anyway...")
    
    # Install core dependencies
    dependencies = [
        ("sentence-transformers==2.2.2", "Sentence Transformers"),
        ("transformers==4.35.0", "Transformers"),
        ("torch==2.1.0", "PyTorch"),
        ("torchvision==0.16.0", "TorchVision"),
        ("numpy==1.24.3", "NumPy"),
        ("faiss-cpu==1.7.4", "FAISS CPU"),
        ("tqdm==4.66.1", "TQDM"),
        ("click==8.1.7", "Click"),
        ("rich==13.6.0", "Rich"),
        ("typing-extensions==4.8.0", "Typing Extensions"),
        ("psutil==5.9.6", "PSUtil"),
    ]
    
    for dep, name in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {name}"):
            return False
    
    # Install document processing dependencies
    doc_deps = [
        ("PyPDF2==3.0.1", "PyPDF2"),
        ("python-docx==0.8.11", "Python DOCX"),
        ("PyMuPDF==1.23.8", "PyMuPDF (fitz)"),
    ]
    
    for dep, name in doc_deps:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {name}"):
            print(f"⚠️  Failed to install {name}, document processing may be limited")
    
    # Install image processing dependencies
    image_deps = [
        ("Pillow==10.0.1", "Pillow (PIL)"),
        ("opencv-python==4.8.1.78", "OpenCV"),
        ("easyocr==1.7.0", "EasyOCR"),
        ("pytesseract==0.3.10", "PyTesseract"),
    ]
    
    for dep, name in image_deps:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {name}"):
            print(f"⚠️  Failed to install {name}, image processing may be limited")
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\n📋 Next steps:")
    print("1. Test the installation: python test_system.py")
    print("2. Add books to the 'books' folder")
    print("3. Run: python main.py scan-books")
    print("4. Run: python main.py process-all")
    print("5. Start chatting: python main.py chat")
    
    return True

def verify_installation():
    """Verify that all dependencies are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers")
    except ImportError:
        print("❌ Sentence Transformers")
        return False
    
    try:
        import torch
        print("✅ PyTorch")
    except ImportError:
        print("❌ PyTorch")
        return False
    
    try:
        import numpy
        print("✅ NumPy")
    except ImportError:
        print("❌ NumPy")
        return False
    
    try:
        import faiss
        print("✅ FAISS")
    except ImportError:
        print("⚠️  FAISS - Vector search will use simple similarity")
        print("   This is normal on Windows (requires SWIG)")
    
    try:
        import fitz
        print("✅ PyMuPDF (fitz)")
    except ImportError:
        print("⚠️  PyMuPDF (fitz) - PDF processing will be limited")
    
    try:
        import docx
        print("✅ Python DOCX")
    except ImportError:
        print("⚠️  Python DOCX - DOCX processing will be limited")
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError:
        print("⚠️  OpenCV - Image processing will be limited")
    
    try:
        import easyocr
        print("✅ EasyOCR")
    except ImportError:
        print("⚠️  EasyOCR - OCR processing will be limited")
    
    print("\n✅ Core dependencies verified!")
    return True

def main():
    """Main installation function"""
    print("🤖 Offline AI Book Reader - Installation Script")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some dependencies failed to install. The system may have limited functionality.")
    
    print("\n🎉 Installation script completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
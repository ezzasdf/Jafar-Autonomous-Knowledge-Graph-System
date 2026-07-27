#!/usr/bin/env python3
"""
Quick Fix Script for Offline AI Book Reader
Installs the most commonly missing dependencies
"""

import subprocess
import sys
import os

def install_package(package, description):
    """Install a single package"""
    print(f"🔄 Installing {description}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      check=True, capture_output=True, text=True)
        print(f"✅ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {description}: {e}")
        return False

def main():
    """Install commonly missing dependencies"""
    print("🔧 Quick Fix for Offline AI Book Reader")
    print("=" * 40)
    
    # Most commonly missing packages
    packages = [
        ("sentence-transformers==2.2.2", "Sentence Transformers (text embeddings)"),
        ("torch==2.2.1", "PyTorch"),  # Updated to match your installed version
        ("PyPDF2==3.0.1", "PyPDF2 (PDF processing)"),
        ("PyMuPDF==1.23.8", "PyMuPDF (PDF processing)"),
        ("python-docx==0.8.11", "Python DOCX (Word documents)"),
        ("opencv-python==4.8.1.78", "OpenCV (image processing)"),
        ("easyocr==1.7.0", "EasyOCR (text extraction from images)"),
        ("Pillow==10.0.1", "Pillow (image handling)"),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package, description in packages:
        if install_package(package, description):
            success_count += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {success_count}/{total_count} packages installed successfully")
    
    if success_count == total_count:
        print("🎉 All packages installed! The system should work now.")
    else:
        print("⚠️  Some packages failed to install. The system may have limited functionality.")
    
    print("\n💡 Next steps:")
    print("1. Test the system: python simple_test.py")
    print("2. If successful, try: python test_system.py")
    print("3. Start using: python main.py scan-books")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
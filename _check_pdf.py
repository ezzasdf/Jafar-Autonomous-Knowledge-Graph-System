import sys
sys.path.insert(0, '.')
try:
    import PyPDF2
    print(f"PyPDF2: ok")
except Exception as e:
    print(f"PyPDF2: NOT available ({e})")
try:
    import fitz
    print("fitz: ok")
except Exception as e:
    print(f"fitz: NOT available ({e})")
try:
    import pdfminer
    print("pdfminer: ok")
except:
    print("pdfminer: NOT available")
try:
    import pdfplumber
    print("pdfplumber: ok")
except:
    print("pdfplumber: NOT available")

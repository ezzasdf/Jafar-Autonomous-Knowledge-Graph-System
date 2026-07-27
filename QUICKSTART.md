# Quick Start Guide - Offline AI Book Reader

Get up and running with the AI book reading system in minutes!

## 🚀 Installation

### Option 1: Automatic Installation (Recommended)
```bash
python install.py
```

### Option 2: Manual Installation
1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the system:**
   ```bash
   # Simple test (recommended first)
   python simple_test.py
   
   # Full test (requires all dependencies)
   python test_system.py
   ```

### Troubleshooting Import Errors
If you get import errors like "could not be resolved", try:

**Option 1: Quick Fix (Recommended)**
```bash
python quick_fix.py
```

**Option 2: Manual Fix**
```bash
# Install specific packages that might be missing:
pip install sentence-transformers==2.2.2  # For text embeddings
pip install torch==2.1.0                  # For PyTorch
pip install faiss-cpu==1.7.4             # For vector search
pip install PyMuPDF==1.23.8              # For 'fitz' module
pip install python-docx==0.8.11          # For 'docx' module
pip install opencv-python==4.8.1.78      # For 'cv2' module
pip install easyocr==1.7.0               # For OCR functionality
pip install Pillow==10.0.1               # For image handling
```

**Option 3: Minimal Installation**
```bash
# Install only core dependencies:
pip install -r requirements_minimal.txt
```

## 📚 Basic Usage

### 1. Add Books (Two Ways)

**Option A: Individual books**
```bash
# Add a PDF book
python main.py add-book "path/to/your/book.pdf"

# Add a text file
python main.py add-book "path/to/your/document.txt"

# Add with custom title and author
python main.py add-book "book.pdf" --title "My Book" --author "John Doe"
```

**Option B: Books folder (Recommended for multiple books)**
```bash
# Copy your books to the 'books' folder, then:
python main.py scan-books          # See what's available
python main.py process-all         # Process all books at once

# Or add a book from elsewhere to the folder:
python main.py add-to-folder "path/to/book.pdf"
```

### 2. Ask Questions
```bash
# Ask about any book in your library
python main.py ask "What is the main theme of the book?"

# Ask about a specific book
python main.py ask "What is the main theme?" --book-id "your_book_id"

# Ask with custom context length
python main.py ask "Explain the plot" --max-context 2000
```

### 3. Interactive Chat
```bash
# Start interactive mode
python main.py chat
```

### 4. Manage Your Library
```bash
# List all processed books
python main.py list-books

# Scan books folder
python main.py scan-books

# Delete a book
python main.py delete-book "book_id"

# View system statistics
python main.py stats
```

## 🧠 Memory Optimization

For low-end devices, the system automatically optimizes itself, but you can also manually optimize:

```bash
python main.py optimize
```

## 📁 Supported File Formats

- **PDF** (.pdf) - Full text and image extraction
- **Word Documents** (.docx, .doc) - Text extraction
- **Text Files** (.txt, .md, .rtf) - Direct text processing
- **Images** (.jpg, .png, .bmp, .tiff) - OCR text extraction

## 🔧 Configuration

Edit `config.py` to customize:
- Memory limits
- Embedding model size
- Processing parameters
- Database settings

## 📊 Performance Tips

1. **For large books:** Process them in smaller chunks
2. **For low memory:** Use the optimize command
3. **For faster processing:** Reduce batch sizes in config
4. **For better accuracy:** Increase similarity thresholds

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named 'sentence_transformers'"**
   ```bash
   pip install sentence-transformers
   ```

2. **"Memory error"**
   ```bash
   python main.py optimize
   ```

3. **"Database locked"**
   - Close any other instances of the application
   - Restart the system

4. **"OCR not working"**
   ```bash
   pip install easyocr
   ```

### Getting Help:

- Check the logs in `data/app.log`
- Run `python main.py stats` for system information
- Use `python test_system.py` to verify installation

## 🎯 Example Workflow

```bash
# 1. Test the system
python test_system.py

# 2. Add books to the books folder
# Copy your PDF/DOCX files to the 'books' folder

# 3. Scan and process books
python main.py scan-books
python main.py process-all

# 4. Start chatting
python main.py chat

# 5. Ask specific questions
python main.py ask "What are the key concepts?"

# 6. Check your library
python main.py list-books
```

## 📈 Advanced Usage

### Programmatic Usage:
```python
from memory_manager import MemoryManager
from embeddings import EmbeddingGenerator
from vector_db import VectorDatabase
from book_processor import BookProcessor
from qa_system import QuestionAnsweringSystem

# Initialize components
memory_manager = MemoryManager()
embedding_generator = EmbeddingGenerator(memory_manager)
vector_db = VectorDatabase(memory_manager)
book_processor = BookProcessor(memory_manager)
qa_system = QuestionAnsweringSystem(embedding_generator, vector_db, memory_manager)

# Process a book
book_data = book_processor.process_book("book.pdf")

# Ask questions
answer = qa_system.answer_question("What is this book about?")
print(answer["answer"])
```

### Custom Embeddings:
```python
# Generate embeddings for custom text
texts = ["Your text here", "Another text"]
embeddings = embedding_generator.generate_text_embeddings(texts)

# Find similar texts
similar = embedding_generator.find_most_similar(embeddings[0], embeddings)
```

## 🔒 Privacy & Security

- **Fully Offline:** No data leaves your device
- **Local Storage:** All data stored locally in `data/` directory
- **No Internet Required:** Works completely offline
- **Compressed Storage:** Efficient memory usage

## 📞 Support

If you encounter issues:
1. Check the logs in `data/app.log`
2. Run the test suite: `python test_system.py`
3. Verify your Python version (3.7+ required)
4. Ensure all dependencies are installed

---

**Happy Reading! 📚🤖** 
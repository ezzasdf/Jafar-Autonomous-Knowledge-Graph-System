"""
Book Processor for Offline AI Book Reader
Handles extraction of text and images from various document formats
"""

import sys
import os
site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site-packages')
if os.path.isdir(site_packages):
    sys.path.insert(0, site_packages)

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

try:
    import fitz
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import hashlib
import re
from tqdm import tqdm

from config import PROCESSING_CONFIG, SUPPORTED_FORMATS
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class BookProcessor:
    """Process books and extract text and images"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.config = PROCESSING_CONFIG
        self.memory_manager = memory_manager or MemoryManager()
        
        logger.info("Book Processor initialized")
    
    def process_book(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a book file and extract all content"""
        file_path = Path(file_path)
        logger.debug("process_book called: %s", file_path.name)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Book file not found: {file_path}")
        
        # Determine file type
        file_type = self._get_file_type(file_path)
        if not file_type:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        logger.info(f"Processing {file_type} file: {file_path.name}")
        
        # Extract content based on file type
        if file_type == "pdf":
            return self._process_pdf(file_path)
        elif file_type == "docx":
            return self._process_docx(file_path)
        elif file_type == "text":
            return self._process_text(file_path)
        elif file_type == "images":
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _get_file_type(self, file_path: Path) -> Optional[str]:
        """Determine the file type based on extension"""
        extension = file_path.suffix.lower()
        
        for file_type, extensions in SUPPORTED_FORMATS.items():
            if extension in extensions:
                return file_type
        
        return None
    
    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF file and extract text and images"""
        logger.debug("_process_pdf: %s", file_path.name)
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed. Please install it with: pip install PyMuPDF")
        
        try:
            # Open PDF with PyMuPDF for better text extraction
            pdf_document = fitz.open(file_path)
            
            pages = []
            images = []
            total_pages = len(pdf_document)
            
            logger.info(f"Processing PDF with {total_pages} pages")
            
            for page_num in tqdm(range(total_pages), desc="Processing PDF pages"):
                page = pdf_document[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Clean and chunk text
                cleaned_text = self._clean_text(text)
                text_chunks = self._chunk_text(cleaned_text, page_num + 1)
                
                # Extract images
                page_images = self._extract_images_from_pdf_page(page, page_num + 1, file_path)
                images.extend(page_images)
                
                # Store page data
                page_data = {
                    "page_number": page_num + 1,
                    "text": cleaned_text,
                    "text_chunks": text_chunks,
                    "has_images": len(page_images) > 0,
                    "image_paths": [img["path"] for img in page_images]
                }
                
                pages.append(page_data)
                
                # Log memory usage
                self.memory_manager.log_memory_usage(f"pdf_page_{page_num + 1}")
            
            # Aggregate chunk stats
            total_chunks = sum(len(p["text_chunks"]) for p in pages)
            if total_chunks > 0:
                total_chars = sum(len(c["text"]) for p in pages for c in p["text_chunks"])
                avg = total_chars // total_chunks
                logger.info(f"Chunks created: {total_chunks}, avg size: {avg} chars")
            
            pdf_document.close()
            
            # Generate book metadata
            book_id = self._generate_book_id(file_path)
            metadata = self._extract_pdf_metadata(file_path)
            
            return {
                "book_id": book_id,
                "title": metadata.get("title", file_path.stem),
                "author": metadata.get("author", "Unknown"),
                "file_path": str(file_path),
                "file_type": "pdf",
                "total_pages": total_pages,
                "pages": pages,
                "images": images,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def _process_docx(self, file_path: Path) -> Dict[str, Any]:
        """Process DOCX file and extract text"""
        if docx is None:
            raise ImportError("python-docx is not installed. Please install it with: pip install python-docx")
        
        try:
            # Load document
            doc = docx.Document(file_path)
            
            pages = []
            current_page = []
            current_text = ""
            
            logger.info("Processing DOCX document")
            
            # Process paragraphs
            for para in tqdm(doc.paragraphs, desc="Processing DOCX paragraphs"):
                text = para.text.strip()
                if text:
                    current_text += text + "\n"
                    current_page.append(text)
                
                # Simple page break detection (every ~500 characters)
                if len(current_text) > 500:
                    # Create page
                    page_data = {
                        "page_number": len(pages) + 1,
                        "text": current_text,
                        "text_chunks": self._chunk_text(current_text, len(pages) + 1),
                        "has_images": False,
                        "image_paths": []
                    }
                    pages.append(page_data)
                    
                    # Reset for next page
                    current_text = ""
                    current_page = []
            
            # Add remaining content as last page
            if current_text.strip():
                page_data = {
                    "page_number": len(pages) + 1,
                    "text": current_text,
                    "text_chunks": self._chunk_text(current_text, len(pages) + 1),
                    "has_images": False,
                    "image_paths": []
                }
                pages.append(page_data)
            
            # Generate book metadata
            book_id = self._generate_book_id(file_path)
            
            return {
                "book_id": book_id,
                "title": file_path.stem,
                "author": "Unknown",
                "file_path": str(file_path),
                "file_type": "docx",
                "total_pages": len(pages),
                "pages": pages,
                "images": [],
                "metadata": {"title": file_path.stem}
            }
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise
    
    def _process_text(self, file_path: Path) -> Dict[str, Any]:
        """Process text file"""
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean text
            cleaned_text = self._clean_text(content)
            
            # Chunk text into pages
            text_chunks = self._chunk_text(cleaned_text, 1)
            
            # Create single page
            page_data = {
                "page_number": 1,
                "text": cleaned_text,
                "text_chunks": text_chunks,
                "has_images": False,
                "image_paths": []
            }
            
            # Generate book metadata
            book_id = self._generate_book_id(file_path)
            
            return {
                "book_id": book_id,
                "title": file_path.stem,
                "author": "Unknown",
                "file_path": str(file_path),
                "file_type": "text",
                "total_pages": 1,
                "pages": [page_data],
                "images": [],
                "metadata": {"title": file_path.stem}
            }
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            raise
    
    def _process_image(self, file_path: Path) -> Dict[str, Any]:
        """Process single image file"""
        try:
            # For images, we'll treat them as single-page documents
            # Text will be extracted via OCR in the embedding process
            
            page_data = {
                "page_number": 1,
                "text": "",  # Will be filled by OCR
                "text_chunks": [],
                "has_images": True,
                "image_paths": [str(file_path)]
            }
            
            # Generate book metadata
            book_id = self._generate_book_id(file_path)
            
            return {
                "book_id": book_id,
                "title": file_path.stem,
                "author": "Unknown",
                "file_path": str(file_path),
                "file_type": "image",
                "total_pages": 1,
                "pages": [page_data],
                "images": [{"path": str(file_path), "page_number": 1}],
                "metadata": {"title": file_path.stem}
            }
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and math operators
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\+\=\<\>\*\/\&\|\%\#\@\~\^\"\'\$\\]', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """Split text into chunks for processing"""
        if not text:
            return []
        
        chunk_size = self.config["chunk_size"]
        overlap = self.config["overlap"]
        min_chunk_size = self.config["min_chunk_size"]
        max_chunk_size = self.config["max_chunk_size"]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine chunk end
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start + min_chunk_size:
                    end = last_space
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    "page_number": page_number,
                    "chunk_id": f"page_{page_number}_chunk_{len(chunks) + 1}"
                })
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_images_from_pdf_page(self, page, page_number: int, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract images from PDF page"""
        images = []
        
        try:
            # Get image list from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    
                    # Save image
                    img_filename = f"{pdf_path.stem}_page_{page_number}_img_{img_index + 1}.png"
                    img_path = pdf_path.parent / "extracted_images" / img_filename
                    
                    # Create directory if it doesn't exist
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save image
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    
                    images.append({
                        "path": str(img_path),
                        "page_number": page_number,
                        "image_index": img_index + 1
                    })
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Error extracting image {img_index} from page {page_number}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Error extracting images from page {page_number}: {e}")
        
        return images
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata:
                    metadata = pdf_reader.metadata
                    return {
                        "title": metadata.get('/Title', file_path.stem),
                        "author": metadata.get('/Author', 'Unknown'),
                        "subject": metadata.get('/Subject', ''),
                        "creator": metadata.get('/Creator', ''),
                        "producer": metadata.get('/Producer', ''),
                        "creation_date": metadata.get('/CreationDate', ''),
                        "modification_date": metadata.get('/ModDate', '')
                    }
                
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
        
        return {"title": file_path.stem, "author": "Unknown"}
    
    def _generate_book_id(self, file_path: Path) -> str:
        """Generate unique book ID based on file path and content"""
        try:
            # Use file path and modification time for ID
            stat = file_path.stat()
            content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            
            # Generate hash
            book_id = hashlib.md5(content.encode()).hexdigest()[:16]
            return book_id
            
        except Exception as e:
            logger.error(f"Error generating book ID: {e}")
            # Fallback to simple hash
            return hashlib.md5(str(file_path).encode()).hexdigest()[:16]
    
    def get_processing_stats(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about processed book"""
        total_chunks = sum(len(page["text_chunks"]) for page in book_data["pages"])
        total_images = len(book_data["images"])
        total_text_length = sum(len(page["text"]) for page in book_data["pages"])
        
        return {
            "book_id": book_data["book_id"],
            "title": book_data["title"],
            "total_pages": book_data["total_pages"],
            "total_chunks": total_chunks,
            "total_images": total_images,
            "total_text_length": total_text_length,
            "average_chunks_per_page": total_chunks / book_data["total_pages"] if book_data["total_pages"] > 0 else 0,
            "file_size_mb": Path(book_data["file_path"]).stat().st_size / (1024 * 1024)
        } 
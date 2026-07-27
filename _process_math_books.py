"""
Process the L8 Math Foundation textbooks into the vector DB.
"""
import sys; sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO)
import time
from pathlib import Path

from book_processor import BookProcessor
from vector_db import VectorDatabase
from embeddings import EmbeddingGenerator
from memory_manager import MemoryManager
from book_manager import BookManager
from config import DATA_DIR, USER_BOOKS_DIR

MATH_BOOKS = [
    ("Calculus - Michael Spivak.pdf",     "Spivak Calculus",          "Michael Spivak"),
    ("Principles_of_Mathematical_Analysis-Rudin.pdf", "Principles of Mathematical Analysis", "Walter Rudin"),
    ("Linear Algebra Done Right.pdf",     "Linear Algebra Done Right",  "Sheldon Axler"),
    ("Abstract Algebra.pdf",              "Abstract Algebra",          "Dummit & Foote"),
    ("The Princeton companion to mathematics.pdf", "The Princeton Companion to Mathematics", "Timothy Gowers"),
]

mm = MemoryManager()
bp = BookProcessor(memory_manager=mm)
vdb = VectorDatabase()
eg = EmbeddingGenerator()
bm = BookManager(book_processor=bp, vector_db=vdb, embedding_generator=eg, memory_manager=mm)

for fname, title, author in MATH_BOOKS:
    fpath = USER_BOOKS_DIR / fname
    if not fpath.exists():
        print(f"MISSING: {fpath}")
        continue
    if fpath.stem in bm.processed_books:
        print(f"ALREADY PROCESSED: {title}")
        continue
    print(f"\n{'='*60}\nProcessing: {title} ({fname})\n{'='*60}")
    t0 = time.time()
    result = bm.process_single_book(fpath, title=title, author=author, skip_ocr=True)
    elapsed = time.time() - t0
    if result["success"]:
        s = result["stats"]
        print(f"  OK: {title} — {s['total_pages']} pages, {s['total_chunks']} chunks, {s['total_text_length']} chars ({elapsed:.1f}s)")
    else:
        print(f"  FAIL: {title} — {result.get('error', 'unknown')}")

print("\n\nDone. Books in DB:")
vdb.cursor.execute("SELECT id, title, author, total_pages FROM books WHERE title LIKE '%Spivak%' OR title LIKE '%Rudin%' OR title LIKE '%Axler%' OR title LIKE '%Abstract Algebra%' OR title LIKE '%Princeton%'")
for r in vdb.cursor.fetchall():
    print(f"  {r[0][:20]:20s} | {str(r[1]):50s} | {str(r[2]):30s} | {r[3]} pages")
vdb.conn.close()

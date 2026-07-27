"""Inspect the database: tables, books, and their processing status."""
import sqlite3
from config import DATA_DIR

db_path = str(DATA_DIR / "database.db")
con = sqlite3.connect(db_path)

print("=== Tables ===")
tables = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
for t in tables:
    cols = con.execute(f"PRAGMA table_info({t[0]})").fetchall()
    col_str = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
    print(f"  {t[0]}: {col_str}")

print("\n=== Books ===")
# Try different column names
books_cols = [c[1] for c in con.execute("PRAGMA table_info(books)").fetchall()]
print(f"  Columns: {books_cols}")

# Build SELECT dynamically
id_col = "id" if "id" in books_cols else [c for c in books_cols if "id" in c.lower()][0]
title_col = "title" if "title" in books_cols else [c for c in books_cols if "title" in c.lower()][0]
author_col = "author" if "author" in books_cols else None

query = f"SELECT {id_col}, {title_col}"
if author_col:
    query += f", {author_col}"
query += " FROM books ORDER BY title"

rows = con.execute(query).fetchall()
for r in rows:
    bid = r[0]
    title = str(r[1] or "")[:60]
    author = str(r[2] or "")[:40] if len(r) > 2 else "?"
    page_count = con.execute("SELECT COUNT(*) FROM pages WHERE book_id=?", (str(bid),)).fetchone()[0]
    print(f"  {str(bid)[:30]:30s} | {title:60s} | {author:40s} | {page_count} pages")

print("\n=== Books without pages ===")
for r in rows:
    bid = r[0]
    pc = con.execute("SELECT COUNT(*) FROM pages WHERE book_id=?", (str(bid),)).fetchone()[0]
    if pc == 0:
        print(f"  {str(bid)[:30]:30s} | {str(r[1] or '')[:60]}")

con.close()

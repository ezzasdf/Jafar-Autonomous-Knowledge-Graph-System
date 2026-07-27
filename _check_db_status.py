import sqlite3
db = sqlite3.connect('E:/projects/Ai/data/database.db')
rows = db.execute('SELECT id, title, total_pages FROM books WHERE title LIKE ? OR title LIKE ? OR title LIKE ? OR title LIKE ? OR title LIKE ?',
    ('%Spivak%','%Rudin%','%Linear Algebra%','%Abstract Algebra%','%Princeton%')).fetchall()
for r in rows:
    c = db.execute('SELECT COUNT(*) FROM pages WHERE book_id=?', (r[0],)).fetchone()[0]
    print(f'{r[1][:50]:50s} | {r[0][:20]:20s} | pages={c:>4d}/{r[2]}')
db.close()

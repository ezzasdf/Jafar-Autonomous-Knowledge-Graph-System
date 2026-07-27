import sqlite3
from pathlib import Path

db = sqlite3.connect('E:/projects/Ai/data/database.db')

book_ids = ['76edbaaa120d41e4', '36e0c691f258b195', 'b74c500a69bc22cc']
for bid in book_ids:
    c = db.execute('SELECT COUNT(*) FROM pages WHERE book_id=?', (bid,)).fetchone()[0]
    info = db.execute('SELECT id, title FROM books WHERE id=?', (bid,)).fetchone()
    if info:
        label = f'{info[1]}' if info[1] else 'UNKNOWN'
        print(f'{bid}: {label} - pages_in_db={c} -> deleting')
        db.execute('DELETE FROM pages WHERE book_id=?', (bid,))
        db.execute('DELETE FROM books WHERE id=?', (bid,))
    else:
        print(f'{bid}: NOT FOUND')

db.commit()
db.close()
print('Done')

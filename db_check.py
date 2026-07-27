import sqlite3
conn = sqlite3.connect('data/database.db')
c = conn.cursor()

c.execute("PRAGMA table_info(learning_goals)")
for r in c.fetchall():
    print(r)

print()
c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='learning_goals'")
print(c.fetchone()[0])

conn.close()

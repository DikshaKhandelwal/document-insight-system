import sqlite3, os

db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'documents.db')
print('DB path:', db_path)
if not os.path.exists(db_path):
    print('DB not found')
    raise SystemExit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

try:
    cur.execute('SELECT COUNT(*) FROM documents')
    docs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM documents WHERE processing_status='completed'")
    completed = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM sections')
    sections = cur.fetchone()[0]
    print(f'documents: {docs}, completed: {completed}, sections: {sections}')

    print('\nLast 5 sections:')
    cur.execute('SELECT id, document_id, section_title, substr(section_text,1,200), page_number, level FROM sections ORDER BY id DESC LIMIT 5')
    rows = cur.fetchall()
    for r in rows:
        print({'id': r[0], 'document_id': r[1], 'title': (r[2] or '')[:80], 'text_preview': (r[3] or '')[:140], 'page': r[4], 'level': r[5]})

except Exception as e:
    print('DB query error:', e)
finally:
    conn.close()

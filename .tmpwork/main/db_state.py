import os, sys
sys.path.insert(0, r"D:\project\python\asisten")
from dotenv import load_dotenv
load_dotenv(r"D:\project\python\asisten\.env")
import pymysql

conn = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", ""),
    charset="utf8mb4",
)
cur = conn.cursor()
cur.execute("SHOW DATABASES")
dbs = [r[0] for r in cur.fetchall()]
print("DATABASES:", dbs)

for db in dbs:
    if db in ("information_schema", "mysql", "performance_schema", "sys"):
        continue
    cur.execute(f"SHOW TABLES FROM `{db}`")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\n--- {db}: {len(tables)} tables ---")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM `{db}`.`{t}`")
        n = cur.fetchone()[0]
        print(f"   {t:28s} rows={n}")
conn.close()

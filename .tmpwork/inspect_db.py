"""Read-only inspection of the local MySQL schema (no secrets printed)."""
import os
from pathlib import Path

from dotenv import load_dotenv
import pymysql

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

conn = pymysql.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", ""),
    database=os.getenv("DB_NAME", "asisten_db"),
    charset="utf8mb4",
)

with conn.cursor() as cur:
    cur.execute("SHOW TABLES")
    tables = [r[0] for r in cur.fetchall()]
    print("tables in %s:" % os.getenv("DB_NAME"))
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM `{t}`")
        n = cur.fetchone()[0]
        cur.execute(f"SHOW COLUMNS FROM `{t}`")
        cols = [c[0] for c in cur.fetchall()]
        print(f"  - {t:<22} rows={n:<4} cols={cols}")

conn.close()
print("\n(secrets were read from .env and never printed)")

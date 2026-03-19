import sqlite3

conn = sqlite3.connect("full_db.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

for (name,) in tables:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({name})").fetchall()]
    rows = conn.execute(f"SELECT * FROM {name} LIMIT 5").fetchall()

    if not rows:
        print("  (empty table)")
        continue

    widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*cols))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))

conn.close()

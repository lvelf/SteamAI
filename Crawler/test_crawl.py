# test_crawl_one.py
from Crawler.db import get_conn, ensure_tables, get_state, upsert_reviews, update_state
from Crawler.steam_reviews import fetch_reviews_incremental
from pathlib import Path

DB_PATH = str((Path(__file__).resolve().parents[1]) / "steam_reviews_raw.sqlite")
appid = 620  # Portal 2

conn = get_conn(DB_PATH)
ensure_tables(conn)

since = get_state(conn, appid)
res = fetch_reviews_incremental(appid, since, max_pages=3, sleep_s=1.0, verbose=True)
print("pages:", res.pages_fetched, "stopped:", res.stopped_reason)
print("fetched:", len(res.reviews), "newest:", res.newest_ts_updated)

if res.reviews:
    upsert_reviews(conn, appid, res.reviews)
    update_state(conn, appid, res.newest_ts_updated)

conn.commit()
conn.close()
print("done")

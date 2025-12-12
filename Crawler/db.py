# SteamAI/Crawler/db.py
import sqlite3
from datetime import datetime, timezone

def get_conn(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS reviews_raw (
      appid INTEGER NOT NULL,
      recommendationid TEXT NOT NULL,
      author_steamid TEXT,
      language TEXT,
      review TEXT,
      voted_up INTEGER,
      votes_up INTEGER,
      votes_funny INTEGER,
      weighted_vote_score REAL,
      playtime_forever INTEGER,
      playtime_last_two_weeks INTEGER,
      playtime_at_review INTEGER,
      timestamp_created INTEGER,
      timestamp_updated INTEGER,
      steam_purchase INTEGER,
      received_for_free INTEGER,
      written_during_early_access INTEGER,
      PRIMARY KEY (appid, recommendationid)
    );
    CREATE TABLE IF NOT EXISTS crawl_state (
      appid INTEGER PRIMARY KEY,
      last_timestamp_updated INTEGER DEFAULT 0,
      last_run_at TEXT
    );
    """)

def get_state(conn: sqlite3.Connection, appid: int) -> int:
    row = conn.execute("SELECT last_timestamp_updated FROM crawl_state WHERE appid=?", (appid,)).fetchone()
    return int(row[0]) if row else 0

def upsert_reviews(conn: sqlite3.Connection, appid: int, reviews: list[dict]):
    sql = """
    INSERT OR REPLACE INTO reviews_raw (
      appid, recommendationid, author_steamid, language, review, voted_up,
      votes_up, votes_funny, weighted_vote_score,
      playtime_forever, playtime_last_two_weeks, playtime_at_review,
      timestamp_created, timestamp_updated,
      steam_purchase, received_for_free, written_during_early_access
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    rows = []
    for rv in reviews:
        author = rv.get("author") or {}
        rows.append((
            appid,
            str(rv.get("recommendationid")),
            str(author.get("steamid")) if author.get("steamid") else None,
            rv.get("language"),
            rv.get("review"),
            int(bool(rv.get("voted_up"))) if rv.get("voted_up") is not None else None,
            rv.get("votes_up"),
            rv.get("votes_funny"),
            float(rv.get("weighted_vote_score")) if rv.get("weighted_vote_score") not in (None, "") else None,
            author.get("playtime_forever"),
            author.get("playtime_last_two_weeks"),
            author.get("playtime_at_review"),
            rv.get("timestamp_created"),
            rv.get("timestamp_updated"),
            int(bool(rv.get("steam_purchase"))) if rv.get("steam_purchase") is not None else None,
            int(bool(rv.get("received_for_free"))) if rv.get("received_for_free") is not None else None,
            int(bool(rv.get("written_during_early_access"))) if rv.get("written_during_early_access") is not None else None,
        ))
    conn.executemany(sql, rows)

def update_state(conn: sqlite3.Connection, appid: int, newest_ts_updated: int):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
      INSERT INTO crawl_state(appid, last_timestamp_updated, last_run_at)
      VALUES (?, ?, ?)
      ON CONFLICT(appid) DO UPDATE SET
        last_timestamp_updated=excluded.last_timestamp_updated,
        last_run_at=excluded.last_run_at
    """, (appid, int(newest_ts_updated), now))

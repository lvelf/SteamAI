from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../SteamAI/SteamAI
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# DB / Parquet Ptah 
DEFAULT_DB_PATH = str(PROJECT_ROOT / "steam_reviews_raw.sqlite")
DB_PATH = Variable.get("STEAMAI_REVIEWS_DB", default_var=DEFAULT_DB_PATH)

DEFAULT_APPS_STATS_PATH = str(PROJECT_ROOT / "data" / "processed" / "apps_with_stats.parquet")
APPS_STATS_PATH = Variable.get("STEAMAI_APPS_STATS_PATH", default_var=DEFAULT_APPS_STATS_PATH)

TOPK = int(Variable.get("STEAMAI_CRAWL_TOPK", default_var="200"))
MAX_PAGES = int(Variable.get("STEAMAI_CRAWL_MAX_PAGES", default_var="10"))
SLEEP_S = float(Variable.get("STEAMAI_CRAWL_SLEEP_S", default_var="1.0"))

from Crawler.db import get_conn, ensure_tables, get_state, upsert_reviews, update_state
from Crawler.steam_reviews import fetch_reviews_incremental

default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="steam_reviews_crawl_raw",
    start_date=datetime(2025, 12, 12),
    schedule="0 3 * * *",  # 03:00
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["steamai", "crawler"],
) as dag:

    @task
    def pick_appids() -> List[int]:
        """
        pick appid list
        from apps_with_stats.parquet take TOPK
        """
        if not os.path.exists(APPS_STATS_PATH):
            return [620, 440, 570][: min(TOPK, 3)]

        df = pd.read_parquet(APPS_STATS_PATH)

        if "appid" not in df.columns:
            return [620, 440, 570][: min(TOPK, 3)]

    
        if "review_count" in df.columns:
            df = df.sort_values("review_count", ascending=False)

        df = df[df["appid"].notna()]
        appids = df["appid"].astype(int).head(TOPK).tolist()
        return appids

    @task(pool="steam_api_pool")
    def crawl_one(appid: int) -> dict:
        """
        singel appid: incremental fetch → write SQLite raw → update crawl_state watermark
        """
        conn = get_conn(DB_PATH)
        ensure_tables(conn)

        since = get_state(conn, appid)

        res = fetch_reviews_incremental(
            appid=appid,
            since_ts_updated=since,
            max_pages=MAX_PAGES,
            sleep_s=SLEEP_S,
            timeout_s=30,
            max_retries=8,
            base_backoff_s=1.0,
            max_backoff_s=60.0,
            filter_mode="updated",
            language="all",
            verbose=False,
        )

        fetched = len(res.reviews)
        if fetched > 0:
            upsert_reviews(conn, appid, res.reviews)
            update_state(conn, appid, res.newest_ts_updated)

        conn.commit()
        conn.close()

        return {
            "appid": appid,
            "since": since,
            "fetched": fetched,
            "pages": res.pages_fetched,
            "stopped": res.stopped_reason,
            "newest": res.newest_ts_updated,
            "db_path": DB_PATH,
        }

    appids = pick_appids()
    crawl_one.expand(appid=appids)

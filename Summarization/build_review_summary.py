from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List

import pandas as pd

from summarizer import ReviewSummarizer, SummarizerConfig



THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/Website
PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final
DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data
PROCESSED_DIR = os.path.join(DATA_BASE, "processed/")
THIS_DIR_DATA = os.path.join(THIS_DIR, "data")

APPS_PATH = os.path.join(PROCESSED_DIR, "apps_with_stats.parquet")
REVIEWS_PATH = os.path.join(PROCESSED_DIR, "reviews.parquet")
SUMMARIES_PATH = os.path.join(PROCESSED_DIR, "review_summaries.parquet")
SUMMARIES_PATH2 = os.path.join(THIS_DIR_DATA, "review_summaries.parquet")

model_name = "mistralai/Mistral-7B-Instruct-v0.2"


def pick_reviews_for_app(df_app: pd.DataFrame, max_reviews: int = 80) -> List[str]:
    """
    Choose reviews to summarize
    """
    
    df = df_app.copy()

    if "votes_up" in df.columns:
        df["votes_up"] = df["votes_up"].fillna(0)
    else:
        df["votes_up"] = 0

    if "timestamp_created" in df.columns:
        df["timestamp_created"] = df["timestamp_created"].fillna(0)
    else:
        df["timestamp_created"] = 0

    df = df.sort_values(
        by=["votes_up", "timestamp_created"],
        ascending=[False, False],
    )

    texts = df["review_text"].dropna().astype(str).tolist()
    return texts[:max_reviews]

def build_review_summaries(
    top_n_apps: int = 100,
    min_review_count: int = 50,
    sleep_sec: float = 0.2,
    model_name: str = model_name,
    use_4bit: bool = False,
) -> pd.DataFrame:
    
    print("DATA_BASE:", DATA_BASE)
    print("APPS_PATH:", APPS_PATH)
    print("REVIEWS_PATH:", REVIEWS_PATH)
    print("SUMMARIES_PATH:", SUMMARIES_PATH)
    
    print("Loading apps_clean ...")
    apps = pd.read_parquet(APPS_PATH)
    assert "appid" in apps.columns
    assert "review_count" in apps.columns

    print("Loading reviews ...")
    reviews = pd.read_parquet(REVIEWS_PATH)
    assert "appid" in reviews.columns
    assert "review_text" in reviews.columns

    
    # cast to int
    apps["appid"] = pd.to_numeric(apps["appid"], errors="coerce")
    reviews["appid"] = pd.to_numeric(reviews["appid"], errors="coerce")

    # only save English
    if "language" in reviews.columns:
        reviews = reviews[reviews["language"].str.lower() == "english"]
        
    
    # review count filter
    apps_popular = (
        apps[apps["review_count"] >= min_review_count]
        .sort_values("review_count", ascending=False)
        .head(top_n_apps)
    )
    target_appids = apps_popular["appid"].tolist()
    print(f"Target apps: {len(target_appids)} (min_review_count={min_review_count})")
    
    summaries_rows = []
    done_appids = set()
    
    
    if os.path.exists(SUMMARIES_PATH):
        print("Found existing summary file, resuming:", SUMMARIES_PATH)
        existing = pd.read_parquet(SUMMARIES_PATH)

        if "appid" in existing.columns:
            summaries_rows.extend(existing.to_dict("records"))
            done_appids = set(existing["appid"].tolist())
            print("Already summarized apps:", len(done_appids))
        else:
            print("Existing summary file has no 'appid' column, ignoring old summaries.")
    
    appid_to_name = dict(zip(apps["appid"], apps["name"]))
    appid_to_total_reviews = dict(zip(apps["appid"], apps["review_count"]))
    
    grouped = reviews.groupby("appid")
    
    cfg = SummarizerConfig(
        model_name=model_name,
        use_4bit=use_4bit,
        max_new_tokens=512,
        temperature=0.4,
        max_input_chars=12000,
    )
    summarizer = ReviewSummarizer(cfg)

    total = len(target_appids)
    for idx, appid in enumerate(target_appids, start=1):
        if appid in done_appids:
            print(f"[{idx}/{total}] appid={appid} already summarized, skip.")
            continue

        if appid not in grouped.groups:
            print(f"[{idx}/{total}] appid={appid} has no reviews, skip.")
            continue

        df_app = grouped.get_group(appid)
        texts = pick_reviews_for_app(df_app, max_reviews=80)
        if not texts:
            print(f"[{idx}/{total}] appid={appid} has no review_text, skip.")
            continue

        app_name = appid_to_name.get(appid, f"appid={appid}")
        n_total_reviews = int(appid_to_total_reviews.get(appid, len(df_app)))

        print(
            f"[{idx}/{total}] Summarizing {appid} - {app_name} "
            f"(using {len(texts)} reviews, total={n_total_reviews})"
        )

        try:
            summary_text = summarizer.summarize_reviews(app_name, texts)
        except Exception as e:
            print("  Error during summarization:", e)
            summary_text = None

        row = {
            "appid": int(appid),
            "name": app_name,
            "summary": summary_text,
            "n_reviews_used": len(texts),
            "n_reviews_total": n_total_reviews,
            "last_updated": datetime.utcnow(),
        }
        summaries_rows.append(row)
        done_appids.add(appid)

        
        if len(summaries_rows) % 10 == 0:
            tmp_df = pd.DataFrame(summaries_rows)
            tmp_df.to_parquet(SUMMARIES_PATH, index=False)
            print("  [autosave] wrote", len(tmp_df), "rows to", SUMMARIES_PATH)
            tmp_df.to_parquet(SUMMARIES_PATH2, index=False)
            print("  [autosave] wrote", len(tmp_df), "rows to", SUMMARIES_PATH2)

        time.sleep(sleep_sec)

   
    final_df = pd.DataFrame(summaries_rows)
    final_df.to_parquet(SUMMARIES_PATH, index=False)
    print("All done. Total summaries:", len(final_df))
    print("Saved to:", SUMMARIES_PATH)

    return final_df

if __name__ == "__main__":
    
    df = build_review_summaries(
        top_n_apps=10,
        min_review_count=10,
        model_name= model_name,
        use_4bit=False,
    )
    print(df.head())
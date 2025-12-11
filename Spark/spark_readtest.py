from pyspark.sql import SparkSession
import os
import pandas as pd
spark = SparkSession.builder.master("local[*]").appName("check").getOrCreate()

THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/Website
PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final
DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data
PROCESSED_PATh = os.path.join(DATA_BASE, "processed")
app_clean_path = os.path.join(PROCESSED_PATh, "apps_clean.parquet")
app_with_stats_path = os.path.join(PROCESSED_PATh, "apps_with_stats.parquet")

PROCESSED_DIR = os.path.join(DATA_BASE, "processed")
APPS_PATH = os.path.join(PROCESSED_DIR, "apps_with_stats.parquet")
REVIEWS_PATH = os.path.join(PROCESSED_DIR, "reviews.parquet")

print("APPS_PATH:", APPS_PATH)
print("REVIEWS_PATH:", REVIEWS_PATH)

"""
apps = spark.read.parquet(app_clean_path)
apps.printSchema()
apps.show(5, truncate=False)

apps_stats = spark.read.parquet(app_with_stats_path)
apps_stats.printSchema()
apps_stats.show(5, truncate=False)

emb_path = os.path.join(PROCESSED_PATh, "apps_embeddings.parquet")
emb = spark.read.parquet(emb_path)
emb.printSchema()
emb.show(3, truncate=False)

spark.stop()
"""

print("\n=== Loading parquet files ===")
apps = pd.read_parquet(APPS_PATH)
reviews = pd.read_parquet(REVIEWS_PATH)

apps["appid"] = pd.to_numeric(apps["appid"], errors="coerce")
reviews["appid"] = pd.to_numeric(reviews["appid"], errors="coerce")

print("apps_with_stats rows:", len(apps))
print("reviews rows:", len(reviews))

print("\napps_with_stats columns:", apps.columns.tolist())
print("reviews columns:", reviews.columns.tolist())

print("\n=== Top 10 apps by review_count (from apps_with_stats) ===")
if "review_count" not in apps.columns:
    print("WARNING: apps_with_stats has no 'review_count' column!")
else:
    top_apps = apps.sort_values("review_count", ascending=False)[["appid", "name", "review_count"]].head(10)
    print(top_apps.to_string(index=False))

    top_appids = top_apps["appid"].tolist()

   
    print("\n=== For those top 10 appids, how many rows exist in reviews.parquet? ===")
    for appid in top_appids:
        n_all = (reviews["appid"] == appid).sum()
        print(f"appid={appid:<8} | reviews rows (all languages): {n_all}")

   
    if "language" in reviews.columns:
        reviews_eng = reviews[reviews["language"].str.lower() == "english"]
        print("\nreviews (english only) rows:", len(reviews_eng))

        print("\n=== For those top 10 appids, english-only review counts ===")
        for appid in top_appids:
            n_eng = ((reviews_eng["appid"] == appid)).sum()
            print(f"appid={appid:<8} | reviews rows (english only): {n_eng}")
    else:
        print("\nreviews.parquet has no 'language' column, skip language-based check.")

print("\n=== Overall appid coverage check ===")
apps_appids = set(apps["appid"].tolist())
reviews_appids = set(reviews["appid"].tolist())

print("unique appids in apps_with_stats:", len(apps_appids))
print("unique appids in reviews.parquet:", len(reviews_appids))
print("intersection size (appid in both):", len(apps_appids & reviews_appids))

print("\nDone.")
import os
from steam_recommender import SteamRecommender

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_BASE = os.path.join(FINAL_ROOT, "data")
PROCESSED = os.path.join(DATA_BASE, "processed")

apps_parquet = os.path.join(PROCESSED, "apps_with_stats.parquet")
emb_parquet = os.path.join(PROCESSED, "apps_embeddings.parquet")

rec = SteamRecommender(apps_parquet, emb_parquet)

# Fuzz Search
print(rec.find_appid_by_name("crown tric", top_k=5))

# filter
df = rec.recommend_by_name(
    "Portal 2",
    top_k_candidates=3,
    top_k=10,
    min_year=2015,
    min_review_count=100,
    min_positive_ratio=0.8,
)
print(df.head(10))


df = rec.recommend_by_name(
    "Portal 2",
    top_k_candidates=3,
    top_k=10,
    min_year=2010,
    min_review_count=10,
    min_positive_ratio=0.6,
)
print(df.head(10))

df = rec.recommend_by_name(
    "Portal 2",
)
print(df.head(10))
import os
import pandas as pd


THIS_DIR = os.path.dirname(__file__)             # .../SteamAI/SteamAI/Summarization
PROJECT_ROOT = os.path.dirname(THIS_DIR)         # .../SteamAI/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)       # .../SteamAI
DATA_BASE = os.path.join(FINAL_ROOT, "data")     # .../SteamAI/data
PROCESSED_DIR = os.path.join(DATA_BASE, "processed")

SUMMARIES_PATH = os.path.join(PROCESSED_DIR, "review_summaries.parquet")
APPS_PATH = os.path.join(PROCESSED_DIR, "apps_with_stats.parquet")

print("SUMMARIES_PATH:", SUMMARIES_PATH)
print("APPS_PATH:", APPS_PATH)

# ========= read summary  =========
df = pd.read_parquet(SUMMARIES_PATH)
print("\n=== review_summaries.parquet ===")
print("rows:", len(df))
print("columns:", df.columns.tolist())


preview = df.copy()
if "summary" in preview.columns:
    preview["summary_preview"] = preview["summary"].astype(str).str.slice(0, 120)
    cols = [c for c in preview.columns if c != "summary"]
    cols.append("summary_preview")
    print(preview[cols].head().to_string(index=False))
else:
    print(df.head())



TEST_APPID = None          
TEST_NAME_SUBSTR = "Portal" 


if "name" in df.columns:
    mask = df["name"].astype(str).str.contains(TEST_NAME_SUBSTR, case=False, na=False)
    df_match = df[mask]
    if df_match.empty:
        print("No '%s'  summary" % TEST_NAME_SUBSTR)
    else:
        row = df_match.iloc[0]
        print(f"\n appid={row.get('appid')}  name={row.get('name')}")
        print("\n---- FULL SUMMARY ----")
        print(row.get("summary"))
else:
    print("No summary  'name' column")

if TEST_APPID is not None and "appid" in df.columns:
    
    df_app = df[df["appid"] == int(TEST_APPID)]
    if df_app.empty:
        print("No appid=%s 的 summary" % TEST_APPID)
    else:
        row = df_app.iloc[0]
        print(f"\n appid={row.get('appid')}  name={row.get('name')}")
        print("\n---- FULL SUMMARY ----")
        print(row.get("summary"))


if os.path.exists(APPS_PATH):
    apps = pd.read_parquet(APPS_PATH)
    
    apps["appid"] = pd.to_numeric(apps["appid"], errors="coerce")
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    apps = apps.dropna(subset=["appid"])
    df = df.dropna(subset=["appid"])
    apps["appid"] = apps["appid"].astype("int64")
    df["appid"] = df["appid"].astype("int64")

    merged = df.merge(apps[["appid", "name", "review_count"]], on="appid", how="left", suffixes=("", "_apps"))
    print(
        merged[["appid", "name", "review_count", "n_reviews_used", "n_reviews_total"]]
        .head()
        .to_string(index=False)
    )

print("\nDone.")

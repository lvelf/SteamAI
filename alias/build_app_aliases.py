# build_app_aliases.py
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd

from alias_generator import AliasGenerator, AliasConfig


THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/alias
PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final
DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data
PROCESSED_DIR = os.path.join(DATA_BASE, "processed/")
THIS_DIR_DATA = os.path.join(THIS_DIR, "data")

APPS_PATH = os.path.join(PROCESSED_DIR, "apps_with_stats.parquet")
ALIASES_PATH = os.path.join(PROCESSED_DIR, "app_aliases.parquet")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def to_genres_text(genres) -> str:
    if genres is None:
        return ""
    if isinstance(genres, float) and pd.isna(genres):
        return ""
    if isinstance(genres, (list, tuple, set)):
        items = [str(x).strip() for x in genres if x is not None and str(x).strip()]
        return ", ".join(items)
    try:
        import numpy as np
        if isinstance(genres, np.ndarray):
            items = [str(x).strip() for x in genres.tolist() if x is not None and str(x).strip()]
            return ", ".join(items)
    except Exception:
        pass

    
    s = str(genres).strip()
    s = s.strip("[]")
    return s


def build_app_aliases(
    top_n_apps: int = 200,
    min_review_count: int = 50,
    sleep_sec: float = 0.15,
    model_name: str = MODEL_NAME,
    use_4bit: bool = False,
    alias_k: int = 6,
) -> pd.DataFrame:
    print("DATA_BASE:", DATA_BASE)
    print("APPS_PATH:", APPS_PATH)
    print("ALIASES_PATH:", ALIASES_PATH)

    print("Loading apps_with_stats ...")
    apps = pd.read_parquet(APPS_PATH)
    assert "appid" in apps.columns
    assert "name" in apps.columns

    
    apps["appid"] = pd.to_numeric(apps["appid"], errors="coerce")
    apps = apps.dropna(subset=["appid"])
    apps["appid"] = apps["appid"].astype(int)

    if "review_count" in apps.columns:
        apps_target = (
            apps[apps["review_count"] >= min_review_count]
            .sort_values("review_count", ascending=False)
            .head(top_n_apps)
        )
    else:
        apps_target = apps.head(top_n_apps)

    target_appids = apps_target["appid"].tolist()
    print(f"Target apps: {len(target_appids)} (min_review_count={min_review_count})")

    
    rows = []
    done = set()
    if os.path.exists(ALIASES_PATH):
        print("Found existing aliases file, resuming:", ALIASES_PATH)
        existing = pd.read_parquet(ALIASES_PATH)
        if "appid" in existing.columns:
            rows.extend(existing.to_dict("records"))
            done = set(existing["appid"].astype(int).tolist())
            print("Already generated:", len(done))
        else:
            print("Existing aliases file missing 'appid', ignoring it.")

    
    cfg = AliasConfig(
        model_name=model_name,
        use_4bit=use_4bit,
        max_new_tokens=192,
        temperature=0.2,
        max_input_chars=6000,
    )
    gen = AliasGenerator(cfg)

    
    has_desc = "short_description" in apps.columns
    has_genres = "genres" in apps.columns

    total = len(target_appids)
    apps_target_indexed = apps_target.set_index("appid")

    for idx, appid in enumerate(target_appids, start=1):
        if appid in done:
            print(f"[{idx}/{total}] appid={appid} already done, skip.")
            continue

        row_app = apps_target_indexed.loc[appid]
        name = str(row_app.get("name", "")).strip()
        if not name:
            print(f"[{idx}/{total}] appid={appid} missing name, skip.")
            continue

        short_desc = str(row_app.get("short_description", "")) if has_desc else ""
        
        genres = row_app.get("genres", None)
        genres_text = to_genres_text(genres)

        print(f"[{idx}/{total}] Generating aliases for {appid} - {name}")

        try:
            aliases = gen.generate_aliases(
                name=name,
                short_description=short_desc,
                genres=genres_text,
                extra="",
                k=alias_k,
            )
        except Exception as e:
            print("  Error during alias generation:", e)
            aliases = []

        out_row = {
            "appid": int(appid),
            "name": name,
            "aliases": aliases,                              
            "alias_text": "; ".join(aliases) if aliases else "", 
            "last_updated": datetime.utcnow(),
        }
        rows.append(out_row)
        done.add(appid)

        

        time.sleep(sleep_sec)

    final_df = pd.DataFrame(rows)
    final_df.to_parquet(ALIASES_PATH, index=False)
    print("All done. Total aliases rows:", len(final_df))
    print("Saved to:", ALIASES_PATH)
    return final_df


if __name__ == "__main__":
    df = build_app_aliases(
        top_n_apps=30,
        min_review_count=50,
        model_name=MODEL_NAME,
        use_4bit=False,
        alias_k=6,
    )
    print(df.head())

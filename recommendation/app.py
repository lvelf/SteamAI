import os
import sys
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import encode_text, df_to_json_records, parse_filters_from_request
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/Website
PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final


DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data


RECOMMENDATION_DIR = os.path.join(PROJECT_ROOT, "recommendation")
sys.path.append(RECOMMENDATION_DIR)
from steam_recommender import SteamRecommender

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "Website", "templates")
STATIC_DIR   = os.path.join(PROJECT_ROOT, "Website", "static")
PROCESSED = os.path.join(DATA_BASE, "processed")
print("TEMPLATE_DIR =", TEMPLATE_DIR)
print("STATIC_DIR   =", STATIC_DIR)


print("DATA_BASE =", DATA_BASE)

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

apps_parquet = os.path.join(PROCESSED, "apps_with_stats.parquet")
emb_parquet = os.path.join(PROCESSED, "apps_embeddings.parquet")

"""
rec = SteamRecommender(
    apps_csv_path=os.path.join(
        DATA_BASE,
        "archive/steam_dataset_2025_csv_package_v1/steam_dataset_2025_csv/applications.csv",
    ),
    emb_npy_path=os.path.join(
        DATA_BASE,
        "archive/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embeddings.npy",
    ),
    emb_map_csv_path=os.path.join(
        DATA_BASE,
        "archive/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embedding_map.csv",
    ),
)
"""
rec = SteamRecommender(apps_parquet, emb_parquet)

print("Loading text encoder (BAAI/bge-m3)...")
text_encoder = SentenceTransformer("BAAI/bge-m3")
print("Text encoder loaded.")

# ---------- Web Route ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend")
def recommend_page():
    return render_template("recommend.html")

@app.route("/graph")
def graph_page():
    return render_template("graph.html")

@app.route("/prompt")
def prompt_page():
    return render_template("prompt.html")


# ---------- API ----------

@app.route("/api/search_game")
def api_search_game():
    name = request.args.get("name", "")
    matches = rec.find_appid_by_name(name, top_k=5)
    return jsonify({"matches": matches})

@app.route("/api/recommend")
def api_recommend():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "Missing game name"}), 400

    print("recommending for:", name)

    # fuzz find app_id 
    candidates = rec.find_appid_by_name(name, top_k=3)
    if not candidates:
        return jsonify({"error": f"No game found for '{name}'"}), 404

    seed = candidates[0]
    print("seed game:", seed)
    # filter
    #filter_kwargs = parse_filters_from_request(request)
    
    df = rec.recommend_similar(
        seed["appid"],
        top_k=10,
        # **filter_kwargs,
    )
    df_clean = df.where(pd.notnull(df), None)
    df_clean = df.replace({np.nan: None})
    recs = df_clean.to_dict(orient="records")
    
    print(df.head(10))
    js_ = jsonify({
        "center": seed,
        "recommendations": recs,
    })
    return js_

@app.route("/api/graph")
def api_graph():
    appid = request.args.get("appid")
    name = request.args.get("name")
    if not name:
        return jsonify({"error": "Missing game name"}), 400
    
    candidates = rec.find_appid_by_name(name, top_k=3)
    if not candidates:
        return jsonify({"error": f"No game found for '{name}'"}), 404


    seed = candidates[0]
    center_name = seed
    appid = seed["appid"]
    
    print("seed game:", seed)
    df = rec.recommend_similar(
        seed["appid"],
        top_k=10,
    )
    df_clean = df.where(pd.notnull(df), None)
    df_clean = df.replace({np.nan: None})
    recs = df_clean.to_dict(orient="records")

    
    nodes = [{
        "id": str(appid),
        "label": center_name,
        "group": 0, 
    }]
    links = []

    
    rec_appids = []

    for r in recs:
        nid = str(r["appid"])
        rec_appids.append(r["appid"])
        nodes.append({
            "id": nid,
            "label": r["name"],
            "group": 1, 
        })
       
        links.append({
            "source": str(appid),
            "target": nid,
            "value": float(r["similarity"]),
        })

   
    if rec_appids:
        rec_indices = [rec.appid2idx[a] for a in rec_appids]
        sub_emb = rec.emb_norm[rec_indices]
        sim_mat = cosine_similarity(sub_emb, sub_emb)

        n = len(rec_appids)
        SIM_THRESHOLD = 0.7 

        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim_mat[i, j])
                if s >= SIM_THRESHOLD:
                    links.append({
                        "source": str(rec_appids[i]),
                        "target": str(rec_appids[j]),
                        "value": s,
                    })

    return jsonify({"nodes": nodes, "links": links, "center": {"appid": appid, "name": center_name}})

@app.route("/api/prompt_recommend", methods=["POST"])
def api_prompt_recommend():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    mode = "semantic"
    center = None
    df = None

    if text.isdigit():
        appid = int(text)
        row = rec.apps[rec.apps["appid"] == appid]
        if not row.empty:
            mode = "appid"
            center = {
                "appid": appid,
                "name": row["name"].iloc[0],
            }
            df = rec.recommend_similar(appid, top_k=10)

   
    if df is None:
        exact = rec.apps[rec.apps["name"].str.casefold() == text.casefold()]
        if not exact.empty:
            mode = "name"
            appid = int(exact["appid"].iloc[0])
            center = {
                "appid": appid,
                "name": exact["name"].iloc[0],
            }
            df = rec.recommend_similar(appid, top_k=10)

    
    if df is None:
        mode = "semantic"
        vec = encode_text(text_encoder, text)         # (1, 1024)
        df = rec.search_by_vector(vec, top_k=10)
    
    records = df_to_json_records(df)
    
    return_json = jsonify({
        "mode": mode,
        "center": center,
        "recommendations": records,
    })
    
    print(records)
    return return_json



if __name__ == "__main__":
    app.run(debug=True)
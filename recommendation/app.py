# Website/app.py
import os
import sys
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/Website
PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final


DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data


RECOMMENDATION_DIR = os.path.join(PROJECT_ROOT, "recommendation")
sys.path.append(RECOMMENDATION_DIR)
from steam_recommender import SteamRecommender

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "Website", "templates")
STATIC_DIR   = os.path.join(PROJECT_ROOT, "Website", "static")
print("TEMPLATE_DIR =", TEMPLATE_DIR)
print("STATIC_DIR   =", STATIC_DIR)


print("DATA_BASE =", DATA_BASE)

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

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


# ---------- API ----------

@app.route("/api/search_game")
def api_search_game():
    name = request.args.get("name", "")
    matches = rec.find_appid_by_name(name, top_k=5)
    return jsonify({"matches": matches})

@app.route("/api/recommend")
def api_recommend():
    appid = request.args.get("appid")
    name = request.args.get("name")

    if appid is None:
        if not name:
            return jsonify({"error": "appid or name is required"}), 400

        matches = rec.find_appid_by_name(name, top_k=1)
        if not matches:
            return jsonify({"error": f"No game found for name '{name}'"}), 404

        appid = matches[0]["appid"]
        center_name = matches[0]["name"]
    else:
        appid = int(appid)
        row = rec.apps[rec.apps["appid"] == appid]
        center_name = row["name"].iloc[0] if not row.empty else str(appid)

    df = rec.recommend_similar(appid, top_k=10)
    data = df.to_dict(orient="records")

    return jsonify({
        "center": {"appid": appid, "name": center_name},
        "recommendations": data,
    })

@app.route("/api/graph")
def api_graph():
    appid = request.args.get("appid")
    name = request.args.get("name")

    
    if appid is None:
        if not name:
            return jsonify({"error": "appid or name is required"}), 400

        matches = rec.find_appid_by_name(name, top_k=1)
        if not matches:
            return jsonify({"error": f"No game found for name '{name}'"}), 404

        appid = matches[0]["appid"]
        center_name = matches[0]["name"]
    else:
        appid = int(appid)
        row = rec.apps[rec.apps["appid"] == appid]
        center_name = row["name"].iloc[0] if not row.empty else str(appid)

    
    df = rec.recommend_similar(appid, top_k=10)
    recs = df.to_dict(orient="records")

    
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
        rec_indices = [rec.app2idx[a] for a in rec_appids]
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

if __name__ == "__main__":
    app.run(debug=True)
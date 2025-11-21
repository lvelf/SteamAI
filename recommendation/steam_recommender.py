import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

class SteamRecommender:
    def __init__(
        self,
        apps_csv_path: str,
        emb_npy_path: str,
        emb_map_csv_path: str,
    ):
        print("Loading applications...")
        self.apps = pd.read_csv(apps_csv_path)  # applications.csv
        print(self.apps.columns)
        print("applications shape:", self.apps.shape)
        
        print("Loading embedding map...")
        emb_map = pd.read_csv(emb_map_csv_path)
        print(emb_map.columns)

        print("Loading embeddings as raw float32 binary...")

        raw = np.fromfile(emb_npy_path, dtype=np.float32)
        n_rows = len(emb_map)    
        dim = 1024               

        expected_size = n_rows * dim
        assert raw.size == expected_size, (
										f"Embedding file size mismatch: got {raw.size}, "
										f"expected {expected_size} (= {n_rows} * {dim})"
										)

        emb = raw.reshape((n_rows, dim))
        print("Embeddings shape:", emb.shape)

        self.emb_norm = normalize(emb)

        self.app2idx = dict(zip(emb_map["appid"], emb_map["vector_index"]))
        self.apps = self.apps.merge(
            emb_map[["appid", "vector_index"]],
            on="appid",
            how="inner",
        )

        print(f"SteamRecommender initialized. {len(self.apps)} games loaded.")
        
    def recommend_similar(self, appid: int, top_k: int = 10):
        if appid not in self.app2idx:
            raise ValueError(f"appid {appid} not found in embedding map")

        idx = self.app2idx[appid]
        v = self.emb_norm[idx : idx + 1]

        sims = cosine_similarity(v, self.emb_norm)[0]
        sims[idx] = -1 

        top_idx = sims.argsort()[::-1][:top_k]

        
        top_apps = self.apps.iloc[top_idx].copy()
        top_apps["similarity"] = sims[top_idx]

        
        return top_apps[["appid", "name", "similarity"]]

    
    def find_appid_by_name(self, name: str, top_k: int = 5):
        name = name.strip().lower()
        mask = self.apps["name"].str.lower().str.contains(name, na=False)
        results = self.apps.loc[mask, ["appid", "name"]].drop_duplicates()

        if results.empty:
            return []

        return results.head(top_k).to_dict(orient="records")
    
    def search_by_vector(self, vec, top_k: int = 10):

        if vec.ndim == 1:
            v = vec.reshape(1, -1)
        else:
            v = vec

        sims = cosine_similarity(v, self.emb_norm)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        results = self.apps.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results[["appid", "name", "short_description", "similarity"]]
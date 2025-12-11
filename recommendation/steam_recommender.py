import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Any


class SteamRecommender:
    """
    1) apps_parquet:
       
         - appid (string/int)
         - name
         - short_description
         - type
         - genres (array<string>)
         - release_date (int)
         - is_free
         - mat_final_price
         - positive_ratio (0~1)
         - review_count (int)
         - name_normalized 
    2) emb_parquet:     
         - appid (string)
         - vector_index (int)
         - embedding (array<float>)
    """
    
    def __init__(
        self,
        apps_parquet_path: str,
        emb_parquet_path: str,
    ):
        
        print("Loading apps parquet...")
        apps = pd.read_parquet(apps_parquet_path)
        print("apps columns:", apps.columns)
        print("apps shape:", apps.shape)

        print("Loading embeddings parquet...")
        emb_df = pd.read_parquet(emb_parquet_path)
        print("embeddings columns:", emb_df.columns)
        print("embeddings shape:", emb_df.shape)

        
         #  appid -> string
        apps["appid"] = apps["appid"].astype(str)
        emb_df["appid"] = emb_df["appid"].astype(str)
        
        merged = apps.merge(emb_df[["appid", "embedding"]], on="appid", how="inner")
        print("apps with embeddings:", merged.shape)
        
        # delete DLC / demo
        if "type" in merged.columns:
            merged = merged[merged["type"] == "game"].copy()
        
        # genres -> List[str]
        if "genres" in merged.columns:
            def _normalize_genres(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return [str(g).strip() for g in x if str(g).strip()]

                # None / NaN
                if x is None:
                    return []
                try:
                    if pd.isna(x):
                        return []
                except Exception:
                    pass

                # String
                s = str(x).strip()
                if not s:
                    return []
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
                return [p for p in parts if p]

            merged["genres"] = merged["genres"].apply(_normalize_genres)
        else:
            merged["genres"] = [[] for _ in range(len(merged))]
            
        self.apps = merged.reset_index(drop=True)
        
        print("Stacking embedding matrix...")
        emb_matrix = np.stack(self.apps["embedding"].to_numpy())
        print("Embedding matrix shape:", emb_matrix.shape)
        
        self.emb_norm = normalize(emb_matrix)
        
        # appid -> line idx
        self.appid2idx: Dict[str, int] = {
            appid: i for i, appid in enumerate(self.apps["appid"])
        }

        print(f"SteamRecommenderParquet initialized with {len(self.apps)} games.")

    def _apply_filters(
        self,
        df: pd.DataFrame,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[List[str]] = None,
        is_free: Optional[bool] = None,
        min_positive_ratio: Optional[float] = None,
        min_review_count: Optional[int] = None,
        max_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """ Filter DF """

        out = df

        if min_year is not None and "release_date" in out.columns:
            out = out[out["release_date"].fillna(0) >= min_year]

        if max_year is not None and "release_date" in out.columns:
            out = out[out["release_date"].fillna(9999) <= max_year]

        if genres:
            gset = {g.lower() for g in genres}

            def _match_genres(gs):
                if not isinstance(gs, (list, tuple, set)):
                    return False
                lower = {str(x).lower() for x in gs}
                return bool(lower & gset)

            out = out[out["genres"].apply(_match_genres)]

        if is_free is not None and "is_free" in out.columns:
            
            def _to_bool(x):
                if isinstance(x, bool):
                    return x
                if pd.isna(x):
                    return False
                s = str(x).strip().lower()
                if s in ("true", "1", "t", "yes"):
                    return True
                if s in ("false", "0", "f", "no"):
                    return False
                return False

            out = out[out["is_free"].apply(_to_bool) == bool(is_free)]

        if min_positive_ratio is not None and "positive_ratio" in out.columns:
            out = out[out["positive_ratio"].fillna(0.0) >= min_positive_ratio]

        if min_review_count is not None and "review_count" in out.columns:
            out = out[out["review_count"].fillna(0) >= min_review_count]

        if max_price is not None and "mat_final_price" in out.columns:
            out = out[out["mat_final_price"].fillna(float("inf")) <= max_price]

        return out
    
    def find_appid_by_name(self, name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fuzz Search
        return: [{"appid": ..., "name": ...}, ...]
        """
        q = name.strip().lower()
        if not q:
            return []

        apps = self.apps

        masks = []

        if "name_normalized" in apps.columns:
            masks.append(
                apps["name_normalized"]
                .fillna("")
                .str.lower()
                .str.contains(q, na=False)
            )

        masks.append(
            apps["name"]
            .fillna("")
            .str.lower()
            .str.contains(q, na=False)
        )

        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m

        results = apps.loc[mask, ["appid", "name"]].drop_duplicates()

        if not results.empty:
            return results.head(top_k).to_dict(orient="records")

        # contains miss → fallback fuzz search
        series = (
            apps["name_normalized"]
            if "name_normalized" in apps.columns
            else apps["name"]
        ).fillna("")

        def sim(a, b):
            return SequenceMatcher(None, a, b).ratio()

        scores = series.str.lower().apply(lambda x: sim(q, x))
        top_idx = scores.sort_values(ascending=False).head(top_k).index

        fuzzy_results = apps.loc[top_idx, ["appid", "name"]].drop_duplicates()
        return fuzzy_results.to_dict(orient="records")
    
    def recommend_similar(
        self,
        appid: str,
        top_k: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[List[str]] = None,
        is_free: Optional[bool] = None,
        min_positive_ratio: Optional[float] = None,
        min_review_count: Optional[int] = None,
        max_price: Optional[float] = None,
        use_multi_modal_score: bool = False,
    ) -> pd.DataFrame:
        """
        use appid to recommend:
        - Base: embedding cos similarity
        - Optional: rating score, review amounts multi-modal
        """
        appid_str = str(appid)
        if appid_str not in self.appid2idx:
            raise ValueError(f"appid {appid} not found in embeddings")

        idx = self.appid2idx[appid_str]
        v = self.emb_norm[idx : idx + 1]

        sims = cosine_similarity(v, self.emb_norm)[0]
        sims[idx] = -1.0  # delete self

        # add similarity to DataFrame
        df = self.apps.copy()
        df["similarity"] = sims

        # filter then rank
        df = self._apply_filters(
            df,
            min_year=min_year,
            max_year=max_year,
            genres=genres,
            is_free=is_free,
            min_positive_ratio=min_positive_ratio,
            min_review_count=min_review_count,
            max_price=max_price,
        )
        
        if df.empty:
            cols = [
                "appid",
                "name",
                "short_description",
                "similarity",
                "positive_ratio",
                "review_count",
                "release_date",
                "genres",
                "score",
            ]
            cols = [c for c in cols if c in df.columns or c in ["score"]]
            empty = pd.DataFrame(columns=cols)
            return empty


        #  use multi-modal finetune score
        if use_multi_modal_score:
            df = self._apply_multi_modal_score(df)
            sort_col = "score"
        else:
            sort_col = "similarity"

        df = df.sort_values(sort_col, ascending=False).head(top_k)

        cols = [
            "appid",
            "name",
            "short_description",
            "similarity",
        ]
        for c in ["positive_ratio", "review_count", "release_date", "genres", "score"]:
            if c in df.columns:
                cols.append(c)

        return df[cols]
    
    def _apply_multi_modal_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add similarity and other metrics to a score。
        """
        out = df.copy()
        
        if out.empty:
            out["score"] = pd.Series(dtype=float)
            return out

        sim = out["similarity"].values
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

        # score
        if "positive_ratio" in out.columns:
            pr = out["positive_ratio"].fillna(0.5).values
        else:
            pr = np.full_like(sim_norm, 0.5)

        # review count
        if "review_count" in out.columns:
            rc = np.log1p(out["review_count"].fillna(0).values)
            rc_norm = (rc - rc.min()) / (rc.max() - rc.min() + 1e-8)
        else:
            rc_norm = np.ones_like(sim_norm) * 0.5

        # multi-modal scoring
        score = 0.6 * sim_norm + 0.25 * pr + 0.15 * rc_norm
        out["score"] = score

        return out

    def search_by_vector(
        self,
        vec: np.ndarray,
        top_k: int = 10,
        **filter_kwargs,
    ) -> pd.DataFrame:
        """
        given an embedding vector
        search game
        """
        if vec.ndim == 1:
            v = vec.reshape(1, -1)
        else:
            v = vec

        sims = cosine_similarity(v, self.emb_norm)[0]
        df = self.apps.copy()
        df["similarity"] = sims

        df = self._apply_filters(df, **filter_kwargs)
        df = self._apply_multi_modal_score(df)

        df = df.sort_values("score", ascending=False).head(top_k)

        cols = [
            "appid",
            "name",
            "short_description",
            "similarity",
            "score",
        ]
        for c in ["positive_ratio", "review_count", "release_date", "genres"]:
            if c in df.columns:
                cols.append(c)

        return df[cols]
    
    def recommend_by_name(
        self,
        name: str,
        top_k_candidates: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """
        use name to recommend
        """
        
        candidates = self.find_appid_by_name(name, top_k=top_k_candidates)
        if not candidates:
            raise ValueError(f"No game found for name '{name}'")
        
        print(f"candidates are: {candidates}")

        seed = candidates[0]
        print(f"Using seed game: {seed['name']} (appid={seed['appid']})")
        return self.recommend_similar(seed["appid"], **kwargs)
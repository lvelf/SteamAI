from steam_recommender import SteamRecommender

base = "../../data/archive"  


rec = SteamRecommender(
    apps_csv_path=f"{base}/steam_dataset_2025_csv_package_v1/steam_dataset_2025_csv/applications.csv",
    emb_npy_path=f"{base}/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embeddings.npy",
    emb_map_csv_path=f"{base}/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embedding_map.csv",
)
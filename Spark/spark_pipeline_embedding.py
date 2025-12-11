import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType
from pyspark.sql.functions import pandas_udf
from spark_pipeline_apps import create_spark
import pandas as pd
from tqdm import tqdm
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def build_embeddings_table(spark, embeddings_base_dir, out_path,  batch_size = 5000):
    """
    read applications_embedding_map.csv + applications_embeddings.npy
    generate [appid, vector_index, embedding] parquet
    """
    
    map_path = os.path.join(
        embeddings_base_dir,
        "applications_embedding_map.csv",
    )
    npy_path = os.path.join(
        embeddings_base_dir,
        "applications_embeddings.npy",
    )
    
    print("Reading embedding map (pandas) from:", map_path)
    
    emb_map_pd = pd.read_csv(map_path)
    emb_map_pd["appid"] = emb_map_pd["appid"].astype(str)
    emb_map_pd["vector_index"] = emb_map_pd["vector_index"].astype(int)

    n_rows = len(emb_map_pd)
    print("Number of rows in embedding map:", n_rows)
    
    print("Loading embeddings .npy from:", npy_path)
    raw = np.fromfile(npy_path, dtype=np.float32)
    
    dim = 1024
    
    expected_size = n_rows * dim
    if raw.size != expected_size:
        raise ValueError(
            f"Embedding file size mismatch: got {raw.size}, "
            f"expected {expected_size} (= {n_rows} * {dim})"
        )

    emb_matrix = raw.reshape((n_rows, dim))
    print("Embeddings matrix shape:", emb_matrix.shape)
    
    if os.path.exists(out_path):
        print("Removing existing parquet output:", out_path)
        pass
    
    print(f"Writing embeddings to parquet in batches of {batch_size} rows...")
    first = True

    for start in tqdm(range(0, n_rows, batch_size), desc="Batches"):
        end = min(start + batch_size, n_rows)

        # map + embedding
        batch_map = emb_map_pd.iloc[start:end].copy()
        
        batch_emb_list = [
            row.tolist() for row in emb_matrix[start:end]
        ]
        batch_map["embedding"] = batch_emb_list

        # pandas -> Spark DataFrame
        sdf = spark.createDataFrame(batch_map)

        mode = "overwrite" if first else "append"
        first = False

        (
            sdf.write
            .mode(mode)
            .parquet(out_path)
        )

    print("Done writing embeddings parquet to:", out_path)

if __name__ == "__main__":
    spark = create_spark()

   
    THIS_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.dirname(THIS_DIR)
    FINAL_ROOT = os.path.dirname(PROJECT_ROOT)  # .../Documents/Bigdata/Final_Proj
    DATA_BASE = os.path.join(FINAL_ROOT, "data")

    embeddings_base_dir = os.path.join(
        DATA_BASE,
        "archive/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings",
    )
    processed_dir = os.path.join(DATA_BASE, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    embeddings_parquet_path = os.path.join(processed_dir, "apps_embeddings.parquet")

    build_embeddings_table(
        spark,
        embeddings_base_dir=embeddings_base_dir,
        out_path=embeddings_parquet_path,
    )

    spark.stop()
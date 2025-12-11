from pyspark.sql import SparkSession
import os
spark = SparkSession.builder.master("local[*]").appName("check").getOrCreate()

THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/Website
PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final
DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data
PROCESSED_PATh = os.path.join(DATA_BASE, "processed")
app_clean_path = os.path.join(PROCESSED_PATh, "apps_clean.parquet")
app_with_stats_path = os.path.join(PROCESSED_PATh, "apps_with_stats.parquet")

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
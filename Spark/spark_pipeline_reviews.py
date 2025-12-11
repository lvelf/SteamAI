# spark_pipeline_reviews.py
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, BooleanType, LongType


THIS_DIR = os.path.dirname(__file__)           # .../SteamAI/Spark
PROJECT_ROOT = os.path.dirname(THIS_DIR)       # .../SteamAI
FINAL_ROOT = os.path.dirname(PROJECT_ROOT)     # .../Final
DATA_BASE = os.path.join(FINAL_ROOT, "data")   # .../Final/data

def create_spark(app_name="SteamAI-Reviews"):
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

def main():
    spark = create_spark()

    csv_dir = os.path.join(
        DATA_BASE,
        "archive/steam_dataset_2025_csv_package_v1/steam_dataset_2025_csv/",
    )
    input_path = os.path.join(csv_dir, "reviews.csv")
    processed_dir = os.path.join(DATA_BASE, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "reviews.parquet")

    print("Loading reviews.csv ...")
    df = (
        spark.read
        .option("header", True)
        .option("escape", "\"")
        .option("multiLine", True)
        .csv(input_path)
    )

    df = (
        df.select(
            "recommendationid",
            "appid",
            "author_steamid",
            "author_num_games_owned",
            "author_num_reviews",
            "author_playtime_forever",
            "author_playtime_last_two_weeks",
            "author_playtime_at_review",
            "author_last_played",
            "language",
            "review_text",
            "timestamp_created",
            "timestamp_updated",
            "voted_up",
            "votes_up",
            "votes_funny",
            "weighted_vote_score",
            "comment_count",
            "steam_purchase",
            "received_for_free",
            "written_during_early_access",
            "created_at",
            "updated_at",
        )
        .withColumn("appid", col("appid").cast(IntegerType()))
        .withColumn("votes_up", col("votes_up").cast(LongType()))
        .withColumn("votes_funny", col("votes_funny").cast(LongType()))
        .withColumn("comment_count", col("comment_count").cast(LongType()))
        .withColumn("timestamp_created", col("timestamp_created").cast(LongType()))
        .withColumn("timestamp_updated", col("timestamp_updated").cast(LongType()))
    )

    print("Writing parquet to:", out_path)
    df.write.mode("overwrite").parquet(out_path)
    spark.stop()
    print("Done.")

if __name__ == "__main__":
    main()

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, to_date, year,
    collect_set, when, col, count, avg, regexp_extract
)


def create_spark(app_name="SteamAI-Spark"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")   
        .config("spark.driver.memory", "12g")    
        .config("spark.executor.memory", "12g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    
    return spark

# Clean application table
def build_apps_table(spark, csv_base_dir, out_path):
    """
    Read applications.csv + application_genres.csv + genres.csv
    to generate basic + genres table
    
    appid,name,type,is_free,release_date,required_age,short_description,supported_languages,header_image,background,metacritic_score,recommendations_total,mat_supports_windows,mat_supports_mac,mat_supports_linux,mat_initial_price,mat_final_price,mat_discount_percent,mat_currency,mat_achievement_count,mat_pc_os_min,mat_pc_processor_min,mat_pc_memory_min,mat_pc_graphics_min,mat_pc_os_rec,mat_pc_processor_rec,mat_pc_memory_rec,mat_pc_graphics_rec,created_at,updated_at

    """
    apps_path = os.path.join(csv_base_dir, "applications.csv")
    app_genres_path = os.path.join(csv_base_dir, "application_genres.csv")
    genres_path = os.path.join(csv_base_dir, "genres.csv")
    
    print("Reading applications.csv ...")
    apps_raw = spark.read.csv(apps_path, header=True, inferSchema=False)
    
    # "appid", "genre_id"
    print("Reading application_genres.csv ...")
    app_genres = spark.read.csv(app_genres_path, header=True, inferSchema=False)

    print("Reading genres.csv ...")
    genres = spark.read.csv(genres_path, header=True, inferSchema=False)
    # genres.csv: "id", "name"
    genres_renamed = genres.withColumnRenamed("id", "genre_id")
    
    # app_genres: (appid, genre_id)
    appid_with_genres = (
        app_genres
        .join(genres_renamed, on="genre_id", how="left")
        .groupBy("appid")
        .agg(collect_set("name").alias("genres"))   # genres: array<string>
    )
    
    apps_joined = (
        apps_raw.join(appid_with_genres, on="appid", how="left")
    )

    apps = apps_joined.select(
        col("appid"),

        col("name"),
        col("type"),
        col("is_free"),
        col("release_date"),
        col("required_age"),
        col("short_description"),
        col("supported_languages"),
        col("header_image"),
        col("background"),
        col("metacritic_score"),
        col("recommendations_total"),
        col("mat_supports_windows"),
        col("mat_supports_mac"),
        col("mat_supports_linux"),
        col("mat_initial_price"),
        col("mat_final_price"),
        col("mat_discount_percent"),
        col("mat_currency"),
        col("mat_achievement_count"),
        col("mat_pc_os_min"),
        col("mat_pc_processor_min"),
        col("mat_pc_memory_min"),
        col("mat_pc_graphics_min"),
        col("mat_pc_os_rec"),
        col("mat_pc_processor_rec"),
        col("mat_pc_memory_rec"),
        col("mat_pc_graphics_rec"),
        col("created_at"),
        col("updated_at"),

        col("genres"),  # add genre
    )
    
    # realease date
    apps = apps.withColumn(
        "release_year_str",
        regexp_extract(col("release_date"), r"(\\d{4})", 1)
    )
    
    apps = apps.withColumn(
        "release_year",
        when(col("release_year_str") != "", col("release_year_str").cast("int"))
    )

    apps = apps.drop("release_year_str")
    
    apps = apps.withColumn(
        "name_normalized",
        trim(
            lower(
                regexp_replace(col("name"), r"[^a-zA-Z0-9]+", " ")
            )
        )
    )
    
    print("Writing apps_clean to:", out_path)
    apps.write.mode("overwrite").parquet(out_path)
   
   

def add_review_stats(spark, csv_base_dir, apps_parquet_path, out_path):
    reviews_path = os.path.join(csv_base_dir, "reviews.csv")
    print("Reading reviews.csv ...")

  
    reviews = spark.read.csv(reviews_path, header=True, inferSchema=False)

   
    voted_str = lower(col("voted_up").cast("string"))

    # True -> 1.0,  False/ other -> 0.0
    positive_flag = when(voted_str == "true", 1.0).otherwise(0.0)

    reviews = reviews.withColumn("positive_flag", positive_flag)

    # 3. with appid group
    #    - review_count
    #    - positive_ratio
    review_stats = (
        reviews
        .groupBy("appid")  
        .agg(
            count("*").alias("review_count"),
            avg(col("positive_flag")).alias("positive_ratio")
        )
    )

    print("Reading apps_clean parquet ...")
    apps = spark.read.parquet(apps_parquet_path)

    apps_with_stats = apps.join(review_stats, on="appid", how="left")

    print("Writing apps_with_stats to:", out_path)
    apps_with_stats.write.mode("overwrite").parquet(out_path)


if __name__ == "__main__":
    
    THIS_DIR = os.path.dirname(__file__)                 # .../Final/SteamAI/Website
    PROJECT_ROOT = os.path.dirname(THIS_DIR)             # .../Final/SteamAI
    FINAL_ROOT = os.path.dirname(PROJECT_ROOT)           # .../Final
    DATA_BASE = os.path.join(FINAL_ROOT, "data")         # .../Final/data
    
    
    spark = create_spark()

    csv_dir = os.path.join(
        DATA_BASE,
        "archive/steam_dataset_2025_csv_package_v1/steam_dataset_2025_csv/",
    )
    processed_dir = os.path.join(DATA_BASE, "processed/")
    os.makedirs(processed_dir, exist_ok=True)

    apps_clean_path = os.path.join(processed_dir, "apps_clean.parquet")
    apps_with_stats_path = os.path.join(processed_dir, "apps_with_stats.parquet")

    build_apps_table(spark, csv_dir, apps_clean_path)
    add_review_stats(spark, csv_dir, apps_clean_path, apps_with_stats_path)

    spark.stop()
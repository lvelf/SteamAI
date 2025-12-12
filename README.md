Create environment
```
conda create -n SteamAI

conda activate SteamAI
```

Install Env
```
python -m pip install numpy pandas scikit-learn

python -m pip install "sentence-transformers==3.0.1" "torch>=2.3.0"

sudo apt install openjdk-17-jre-headless

pip install pyspark

python -m pip install pyarrow

pip install "accelerate>=0.31.0"

pip install flask

```

Run code
```
cd SteamAI/recommendation/

python app.py
```

go to website http://127.0.0.1:5000/graph

airflow

```
(SteamAI) nuo@Nuo:~/SteamAI/SteamAI$ airflow config get-value core dags_folder
/home/nuo/airflow/dags
(SteamAI) nuo@Nuo:~/SteamAI/SteamAI$ airflow config get-value core airflow_home
The option [core/airflow_home] is not found in config.
(SteamAI) nuo@Nuo:~/SteamAI/SteamAI$ mkdir -p /home/nuo/airflow/dags
(SteamAI) nuo@Nuo:~/SteamAI/SteamAI$ ln -sf /home/nuo/SteamAI/SteamAI/dags/steam_reviews_crawl_raw.py /home/nuo/airflow/dags/steam_reviews_crawl_raw.py

airflow variables set STEAMAI_REVIEWS_DB /home/nuo/SteamAI/SteamAI/steam_reviews_raw.sqlite

airflow pools set steam_api_pool 5 "Limit Steam API concurrency"

check

airflow pools list | grep steam_api_pool


```
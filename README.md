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

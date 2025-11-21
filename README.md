Create environment
```
conda create -n SteamAI

conda activate SteamAI
```

Install Env
```
conda install numpy pandas sklearn-kit

python -m pip install "sentence-transformers==3.0.1" "torch>=2.3.0"

```

Run code
```
cd SteamAI/recommendation/

python app.py
```

go to website http://127.0.0.1:5000/graph

import os
import sys
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def encode_text(text_encoder, text: str) -> np.ndarray:
   
    emb = text_encoder.encode([text], normalize_embeddings=True)
    return emb 

import numpy as np
import math

def df_to_json_records(df):
    
    df = df.copy()
    
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.floating):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(np.nan)  

    
    df = df.replace({np.nan: None})

    #
    records = df.to_dict(orient="records")

    for r in records:
        if "short_description" in r and r["short_description"] is None:
            r["short_description"] = ""
            
    return records

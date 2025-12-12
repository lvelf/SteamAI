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

def parse_filters_from_request(req):
    """Parse filter parameters from request.args"""
    def _get_int(name):
        v = req.args.get(name)
        if v is None or v == "" or v.lower() == "null":
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def _get_float(name):
        v = req.args.get(name)
        if v is None or v == "" or v.lower() == "null":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _get_bool(name):
        v = req.args.get(name)
        if v is None or v == "":
            return None
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        return None

    # genres : "Action,Adventure,RPG"
    genres_raw = req.args.get("genres")
    if genres_raw:
        genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
    else:
        genres = None

    return dict(
        min_year=_get_int("min_year"),
        max_year=_get_int("max_year"),
        genres=genres,
        is_free=_get_bool("is_free"),
        min_positive_ratio=_get_float("min_positive_ratio"),
        min_review_count=_get_int("min_review_count"),
        max_price=_get_float("max_price"),
    )


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
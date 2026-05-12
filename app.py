import math
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

TEXT_MODEL_PATH = MODELS_DIR / "text_model.pkl"
URL_MODEL_PATH = MODELS_DIR / "url_model.pkl"
URL_FEATURES_PATH = MODELS_DIR / "url_features.pkl"

text_model = joblib.load(TEXT_MODEL_PATH)
url_model = joblib.load(URL_MODEL_PATH)
url_features = joblib.load(URL_FEATURES_PATH)

app = FastAPI(title="KindKlick Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
    threshold: float = 0.55


class UrlRequest(BaseModel):
    url: str
    threshold: float = 0.60


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    suspicious_words = [
        "login", "verify", "account", "secure", "update",
        "bank", "signin", "confirm", "password", "wallet",
        "payment", "bonus", "free", "claim"
    ]

    suspicious_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".gq"]
    shorteners = [
        "bit.ly", "tinyurl.com", "t.co", "goo.gl",
        "is.gd", "buff.ly", "ow.ly", "rb.gy"
    ]

    probs = [url.count(c) / len(url) for c in set(url)] if url else [1]

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "slash_count": url.count("/"),
        "digit_count": sum(c.isdigit() for c in url),
        "special_char_count": len(re.findall(r"[@#&=%]", url)),
        "subdomain_count": max(hostname.count(".") - 1, 0),
        "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", hostname))),
        "has_https": int(parsed.scheme == "https"),
        "has_at_symbol": int("@" in url),
        "has_double_slash_path": int("//" in path),
        "has_suspicious_tld": int(any(hostname.endswith(tld) for tld in suspicious_tlds)),
        "has_shortener": int(any(short in hostname for short in shorteners)),
        "has_suspicious_word": int(any(word in url for word in suspicious_words)),
        "entropy": -sum(p * math.log2(p) for p in probs),
    }

    return pd.DataFrame([[features.get(col, 0) for col in url_features]], columns=url_features)


@app.get("/")
def root():
    return {"message": "KindKlick Model API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze/text")
def analyze_text(request: TextRequest):
    cleaned = clean_text(request.text)

    probs = text_model.predict_proba([cleaned])[0]
    labels = list(text_model.classes_)

    top_index = int(probs.argmax())
    top_label = labels[top_index]
    top_score = float(probs[top_index])

    if top_score < request.threshold:
        result = "needs_review"
    else:
        result = top_label

    return {
        "result": result,
        "top_label": top_label,
        "confidence": round(top_score, 4),
        "scores": {
            labels[i]: round(float(probs[i]), 4) for i in range(len(labels))
        }
    }


@app.post("/api/analyze/url")
def analyze_url(request: UrlRequest):
    feature_row = extract_url_features(request.url)
    phishing_probability = float(url_model.predict_proba(feature_row)[0][1])

    if phishing_probability >= request.threshold:
        result = "phishing"
    else:
        result = "safe"

    return {
        "url": request.url,
        "result": result,
        "phishing_probability": round(phishing_probability, 4)
    }

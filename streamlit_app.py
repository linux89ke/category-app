"""
Product Category Predictor
Strategy : Fast TF-IDF shortlist + async parallel AI reranking + Smart Caching
Speed    : Instant loading + 2-5s for batch processing
Cost     : 1 API call per product (0 calls if cached)
"""

import os
import io
import json
import asyncio
import pickle
import difflib
import time
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq, Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CACHE_FILE = "prediction_cache.json"
INDEX_FILE = "category_index.pkl"
CACHE_VERSION = "v2"

st.set_page_config(page_title="Product Category Predictor", layout="wide")

# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────

def path_to_doc(path: str) -> str:
    parts = path.split(" / ")
    return " ".join(parts) + " " + " ".join(parts[-2:]) * 2


# ─────────────────────────────────────────────────────────────
# INDEX LOADER (FIXED)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing fast category index...")
def load_or_build_index(file_path: str):
    # ── Try load cache ─────────────────────────
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                version, data = pickle.load(f)

            if version == CACHE_VERSION and isinstance(data, tuple) and len(data) == 4:
                return data
            else:
                print("⚠️ Old cache version — rebuilding...")
        except Exception as e:
            print("⚠️ Cache corrupted — rebuilding...", e)

    # ── Rebuild index ─────────────────────────
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    all_paths = df.iloc[:, 2].dropna().astype(str).tolist()
    path_set = set(all_paths)

    leaves = [
        p for p in all_paths
        if not any(other.startswith(p + " / ") for other in path_set)
    ]

    docs = [path_to_doc(p) for p in leaves]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(docs)

    result = (leaves, vectorizer, matrix, all_paths)

    # Save with version
    with open(INDEX_FILE, "wb") as f:
        pickle.dump((CACHE_VERSION, result), f)

    return result


# ─────────────────────────────────────────────────────────────
# JSON CACHE
# ─────────────────────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_to_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)

def find_in_cache(query, cache_data, similarity_threshold=0.92):
    query_lower = query.lower().strip()
    if not query_lower:
        return None, None, 0.0

    lower_cache = {k.lower().strip(): (k, v) for k, v in cache_data.items()}

    if query_lower in lower_cache:
        original_key, cached_val = lower_cache[query_lower]
        return cached_val, original_key, 1.0

    matches = difflib.get_close_matches(query_lower, lower_cache.keys(), n=1, cutoff=similarity_threshold)

    if matches:
        best_match_lower = matches[0]
        original_key, cached_val = lower_cache[best_match_lower]
        ratio = difflib.SequenceMatcher(None, query_lower, best_match_lower).ratio()
        return cached_val, original_key, ratio

    return None, None, 0.0


# ─────────────────────────────────────────────────────────────
# AI RERANK
# ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a product categorization expert.
Given a title and candidates, pick the {top_n} best matches.
Return JSON only like:
{{"categories":[{{"category":"..."}}]}}
Do not explain anything."""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_rerank(idx, query, candidates, client, model, top_n, semaphore):
    async with semaphore:
        cand_list = "\n".join(f"- {c}" for c in candidates)

        resp = await client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE.format(top_n=top_n)},
                {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"}
            ],
        )

        data = json.loads(resp.choices[0].message.content.strip()).get("categories", [])
        return idx, data


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.title("⚡ Product Category Predictor")

api_key = st.text_input("Groq API Key", type="password")

model_choice = st.selectbox(
    "Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
)

top_n = st.slider("Top N", 1, 10, 5)
shortlist_k = st.slider("Shortlist size", 5, 50, 25)
concurrency = st.slider("Parallel requests", 1, 20, 10)

if not api_key:
    st.stop()

# ─────────────────────────────────────────────────────────────
# LOAD INDEX (SAFE)
# ─────────────────────────────────────────────────────────────

map_file = None
for f in ["category_map1.csv", "category_map1.xlsx"]:
    if os.path.exists(f):
        map_file = f
        break

if not map_file:
    st.error("Missing category file")
    st.stop()

leaves, vectorizer, matrix, all_paths = load_or_build_index(map_file)

# ─────────────────────────────────────────────────────────────
# SINGLE
# ─────────────────────────────────────────────────────────────

st.subheader("Single Prediction")

query = st.text_input("Product Title")

if st.button("Predict"):
    cache = load_cache()

    cached_val, matched_key, _ = find_in_cache(query, cache)

    if cached_val:
        st.success(f"⚡ Cache hit: {matched_key}")
        st.write(cached_val)
    else:
        qvec = vectorizer.transform([query])
        sims = cosine_similarity(qvec, matrix)[0]

        top_idx = np.argsort(sims)[::-1][:shortlist_k]
        candidates = [leaves[i] for i in top_idx if sims[i] > 0]

        client = Groq(api_key=api_key)

        cand_list = "\n".join(f"- {c}" for c in candidates)

        resp = client.chat.completions.create(
            model=model_choice,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE.format(top_n=top_n)},
                {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"}
            ],
        )

        preds = json.loads(resp.choices[0].message.content.strip()).get("categories", [])

        cache[query] = preds
        save_to_cache(cache)

        st.write(preds)


# ─────────────────────────────────────────────────────────────
# BATCH
# ─────────────────────────────────────────────────────────────

st.subheader("Batch Prediction")

uploaded = st.file_uploader("Upload CSV or Excel")

if uploaded and st.button("Run Batch"):
    df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)

    titles = df.iloc[:, 0].astype(str).tolist()
    cache = load_cache()

    results = []
    to_predict_idx = []
    to_predict_queries = []

    for i, q in enumerate(titles):
        cached_val, _, _ = find_in_cache(q, cache)

        if cached_val:
            results.append({"Title": q, "Category": cached_val[0]["category"], "Status": "Cached"})
        else:
            results.append({"Title": q, "Category": "Pending", "Status": "Pending"})
            to_predict_idx.append(i)
            to_predict_queries.append(q)

    st.dataframe(results)

    if to_predict_queries:
        qmat = vectorizer.transform(to_predict_queries)
        sims = cosine_similarity(qmat, matrix)

        all_cands = []
        for row in sims:
            idxs = np.argsort(row)[::-1][:shortlist_k]
            all_cands.append([leaves[i] for i in idxs if row[i] > 0])

        client = AsyncGroq(api_key=api_key)
        semaphore = asyncio.Semaphore(concurrency)

        async def run():
            tasks = [
                async_rerank(i, q, c, client, model_choice, top_n, semaphore)
                for i, q, c in zip(to_predict_idx, to_predict_queries, all_cands)
            ]
            return await asyncio.gather(*tasks)

        results_async = asyncio.run(run())

        for idx, preds in results_async:
            results[idx]["Category"] = preds[0]["category"] if preds else "Error"
            results[idx]["Status"] = "Predicted"
            cache[titles[idx]] = preds

        save_to_cache(cache)

        st.success("Done")
        st.dataframe(results)

"""
Product Category Predictor
Strategy : Fast TF-IDF shortlist + async parallel AI reranking + Smart Caching
Speed    : Instant loading + 2-5s for batch processing
Cost     : 1 API call per product (0 calls if cached)
"""

import os
import json
import asyncio
import pickle
import difflib
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

st.set_page_config(
    page_title="Product Category Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
.main-title {
    font-size:2.4rem; font-weight:700;
    background:linear-gradient(90deg,#f55036 0%,#ff8c00 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.result-card {
    background:#f8f9fc; border-left:4px solid #f55036;
    padding:0.8rem; border-radius:8px;
}
.cache-badge {
    background:#e6f4ea; color:#1e7e34;
    padding:4px 10px; border-radius:12px;
    font-size:0.8rem; font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def path_to_doc(path: str) -> str:
    parts = path.split(" / ")
    return " ".join(parts) + " " + " ".join(parts[-2:]) * 2

# ─────────────────────────────────────────────────────────────
# INDEX (FIXED)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing category index...")
def load_or_build_index(file_path: str):

    # Try load cache
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                version, data = pickle.load(f)

            if version == CACHE_VERSION and isinstance(data, tuple) and len(data) == 4:
                return data
            else:
                print("⚠️ Outdated cache — rebuilding...")
        except Exception as e:
            print("⚠️ Cache corrupted — rebuilding...", e)

    # Build fresh
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

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)

    result = (leaves, vectorizer, matrix, all_paths)

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

def find_in_cache(query, cache_data, threshold=0.92):
    q = query.lower().strip()
    lower_cache = {k.lower(): (k, v) for k, v in cache_data.items()}

    if q in lower_cache:
        k, v = lower_cache[q]
        return v, k, 1.0

    matches = difflib.get_close_matches(q, lower_cache.keys(), n=1, cutoff=threshold)
    if matches:
        best = matches[0]
        k, v = lower_cache[best]
        return v, k, 0.95

    return None, None, 0

# ─────────────────────────────────────────────────────────────
# AI
# ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a product categorization expert.
Return JSON:
{"categories":[{"category":"..."}]}"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def async_rerank(idx, query, candidates, client, model, top_n, sem):
    async with sem:
        cand_list = "\n".join(f"- {c}" for c in candidates)

        resp = await client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": f"{query}\n\n{cand_list}"}
            ],
        )

        data = json.loads(resp.choices[0].message.content)
        return idx, data.get("categories", [])

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    api_key = st.text_input("API Key", type="password")
    model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    top_n = st.slider("Top N", 1, 10, 5)
    shortlist_k = st.slider("Shortlist", 5, 50, 25)
    concurrency = st.slider("Parallel", 1, 20, 10)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)

if not api_key:
    st.stop()

# Detect file
map_file = None
for f in ["category_map1.csv", "category_map1.xlsx"]:
    if os.path.exists(f):
        map_file = f
        break

if not map_file:
    st.error("Missing category file")
    st.stop()

leaves, vectorizer, matrix, all_paths = load_or_build_index(map_file)

tab1, tab2 = st.tabs(["Single", "Batch"])

# ─────────────────────────────────────────────────────────────
# SINGLE
# ─────────────────────────────────────────────────────────────

with tab1:
    query = st.text_input("Product Title")

    if st.button("Predict"):
        cache = load_cache()
        cached, key, _ = find_in_cache(query, cache)

        if cached:
            st.markdown(f'<div class="cache-badge">Cache: {key}</div>', unsafe_allow_html=True)
            st.write(cached)
        else:
            qvec = vectorizer.transform([query])
            sims = cosine_similarity(qvec, matrix)[0]

            idxs = np.argsort(sims)[::-1][:shortlist_k]
            cands = [leaves[i] for i in idxs if sims[i] > 0]

            client = Groq(api_key=api_key)

            resp = client.chat.completions.create(
                model=model_choice,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": f"{query}\n\n" + "\n".join(cands)}
                ],
            )

            preds = json.loads(resp.choices[0].message.content)["categories"]

            cache[query] = preds
            save_to_cache(cache)

            st.write(preds)

# ─────────────────────────────────────────────────────────────
# BATCH
# ─────────────────────────────────────────────────────────────

with tab2:
    file = st.file_uploader("Upload file")

    if file and st.button("Run Batch"):
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)

        titles = df.iloc[:, 0].astype(str).tolist()
        cache = load_cache()

        results = []
        to_idx, to_q = [], []

        progress = st.progress(0)
        table = st.empty()

        for i, q in enumerate(titles):
            cached, _, _ = find_in_cache(q, cache)
            if cached:
                results.append({"Title": q, "Category": cached[0]["category"], "Status": "Cached"})
            else:
                results.append({"Title": q, "Category": "Pending", "Status": "Pending"})
                to_idx.append(i)
                to_q.append(q)

        table.dataframe(results)

        if to_q:
            qmat = vectorizer.transform(to_q)
            sims = cosine_similarity(qmat, matrix)

            all_cands = []
            for row in sims:
                idxs = np.argsort(row)[::-1][:shortlist_k]
                all_cands.append([leaves[i] for i in idxs if row[i] > 0])

            client = AsyncGroq(api_key=api_key)
            sem = asyncio.Semaphore(concurrency)

            async def run_chunk(start, end):
                tasks = [
                    async_rerank(i, q, c, client, model_choice, top_n, sem)
                    for i, q, c in zip(to_idx[start:end], to_q[start:end], all_cands[start:end])
                ]
                return await asyncio.gather(*tasks)

            for i in range(0, len(to_q), concurrency):
                chunk = asyncio.run(run_chunk(i, i + concurrency))

                for idx, preds in chunk:
                    results[idx]["Category"] = preds[0]["category"] if preds else "Error"
                    results[idx]["Status"] = "Done"
                    cache[titles[idx]] = preds

                progress.progress(min(100, int((i + concurrency) / len(to_q) * 100)))
                table.dataframe(results)

            save_to_cache(cache)
            st.success("Batch complete")

"""
Product Category Predictor
Stable Version — UI Safe + No Blocking + Cache Protected
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CACHE_FILE = "prediction_cache.json"
INDEX_FILE = "category_index.pkl"
CACHE_VERSION = "v3"

st.set_page_config(page_title="Category Predictor", layout="wide")

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
.main-title {
    font-size:2.2rem;
    font-weight:700;
    color:#f55036;
}
.cache {
    color:green;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ALWAYS SHOW UI FIRST
# ─────────────────────────────────────────────

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single", "Batch"])

# ─────────────────────────────────────────────
# SIDEBAR (NON-BLOCKING)
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    manual_key = st.text_input("Override API Key", type="password")

    if manual_key:
        api_key = manual_key

    if not api_key:
        st.warning("⚠️ No API key — AI predictions disabled")

    model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    shortlist_k = st.slider("Shortlist", 5, 50, 20)
    concurrency = st.slider("Parallel Requests", 1, 20, 10)

# ─────────────────────────────────────────────
# FILE HANDLING (SAFE)
# ─────────────────────────────────────────────

map_file = None
for f in ["category_map1.csv", "category_map1.xlsx"]:
    if os.path.exists(f):
        map_file = f
        break

if not map_file:
    st.warning("⚠️ No category map found. Upload one below.")

    uploaded = st.file_uploader("Upload category file", type=["csv", "xlsx"])

    if uploaded:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
        df.to_csv("category_map1.csv", index=False)
        st.success("Uploaded — reload app")
    
    st.stop()

# ─────────────────────────────────────────────
# INDEX (SAFE)
# ─────────────────────────────────────────────

@st.cache_resource
def load_index(path):
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                version, data = pickle.load(f)
                if version == CACHE_VERSION:
                    return data
        except:
            pass

    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)

    paths = df.iloc[:, 2].dropna().astype(str).tolist()
    path_set = set(paths)

    leaves = [p for p in paths if not any(o.startswith(p + " / ") for o in path_set)]

    docs = [" ".join(p.split(" / ")) for p in leaves]

    vec = TfidfVectorizer(ngram_range=(1, 2))
    mat = vec.fit_transform(docs)

    data = (leaves, vec, mat)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((CACHE_VERSION, data), f)

    return data

leaves, vectorizer, matrix = load_index(map_file)

# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE))
        except:
            return {}
    return {}

def save_cache(c):
    json.dump(c, open(CACHE_FILE, "w"), indent=2)

def find_cache(q, cache):
    q = q.lower()
    if q in cache:
        return cache[q]
    return None

# ─────────────────────────────────────────────
# AI
# ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def async_ai(q, cands, client, sem):
    async with sem:
        resp = await client.chat.completions.create(
            model=model_choice,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "You are a product categorisation assistant. "
                    "Given a product title and a list of candidate categories, "
                    "return ONLY a JSON object with a single key 'category' "
                    "whose value is the best matching category path from the list."
                )},
                {"role": "user", "content": f"Product: {q}\n\nCandidates:\n" + "\n".join(cands)}
            ],
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"category": raw.strip()}

# ─────────────────────────────────────────────
# SINGLE TAB
# ─────────────────────────────────────────────

with tab1:
    query = st.text_input("Product title")

    if st.button("Predict"):
        cache = load_cache()

        cached = find_cache(query, cache)
        if cached:
            st.markdown('<p class="cache">From cache</p>', unsafe_allow_html=True)
            st.write(cached)
        else:
            sims = cosine_similarity(vectorizer.transform([query]), matrix)[0]
            idxs = np.argsort(sims)[::-1][:shortlist_k]
            cands = [leaves[i] for i in idxs]

            if not api_key:
                st.error("No API key")
            else:
                try:
                    client = Groq(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=model_choice,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": (
                                "You are a product categorisation assistant. "
                                "Given a product title and a list of candidate categories, "
                                "return ONLY a JSON object with a single key 'category' "
                                "whose value is the best matching category path from the list."
                            )},
                            {"role": "user", "content": f"Product: {query}\n\nCandidates:\n" + "\n".join(cands)}
                        ],
                    )
                    raw = resp.choices[0].message.content
                    try:
                        out = json.loads(raw)
                    except json.JSONDecodeError:
                        out = {"category": raw.strip(), "parse_warning": "Response was not valid JSON"}
                    cache[query.lower()] = out
                    save_cache(cache)
                    st.write(out)
                except Exception as e:
                    st.error(f"Groq API error: {e}")
                    st.info("💡 Check your API key at console.groq.com and confirm the selected model is available.")

# ─────────────────────────────────────────────
# BATCH TAB
# ─────────────────────────────────────────────

with tab2:
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"], key="batch")

    if uploaded and st.button("Run Batch"):
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)

        titles = df.iloc[:, 0].astype(str).tolist()
        cache = load_cache()

        results = []
        todo = []

        for i, q in enumerate(titles):
            c = find_cache(q, cache)
            if c:
                results.append({"Title": q, "Category": c, "Status": "Cached"})
            else:
                results.append({"Title": q, "Category": "", "Status": "Pending"})
                todo.append((i, q))

        table = st.empty()
        table.dataframe(results)

        if todo and api_key:
            client = AsyncGroq(api_key=api_key)
            sem = asyncio.Semaphore(concurrency)

            async def run():
                tasks = []
                for i, q in todo:
                    sims = cosine_similarity(vectorizer.transform([q]), matrix)[0]
                    idxs = np.argsort(sims)[::-1][:shortlist_k]
                    cands = [leaves[x] for x in idxs]
                    tasks.append(async_ai(q, cands, client, sem))
                return await asyncio.gather(*tasks, return_exceptions=True)

            try:
                outputs = asyncio.run(run())

                for (i, q), out in zip(todo, outputs):
                    if isinstance(out, Exception):
                        results[i]["Category"] = f"Error: {out}"
                        results[i]["Status"] = "Failed"
                    else:
                        results[i]["Category"] = out
                        results[i]["Status"] = "Done"
                        cache[q.lower()] = out

                    table.dataframe(results)

                save_cache(cache)
                st.success("Done")
            except Exception as e:
                st.error(f"Batch error: {e}")
                st.info("💡 Check your API key at console.groq.com and confirm the selected model is available.")

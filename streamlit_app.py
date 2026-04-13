"""
Product Category Predictor
Strategy : Optimized TF-IDF shortlist + async parallel Groq reranking
Speed    : Instant file loading + 2-5s for batch processing
"""

import os, io, json, asyncio, pickle, time
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq, Groq
from tenacity import retry, stop_after_attempt, wait_exponential

st.set_page_config(page_title="Product Category Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Optimized Loading & Indexing ---

def path_to_doc(path: str) -> str:
    """Enhanced path document for better TF-IDF matching."""
    parts = path.split(" / ")
    # Weigh the last 2 parts of the path more heavily
    return " ".join(parts) + " " + " ".join(parts[-2:]) * 2

@st.cache_resource(show_spinner=False)
def load_or_build_index(file_path: str, cache_file="category_index.pkl"):
    # Check if we can load from cache to save time
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass 

    # Building the index (Fast TF-IDF)
    try:
        df = pd.read_excel(file_path)
    except:
        df = pd.read_csv(file_path)
    
    # Extract Category Path (Column 3)
    all_paths = df.iloc[:, 2].dropna().astype(str).tolist()
    path_set = set(all_paths)
    
    # Find leaf nodes
    leaves = [p for p in all_paths if not any(other.startswith(p + " / ") for other in path_set)]
    docs = [path_to_doc(p) for p in leaves]
    
    # TF-IDF is significantly faster to build than Semantic Embeddings
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    
    result = (leaves, vectorizer, matrix, all_paths)
    
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        
    return result

def shortlist(query: str, leaves, vectorizer, matrix, k: int = 25) -> list[str]:
    qvec = vectorizer.transform([query])
    sims = cosine_similarity(qvec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [leaves[i] for i in top_idx if sims[i] > 0]

def batch_shortlist(queries: list[str], leaves, vectorizer, matrix, k: int = 25) -> list[list[str]]:
    # Vectorize all queries at once (Matrix Operation - very fast)
    qmat = vectorizer.transform(queries)
    sims = cosine_similarity(qmat, matrix)
    results = []
    for row in sims:
        top_idx = np.argsort(row)[::-1][:k]
        results.append([leaves[i] for i in top_idx if row[i] > 0])
    return results

# --- Async AI Reranking ---

SYSTEM_TEMPLATE = """You are a product categorization expert.
Pick the {top_n} best categories. JSON only.
RULE: If single item, do NOT pick "Sets" categories.
Format: {{"categories": [{{"category": "path", "score": 0.95}}]}}"""

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
                {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"},
            ],
        )
        return idx, json.loads(resp.choices[0].message.content.strip()).get("categories", [])

async def parallel_predict(queries, candidates_list, api_key, model, top_n, concurrency):
    client = AsyncGroq(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [async_rerank(i, q, c, client, model, top_n, semaphore) for i, (q, c) in enumerate(zip(queries, candidates_list))]
    results_raw = await asyncio.gather(*tasks)
    return [r for _, r in sorted(results_raw)]

# --- UI Logic ---

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    shortlist_k = st.slider("Shortlist", 5, 50, 25)
    concurrency = st.slider("Concurrency", 1, 30, 10)

if not api_key:
    st.info("Please enter your API Key in the sidebar.")
    st.stop()

# Auto-load File
excel_path = "category_map1.xlsx"
if os.path.exists(excel_path):
    with st.spinner("Initializing Fast Index..."):
        leaves, vectorizer, matrix, all_paths = load_or_build_index(excel_path)
    st.success(f"Index Ready: {len(leaves):,} categories")
else:
    st.error("category_map1.xlsx not found.")
    st.stop()

# --- Tabs ---
tab_single, tab_batch = st.tabs(["Single Predict", "Batch Predict"])

with tab_single:
    prod_input = st.text_input("Product Title")
    if st.button("Predict"):
        cands = shortlist(prod_input, leaves, vectorizer, matrix, shortlist_k)
        # Use sync client for single predict for simplicity
        client = Groq(api_key=api_key)
        cand_str = "\n".join(cands)
        resp = client.chat.completions.create(
            model=model_choice,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_TEMPLATE.format(top_n=3)},
                      {"role": "user", "content": f"Product: {prod_input}\nCandidates: {cand_str}"}]
        )
        res = json.loads(resp.choices[0].message.content)["categories"]
        st.write(res)

with tab_batch:
    uploaded = st.file_uploader("Upload Batch File")
    if uploaded and st.button("Run Batch"):
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        queries = df.iloc[:, 0].astype(str).tolist() # Assuming first column is titles
        
        t0 = time.time()
        # Step 1: Fast Matrix Shortlisting
        all_cands = batch_shortlist(queries, leaves, vectorizer, matrix, shortlist_k)
        
        # Step 2: Parallel AI Reranking
        all_preds = asyncio.run(parallel_predict(queries, all_cands, api_key, model_choice, 1, concurrency))
        
        df['Predicted Category'] = [p[0]['category'] if p else "N/A" for p in all_preds]
        st.write(f"Done in {time.time()-t0:.2f}s")
        st.data_editor(df)

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
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq, Groq
from tenacity import retry, stop_after_attempt, wait_exponential

st.set_page_config(
    page_title="Product Category Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size:2.4rem; font-weight:700;
        background:linear-gradient(90deg,#f55036 0%,#ff8c00 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:1.5rem;
    }
    .result-card {
        background:#f8f9fc; border-left:4px solid #f55036;
        padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.5rem;
    }
    .stTextArea textarea { border-radius:10px !important; border:2px solid #e0e0f0 !important; }
    .stTextArea textarea:focus { border-color:#f55036 !important; }
    .cache-badge {
        display:inline-block; background:#e6f4ea; color:#1e7e34;
        font-size:0.8rem; font-weight:700; padding:4px 12px;
        border-radius:20px; margin-bottom: 1rem; border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# ─── Optimized Loading & Indexing (TF-IDF) ──────────────────────────────────

def path_to_doc(path: str) -> str:
    parts = path.split(" / ")
    # Weigh the specific leaf nodes more than the parent categories
    return " ".join(parts) + " " + " ".join(parts[-2:]) * 2

@st.cache_resource(show_spinner="Initializing fast category index...")
def load_or_build_index(file_path: str, cache_file="category_index.pkl"):
    # If the pre-computed index exists on disk, load it instantly
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except:
            pass

    # Detect if file is CSV or Excel
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    all_paths = df.iloc[:, 2].dropna().astype(str).tolist()
    path_set  = set(all_paths)
    leaves    = [p for p in all_paths if not any(other.startswith(p + " / ") for other in path_set)]
    
    docs = [path_to_doc(p) for p in leaves]
    
    # TF-IDF is near-instant compared to Semantic Transformers
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    
    result = (leaves, vectorizer, matrix, all_paths)
    
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        
    return result

# ─── Smart JSON Caching ──────────────────────────────────────────────────────

CACHE_FILE = "prediction_cache.json"

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
    if not query_lower: return None, None, 0.0

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

# ─── AI Logic ───────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a product categorization expert.
Given a title and candidates, pick the {top_n} best matches. 
Rules: JSON only. If a single item, DO NOT pick "Sets" or "Packs" categories."""

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
        data = json.loads(resp.choices[0].message.content.strip()).get("categories", [])
        return idx, data

def render_results(preds, score_threshold, show_chart, show_hierarchy):
    # (Rest of your existing rendering logic for charts and cards)
    st.write(preds) # Simplified for this snippet

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## API Key")
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    api_key = st.text_input("Key", value=api_key, type="password")
    
    st.markdown("---")
    st.markdown("## Settings")
    model_choice = st.selectbox("AI Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    top_n = st.slider("Top N results", 1, 10, 5)
    shortlist_k = st.slider("Shortlist", 5, 50, 25)
    concurrency = st.slider("Parallel requests", 1, 30, 10)

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)

if not api_key:
    st.info("Please enter your API key in the sidebar.")
    st.stop()

# Detect file map (Automatically handles CSV vs Excel)
map_file = None
for f in ["category_map1.xlsx - pim category attribute set.csv", "category_map1.csv", "category_map1.xlsx"]:
    if os.path.exists(f): map_file = f; break

if not map_file:
    st.error("Missing category_map1 file.")
    st.stop()

# This is where the 4-value unpacking happens (Corrected)
leaves, vectorizer, matrix, all_paths = load_or_build_index(map_file)

tab_single, tab_batch = st.tabs(["Single Predict", "Batch Predict"])

# ── Single ─────────────────────────────────────────────────────────────────────
with tab_single:
    product_text = st.text_input("Product Title")
    if st.button("Predict"):
        query = product_text.strip()
        pred_cache = load_cache()
        cached_val, matched_key, sim_ratio = find_in_cache(query, pred_cache)
        
        if cached_val:
            st.markdown(f'<div class="cache-badge">⚡ Loaded from Cache ("{matched_key}")</div>', unsafe_allow_html=True)
            st.write(cached_val)
        else:
            with st.spinner("Analyzing..."):
                qvec = vectorizer.transform([query])
                sims = cosine_similarity(qvec, matrix)[0]
                top_idx = np.argsort(sims)[::-1][:shortlist_k]
                candidates = [leaves[i] for i in top_idx if sims[i] > 0]
                
                client = Groq(api_key=api_key)
                cand_list = "\n".join(f"- {c}" for c in candidates)
                resp = client.chat.completions.create(
                    model=model_choice, temperature=0.1, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM_TEMPLATE.format(top_n=top_n)},
                              {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"}]
                )
                preds = json.loads(resp.choices[0].message.content.strip()).get("categories", [])
                pred_cache[query] = preds
                save_to_cache(pred_cache)
                st.write(preds)

# ── Batch ──────────────────────────────────────────────────────────────────────
with tab_batch:
    uploaded = st.file_uploader("Upload Batch File")
    if uploaded and st.button("🚀 Run Batch Prediction"):
        df_input = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
        titles = df_input.iloc[:, 0].astype(str).tolist()
        
        pred_cache = load_cache()
        results_list = []
        to_predict_indices, to_predict_queries = [], []

        # Progress bar and table placeholder
        progress_bar = st.progress(0)
        table_placeholder = st.empty()

        for i, q in enumerate(titles):
            cached_val, _, _ = find_in_cache(q, pred_cache)
            if cached_val:
                results_list.append({"Title": q, "Category": cached_val[0]["category"], "Status": "Cached"})
            else:
                results_list.append({"Title": q, "Category": "Pending...", "Status": "Pending"})
                to_predict_indices.append(i)
                to_predict_queries.append(q)

        table_placeholder.data_editor(pd.DataFrame(results_list), use_container_width=True)

        if to_predict_queries:
            # Step 1: Matrix Shortlisting (Fast)
            qmat = vectorizer.transform(to_predict_queries)
            sims = cosine_similarity(qmat, matrix)
            all_cands = []
            for row in sims:
                t_idx = np.argsort(row)[::-1][:shortlist_k]
                all_cands.append([leaves[x] for x in t_idx if row[x] > 0])

            # Step 2: Parallel AI
            client = AsyncGroq(api_key=api_key)
            semaphore = asyncio.Semaphore(concurrency)
            
            async def run_chunk(start, end):
                tasks = [async_rerank(idx, q, c, client, model_choice, top_n, semaphore) 
                         for idx, q, c in zip(to_predict_indices[start:end], to_predict_queries[start:end], all_cands[start:end])]
                return await asyncio.gather(*tasks)

            for i in range(0, len(to_predict_queries), concurrency):
                chunk_results = asyncio.run(run_chunk(i, i + concurrency))
                for original_idx, preds in chunk_results:
                    results_list[original_idx]["Category"] = preds[0]["category"] if preds else "Error"
                    results_list[original_idx]["Status"] = "Predicted"
                    pred_cache[titles[original_idx]] = preds
                
                perc = int((i + concurrency) / len(to_predict_queries) * 100)
                progress_bar.progress(min(100, perc))
                table_placeholder.data_editor(pd.DataFrame(results_list), use_container_width=True, key=f"table_{i}")

            save_to_cache(pred_cache)

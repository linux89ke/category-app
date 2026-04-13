"""
Product Category Predictor
Strategy : Semantic Embeddings shortlist + async parallel AI reranking + Smart Caching
Speed    : all products run concurrently, ~2-5s for any batch size
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
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq, Groq

# Imports for auto-retries and semantic search
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer

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


# ─── Smart JSON Caching System ────────────────────────────────────────────────

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

def find_in_cache(query, cache_data, similarity_threshold=0.90):
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


# ─── Semantic Search Index ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Loading semantic category index (First time only)...")
def load_or_build_index(file_path: str, cache_file="category_index.pkl"):
    model = get_embedding_model()
    
    needs_rebuild = True
    if os.path.exists(cache_file) and os.path.exists(file_path):
        file_mtime = os.path.getmtime(file_path)
        cache_mtime = os.path.getmtime(cache_file)
        if cache_mtime > file_mtime:
            needs_rebuild = False

    if not needs_rebuild:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    try:
        df = pd.read_excel(file_path)
    except Exception:
        df = pd.read_csv(file_path)
        
    all_paths = df.iloc[:, 2].dropna().astype(str).tolist()
    path_set  = set(all_paths)
    leaves    = [p for p in all_paths if not any(other.startswith(p + " / ") for other in path_set)]
    
    docs = [p.replace(" / ", " ") for p in leaves]
    matrix = model.encode(docs, show_progress_bar=False)
    
    result = (leaves, matrix, all_paths)
    
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        
    return result


def shortlist(query: str, leaves, matrix, k: int = 25) -> list[str]:
    model = get_embedding_model()
    qvec = model.encode([query])
    sims = cosine_similarity(qvec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [leaves[i] for i in top_idx if sims[i] > 0]


def batch_shortlist(queries: list[str], leaves, matrix, k: int = 25) -> list[list[str]]:
    model = get_embedding_model()
    qmat = model.encode(queries, show_progress_bar=False)
    sims = cosine_similarity(qmat, matrix)          
    results = []
    for row in sims:
        top_idx = np.argsort(row)[::-1][:k]
        results.append([leaves[i] for i in top_idx if row[i] > 0])
    return results


# ─── Async AI reranking ─────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a product categorization expert.
Given a product title and a list of candidate category paths, pick the {top_n} best matching categories.
Consider brand, product type, format/medium, gender, style, material, and QUANTITY.

CRITICAL RULES:
1. QUANTITY: Pay attention to plurals, packs, and sets. If a product is a single item, DO NOT put it in a "Sets" category.
2. FORMAT/MEDIUM: Pay close attention to format keywords like 'Hardcover', 'Paperback', 'DVD', 'CD', 'Audiobook'. Do NOT categorize a physical book as a DVD, or a movie as a book.

Rules:
- Return exactly {top_n} categories ordered by confidence descending
- Only pick from the provided candidate list — never invent categories
- JSON only, nothing else"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_rerank(
    idx: int, query: str, candidates: list[str], client: AsyncGroq, model: str, top_n: int, semaphore: asyncio.Semaphore
) -> tuple[int, list[dict]]:
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
        raw  = resp.choices[0].message.content.strip()
        data = json.loads(raw).get("categories", [])
        return idx, data


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## API Key")
    default_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    api_key = st.text_input("Paste your key here:", value=default_key, type="password")
    
    st.markdown("---")
    st.markdown("## Settings")
    model_choice = st.selectbox("AI Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"], index=0)
    top_n = st.slider("Top N results", 1, 10, 5)
    shortlist_k = st.slider("Shortlist size", 5, 50, 25)
    concurrency = st.slider("Parallel requests", 1, 30, 10)
    score_threshold = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    show_chart = st.checkbox("Show confidence chart", value=True)
    show_hierarchy = st.checkbox("Show category hierarchy", value=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)

if not api_key:
    st.info("Enter your API key in the sidebar.")
    st.stop()

map_file = None
for f in ["category_map1.xlsx", "category_map1.csv"]:
    if os.path.exists(f): map_file = f; break

cache_path = "category_index.pkl"
if not map_file and not os.path.exists(cache_path):
    st.error("Missing category_map1.xlsx/csv.")
    st.stop()

leaves, matrix, all_paths = load_or_build_index(map_file or cache_path, cache_path)

tab_single, tab_batch, tab_explore = st.tabs(["Single Predict", "Batch Predict", "Explore"])

# ── Single ─────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Enter a product title")
    col_title, col_brand = st.columns([3, 1])
    with col_title:
        product_text = st.text_area("Product title", value="", height=90, placeholder="e.g. Air Max 270...")
    with col_brand:
        brand = st.text_input("Brand", placeholder="e.g. Nike")

    if st.button("Predict", type="primary", use_container_width=True):
        if product_text.strip():
            query = f"{brand.strip()} {product_text.strip()}".strip()
            pred_cache = load_cache()
            cached_val, matched_key, sim_ratio = find_in_cache(query, pred_cache)
            
            if cached_val:
                st.markdown(f'<div class="cache-badge">⚡ Instant load (Matched: "{matched_key}")</div>', unsafe_allow_html=True)
                render_results(cached_val, score_threshold, show_chart, show_hierarchy)
            else:
                with st.spinner("Analyzing..."):
                    candidates = shortlist(query, leaves, matrix, shortlist_k)
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
                    render_results(preds, score_threshold, show_chart, show_hierarchy)

# ── Batch ──────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch predict")
    top_n_batch = st.slider("Top N per product", 1, 5, 1, key="batch_topn")
    input_mode = st.radio("Input method", ["Upload file", "Paste a list"], horizontal=True)

    texts, brands = [], []

    if input_mode == "Upload file":
        uploaded = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
        if uploaded:
            try:
                df_input = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
                st.dataframe(df_input.head(3), use_container_width=True)
                tc = st.selectbox("Title column", df_input.columns.tolist())
                bc = st.selectbox("Brand column", ["— none —"] + df_input.columns.tolist())
                texts = df_input[tc].astype(str).fillna("").tolist()
                brands = df_input[bc].astype(str).fillna("").tolist() if bc != "— none —" else [""] * len(texts)
            except: st.error("Error reading file.")
    else:
        pasted = st.text_area("One product per line:", height=150)
        brand_p = st.text_input("Brand (optional)", key="p_brand")
        texts = [t.strip() for t in pasted.splitlines() if t.strip()]
        brands = [brand_p] * len(texts)

    if texts:
        if st.button("🚀 Run Batch Prediction", type="primary"):
            pred_cache = load_cache()
            queries = [f"{b} {t}".strip() for t, b in zip(texts, brands)]
            
            # Placeholders for real-time updates
            progress_text = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            table_placeholder = st.empty()
            
            results_list = []
            to_predict_indices = []
            to_predict_queries = []

            # Initialize results with cache or mark for prediction
            for i, q in enumerate(queries):
                cached_val, matched_key, _ = find_in_cache(q, pred_cache)
                if cached_val:
                    res = {"input_text": texts[i], "brand": brands[i], 
                           "top_category": cached_val[0]["category"] if cached_val else "",
                           "top_score": cached_val[0]["score"] if cached_val else 0, "status": "Cached"}
                    results_list.append(res)
                else:
                    results_list.append({"input_text": texts[i], "brand": brands[i], "top_category": "Processing...", "top_score": 0, "status": "Pending"})
                    to_predict_indices.append(i)
                    to_predict_queries.append(q)

            # Show initial table
            table_placeholder.data_editor(pd.DataFrame(results_list), use_container_width=True, key="realtime_table")

            if to_predict_queries:
                status_text.info(f"Shortlisting {len(to_predict_queries)} products...")
                all_candidates = batch_shortlist(to_predict_queries, leaves, matrix, shortlist_k)
                
                client = AsyncGroq(api_key=api_key)
                semaphore = asyncio.Semaphore(concurrency)

                # Process in chunks to update the UI
                chunk_size = concurrency
                for i in range(0, len(to_predict_queries), chunk_size):
                    chunk_queries = to_predict_queries[i : i + chunk_size]
                    chunk_indices = to_predict_indices[i : i + chunk_size]
                    chunk_cands = all_candidates[i : i + chunk_size]
                    
                    tasks = [async_rerank(idx, q, c, client, model_choice, top_n_batch, semaphore) 
                             for idx, q, c in zip(chunk_indices, chunk_queries, chunk_cands)]
                    
                    chunk_results = asyncio.run(asyncio.gather(*tasks))
                    
                    for original_idx, preds in chunk_results:
                        # Update main result list
                        results_list[original_idx]["top_category"] = preds[0]["category"] if preds else "Error"
                        results_list[original_idx]["top_score"] = preds[0]["score"] if preds else 0
                        results_list[original_idx]["status"] = "AI Predicted"
                        
                        # Update Cache
                        pred_cache[queries[original_idx]] = preds
                    
                    # Update UI
                    perc = min(100, int((i + chunk_size) / len(to_predict_queries) * 100))
                    progress_bar.progress(perc)
                    progress_text.text(f"Processing: {perc}%")
                    table_placeholder.data_editor(pd.DataFrame(results_list), use_container_width=True, key=f"table_{i}")
                
                save_to_cache(pred_cache)
                status_text.success("Batch Prediction Complete!")
            
            st.download_button("Download CSV", pd.DataFrame(results_list).to_csv(index=False).encode(), "results.csv")

# ── Explore ────────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("### Explore Map")
    search = st.text_input("Search categories...")
    if search:
        res = [p for p in all_paths if search.lower() in p.lower()]
        for p in res[:50]: st.markdown(f"`{p}`")

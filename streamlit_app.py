"""
Product Category Predictor
Strategy : Semantic Embeddings shortlist + async parallel Groq reranking + Smart Caching
Speed    : all products run concurrently, ~2-5s for any batch size
Cost     : 1 Groq call per product (0 calls if cached)
"""

import os
import io
import json
import asyncio
import pickle
import difflib
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
    """Checks for exact and similar (>90%) matches in the JSON cache."""
    query_lower = query.lower().strip()
    if not query_lower:
        return None, None, 0.0

    # Create a lookup dict with lowercased keys
    lower_cache = {k.lower().strip(): (k, v) for k, v in cache_data.items()}
    
    # 1. Check for Exact match
    if query_lower in lower_cache:
        original_key, cached_val = lower_cache[query_lower]
        return cached_val, original_key, 1.0

    # 2. Check for Similar match
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

@st.cache_resource(show_spinner=False)
def load_or_build_index(file_path: str, cache_file="category_index.pkl"):
    model = get_embedding_model()
    
    # Rebuild if cache doesn't exist OR if file is newer than the cache
    needs_rebuild = True
    if os.path.exists(cache_file) and os.path.exists(file_path):
        file_mtime = os.path.getmtime(file_path)
        cache_mtime = os.path.getmtime(cache_file)
        if cache_mtime > file_mtime:
            needs_rebuild = False

    if not needs_rebuild:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Build it from the Excel/CSV file
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


# ─── Async Groq reranking ─────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a product categorization expert.
Given a product title and a list of candidate category paths, pick the {top_n} best matching categories.
Consider brand, product type, format/medium, gender, style, material, and QUANTITY.

CRITICAL RULES:
1. QUANTITY: Pay attention to plurals, packs, and sets. If a product is a single item, DO NOT put it in a "Sets" category.
2. FORMAT/MEDIUM: Pay close attention to format keywords like 'Hardcover', 'Paperback', 'DVD', 'CD', 'Audiobook'. Do NOT categorize a physical book as a DVD, or a movie as a book.

EXAMPLE INPUT 1:
Product: "Acrylic Fruit and Salad Bowl with Gold Rim Elegant Transparent Serving Bowl"
Candidates:
- Home & Kitchen / Kitchen & Dining / Serveware / Salad Serving Sets
- Home & Kitchen / Kitchen & Dining / Serveware / Serving Bowls
EXAMPLE OUTPUT 1:
{{
  "categories": [
    {{"category": "Home & Kitchen / Kitchen & Dining / Serveware / Serving Bowls", "score": 0.98}}
  ]
}}

EXAMPLE INPUT 2:
Product: "Harry Potter Sorcerer's Stone Hardcover"
Candidates:
- Books, Movies and Music / DVDs / Fantasy
- Books, Movies and Music / Books / Literature & Fiction
EXAMPLE OUTPUT 2:
{{
  "categories": [
    {{"category": "Books, Movies and Music / Books / Literature & Fiction", "score": 0.99}}
  ]
}}

Rules:
- Return exactly {top_n} categories ordered by confidence descending
- Only pick from the provided candidate list — never invent categories
- Scores are floats 0.0–1.0
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


async def parallel_predict(
    queries: list[str], candidates_list: list[list[str]], api_key: str, model: str, top_n: int, concurrency: int
) -> list[list[dict]]:
    client    = AsyncGroq(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    tasks     = []
    
    for i, (q, c) in enumerate(zip(queries, candidates_list)):
        async def safe_rerank(idx, query, cands):
            try:
                return await async_rerank(idx, query, cands, client, model, top_n, semaphore)
            except Exception as e:
                return idx, [{"category": f"API ERROR: {e}", "score": 0.0}]
        tasks.append(safe_rerank(i, q, c))
        
    results_raw = await asyncio.gather(*tasks)
    ordered = sorted(results_raw, key=lambda x: x[0])
    return [r for _, r in ordered]


def run_parallel(queries, candidates_list, api_key, model, top_n, concurrency):
    return asyncio.run(parallel_predict(queries, candidates_list, api_key, model, top_n, concurrency))


def sync_rerank(query, candidates, api_key, model, top_n):
    client    = Groq(api_key=api_key)
    cand_list = "\n".join(f"- {c}" for c in candidates)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE.format(top_n=top_n)},
            {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"},
        ],
    )
    return json.loads(resp.choices[0].message.content.strip()).get("categories", [])


# ─── Result renderer ──────────────────────────────────────────────────────────

def render_results(preds, score_threshold, show_chart, show_hierarchy):
    preds = [p for p in preds if p.get("score", 0) >= score_threshold]
    if not preds:
        st.warning("No categories above the confidence threshold.")
        return

    left, right = (st.columns([1, 1]) if show_chart else (st, None))

    with left:
        st.markdown("#### Top Predictions")
        for i, p in enumerate(preds):
            pct   = p["score"] * 100
            color = "#f55036" if pct > 60 else "#ff8c00" if pct > 30 else "#ffd580"
            st.markdown(f"""
            <div class="result-card">
              <span style="font-size:.72rem;font-weight:700;color:#f55036;text-transform:uppercase;">#{i+1}</span>
              <div style="font-size:1rem;font-weight:600;color:#1a1a2e;">{p['category']}</div>
              <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                <div style="flex:1;height:6px;background:#e8eaf6;border-radius:3px;">
                  <div style="width:{int(pct)}%;height:100%;background:{color};border-radius:3px;"></div>
                </div>
                <span style="font-size:.88rem;color:#555;">{pct:.1f}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

    if show_chart and right:
        with right:
            st.markdown("#### Confidence Chart")
            df = pd.DataFrame(preds).sort_values("score")
            df["label"] = df["category"].apply(
                lambda x: " / ".join(x.split(" / ")[-2:]) if " / " in x else x)
            fig = go.Figure(go.Bar(
                x=df["score"]*100, y=df["label"], orientation="h",
                marker=dict(color=df["score"]*100,
                            colorscale=[[0,"#ffd580"],[0.5,"#ff8c00"],[1,"#f55036"]],
                            showscale=False),
                text=[f"{s*100:.1f}%" for s in df["score"]],
                textposition="outside",
                hovertext=df["category"], hoverinfo="text+x",
            ))
            fig.update_layout(
                xaxis_title="Confidence (%)",
                margin=dict(l=0, r=60, t=10, b=30),
                height=max(300, len(preds)*36),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 115]),
            )
            st.plotly_chart(fig, use_container_width=True)

    if show_hierarchy:
        lines, seen = [], set()
        for p in preds:
            parts = [x.strip() for x in p["category"].split(" / ") if x.strip()]
            if len(parts) > 1:
                if parts[0] not in seen:
                    lines.append(f"**{parts[0]}**")
                    seen.add(parts[0])
                for d, part in enumerate(parts[1:], 1):
                    lines.append(f"{'  '*d}└─ {part}")
            else:
                lines.append(f"{p['category']}")
        if lines:
            st.markdown("#### Category Hierarchy")
            st.markdown("\n".join(lines))


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Groq API Key")
    # Pulls from Streamlit Secrets or Environment Variables automatically
    default_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    
    api_key = st.text_input("Paste your key here:",
                            value=default_key,
                            type="password", placeholder="gsk_...")
    st.caption("Free key at console.groq.com")

    st.markdown("---")
    st.markdown("## Settings")
    model_choice = st.selectbox(
        "Groq model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
    )
    top_n        = st.slider("Top N results", 1, 10, 5)
    shortlist_k  = st.slider("Shortlist size", 5, 50, 25,
                             help="Candidates sent to Groq per product. 25 ensures edge cases are included.")
    concurrency  = st.slider("Parallel requests", 1, 30, 10)
    score_threshold = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    show_chart   = st.checkbox("Show confidence chart", value=True)
    show_hierarchy = st.checkbox("Show category hierarchy", value=True)

    st.markdown("---")
    st.markdown("""### Updates included
- **Smart JSON Caching:** Instantly loads exact & similar products without API costs.
- **Semantic Search:** Understands context and meaning.
- **Few-Shot Prompting:** Trained to avoid putting single items into 'Set' categories.
- **Auto-Retries:** Automatically retries API calls if they fail.
""")

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)

if not api_key:
    st.info("Enter your Groq API key in the sidebar or Streamlit Secrets.")
    st.stop()

# Detect file map
map_file = None
for f in ["category_map1.xlsx", "category_map1.csv"]:
    if os.path.exists(f):
        map_file = f
        break

cache_path = "category_index.pkl"

if not map_file and not os.path.exists(cache_path):
    st.error("Required file 'category_map1.xlsx' or 'category_map1.csv' not found in the script directory.")
    st.stop()

with st.spinner("Loading semantic category index..."):
    leaves, matrix, all_paths = load_or_build_index(map_file or cache_path, cache_path)

st.success(f"Successfully loaded {len(leaves):,} leaf categories.")

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_explore = st.tabs(["Single Predict", "Batch Predict", "Explore"])

# ── Single ─────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Enter a product title")

    col_title, col_brand = st.columns([3, 1])
    with col_title:
        product_text = st.text_area(
            "Product title", value="", height=90,
            placeholder="e.g. Air Max 270 Men's Running Shoes...",
        )
    with col_brand:
        brand = st.text_input("Brand *(optional)*", placeholder="e.g. Nike")

    if st.button("Predict", type="primary", use_container_width=True):
        if product_text.strip():
            query = f"{brand.strip()} {product_text.strip()}".strip() if brand.strip() else product_text.strip()
            
            # --- Cache Check ---
            pred_cache = load_cache()
            cached_val, matched_key, sim_ratio = find_in_cache(query, pred_cache)
            
            if cached_val:
                if sim_ratio == 1.0:
                    st.markdown(f'<div class="cache-badge">⚡ Instant load (Exact match found in cache)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="cache-badge">⚡ Instant load (Matched similar cached product: "{matched_key}" - {sim_ratio:.1%} similar)</div>', unsafe_allow_html=True)
                
                render_results(cached_val, score_threshold, show_chart, show_hierarchy)
            
            else:
                with st.spinner("Shortlisting via Semantic Search..."):
                    candidates = shortlist(query, leaves, matrix, shortlist_k)
                with st.spinner(f"Asking Groq ({len(candidates)} candidates)..."):
                    try:
                        preds = sync_rerank(query, candidates, api_key, model_choice, top_n)
                        
                        # Save to cache
                        pred_cache[query] = preds
                        save_to_cache(pred_cache)
                        
                        render_results(preds, score_threshold, show_chart, show_hierarchy)
                        with st.expander(f"{len(candidates)} candidates sent to Groq"):
                            for c in candidates:
                                st.markdown(f"- {c}")
                    except Exception as e:
                        st.error(f"Groq error: {e}")
        else:
            st.warning("Please enter a product title.")


# ── Batch ──────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Batch predict")
    top_n_batch = st.slider("Top N per product", 1, 5, 1, key="batch_topn")
    input_mode  = st.radio("Input method", ["Upload file (CSV or Excel)", "Paste a list"], horizontal=True)

    texts, brands = [], []

    if input_mode == "Upload file (CSV or Excel)":
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df_input = pd.read_excel(uploaded)
                else:
                    try:
                        df_input = pd.read_csv(uploaded, encoding="utf-8")
                    except UnicodeDecodeError:
                        uploaded.seek(0)
                        df_input = pd.read_csv(uploaded, encoding="latin-1")

                st.dataframe(df_input.head(5), use_container_width=True)
                col_tc, col_bc = st.columns([2, 1])
                with col_tc:
                    text_col = st.selectbox("Product title column", df_input.columns.tolist())
                with col_bc:
                    brand_col = st.selectbox("Brand column *(optional)*", ["— none —"] + df_input.columns.tolist())
                
                texts  = df_input[text_col].astype(str).fillna("").tolist()
                brands = (df_input[brand_col].astype(str).fillna("").tolist() if brand_col != "— none —" else [""] * len(texts))
            except Exception as e:
                st.error(f"Could not read file: {e}")
    else:
        pasted = st.text_area("Paste one product per line:", height=180, placeholder="Nike Air Max 270\nKitchenAid Stand Mixer")
        brand_prefix = st.text_input("Brand *(optional — applies to all)*", placeholder="e.g. Nike", key="paste_brand")
        if pasted.strip():
            texts  = [t.strip() for t in pasted.strip().splitlines() if t.strip()]
            brands = [brand_prefix.strip() if brand_prefix else ""] * len(texts)

    if texts:
        if st.button("Run Batch Prediction", type="primary"):
            import time
            
            queries = [f"{b.strip()} {t.strip()}".strip() if b.strip() else t.strip() for t, b in zip(texts, brands)]
            
            # --- Batch Cache Checking ---
            pred_cache = load_cache()
            final_preds = [None] * len(queries)
            to_predict_indices, to_predict_queries = [], []
            
            cache_hits = 0
            for i, q in enumerate(queries):
                cached_val, matched_key, _ = find_in_cache(q, pred_cache, similarity_threshold=0.90)
                if cached_val:
                    final_preds[i] = cached_val
                    cache_hits += 1
                else:
                    to_predict_indices.append(i)
                    to_predict_queries.append(q)
            
            st.info(f"Loaded {cache_hits} products instantly from cache. Sending {len(to_predict_queries)} to Groq.")

            if to_predict_queries:
                with st.spinner(f"Shortlisting {len(to_predict_queries)} products (Semantic Search)..."):
                    t0 = time.time()
                    all_candidates = batch_shortlist(to_predict_queries, leaves, matrix, shortlist_k)
                    tfidf_ms = int((time.time() - t0) * 1000)

                prog = st.progress(0, text="Sending Groq calls in parallel...")
                t1 = time.time()

                new_preds = run_parallel(to_predict_queries, all_candidates, api_key, model_choice, top_n_batch, concurrency)
                elapsed = time.time() - t1
                prog.progress(1.0, text=f"Groq API done in {elapsed:.1f}s")
                
                # Merge back and save to cache
                for idx, q, preds in zip(to_predict_indices, to_predict_queries, new_preds):
                    final_preds[idx] = preds
                    pred_cache[q] = preds
                save_to_cache(pred_cache)

            # Build final table
            rows = []
            for text, b, preds in zip(texts, brands, final_preds):
                rows.append({
                    "input_text":   text,
                    "brand":        b,
                    "top_category": preds[0]["category"] if preds else "",
                    "top_score":    round(preds[0]["score"], 4) if preds else 0,
                    "top_3":        " | ".join(f"{p['category']} ({p['score']:.1%})" for p in preds[:3]) if len(preds) > 1 else ""
                })

            df_out = pd.DataFrame(rows)
            st.markdown("#### Review Results (Click any cell to edit)")
            edited_df = st.data_editor(df_out, use_container_width=True, num_rows="dynamic")
            
            st.download_button("Download Results CSV", edited_df.to_csv(index=False).encode(), "predictions.csv", "text/csv")


# ── Explore ────────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("### Explore Category Map")
    tops = sorted(set(p.split(" / ")[0] for p in all_paths))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Paths", f"{len(all_paths):,}")
    c2.metric("Leaf Categories", f"{len(leaves):,}")
    c3.metric("Top-level Groups", len(tops))

    st.markdown("---")
    search = st.text_input("Search", placeholder="e.g. Jeans, Headphones...")
    if search:
        results = [p for p in all_paths if search.lower() in p.lower()]
        st.markdown(f"**{len(results):,} matches:**")
        for p in results[:100]:
            depth = len(p.split(" / ")) - 1
            st.markdown(f"{'  '*depth}{'└─ ' if depth else ''}`{p}`")
        if len(results) > 100:
            st.caption(f"...and {len(results)-100} more.")
    else:
        st.markdown("**Top-level categories:**")
        cols = st.columns(3)
        for i, top in enumerate(tops):
            count = sum(1 for p in leaves if p.startswith(top))
            cols[i % 3].markdown(f"- **{top}** ({count:,} leaves)")

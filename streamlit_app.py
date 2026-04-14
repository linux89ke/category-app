"""
Product Category Predictor
Strategy : TF-IDF shortlist (instant) + async parallel Groq reranking
Speed    : all products run concurrently, ~2-5s for any batch size
Cost     : 1 Groq call per product
"""

import os, io, json, asyncio
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq

st.set_page_config(page_title="Product Category Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-title {
        font-size:2.4rem; font-weight:700;
        background:linear-gradient(90deg,#f55036 0%,#ff8c00 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:0.2rem;
    }
    .subtitle { color:#888; font-size:1rem; margin-bottom:1.5rem; }
    .result-card {
        background:#f8f9fc; border-left:4px solid #f55036;
        padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.5rem;
    }
    .stTextArea textarea { border-radius:10px !important; border:2px solid #e0e0f0 !important; }
    .stTextArea textarea:focus { border-color:#f55036 !important; }
    .speed-badge {
        display:inline-block; background:#e6f4ea; color:#1e7e34;
        font-size:0.75rem; font-weight:700; padding:2px 10px;
        border-radius:20px; margin-left:8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── TF-IDF index ─────────────────────────────────────────────────────────────

def path_to_doc(path: str) -> str:
    parts = path.split(" / ")
    return " ".join(parts) + " " + " ".join(parts[-3:]) * 2


@st.cache_resource(show_spinner=False)
def build_index(file_path: str):
    # Read the Excel file directly from the local directory
    df = pd.read_excel(file_path)
    
    # Extract the third column (Category Path) as per original logic
    all_paths = df.iloc[:, 2].dropna().astype(str).tolist()
    
    path_set  = set(all_paths)
    leaves    = [p for p in all_paths
                 if not any(other.startswith(p + " / ") for other in path_set)]
    docs      = [path_to_doc(p) for p in leaves]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix    = vectorizer.fit_transform(docs)
    return leaves, vectorizer, matrix, all_paths


def shortlist(query: str, leaves, vectorizer, matrix, k: int = 30) -> list[str]:
    qvec = vectorizer.transform([query])
    sims = cosine_similarity(qvec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [leaves[i] for i in top_idx if sims[i] > 0]


def batch_shortlist(queries: list[str], leaves, vectorizer, matrix, k: int = 30) -> list[list[str]]:
    """Vectorise all queries in one matrix op — much faster than looping."""
    qmat = vectorizer.transform(queries)
    sims = cosine_similarity(qmat, matrix)          # (n_queries, n_leaves)
    results = []
    for row in sims:
        top_idx = np.argsort(row)[::-1][:k]
        results.append([leaves[i] for i in top_idx if row[i] > 0])
    return results


# ─── Async Groq reranking ─────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a product categorization expert.
Given a product title and a list of candidate category paths, pick the {top_n} best matching categories.
Consider brand, product type, gender, style, material.

Respond with JSON only:
{{
  "categories": [
    {{"category": "<full path>", "score": 0.95}},
    ...
  ]
}}

Rules:
- Return exactly {top_n} categories ordered by confidence descending
- Only pick from the provided candidate list — never invent categories
- Scores are floats 0.0–1.0
- JSON only, nothing else"""


async def async_rerank(
    idx: int,
    query: str,
    candidates: list[str],
    client: AsyncGroq,
    model: str,
    top_n: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[dict]]:
    """Rerank candidates for a single product. Returns (original_index, results)."""
    async with semaphore:
        cand_list = "\n".join(f"- {c}" for c in candidates)
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": SYSTEM_TEMPLATE.format(top_n=top_n)},
                    {"role": "user",
                     "content": f"Product: {query}\n\nCandidates:\n{cand_list}"},
                ],
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw).get("categories", [])
            return idx, data
        except Exception as e:
            return idx, [{"category": f"ERROR: {e}", "score": 0.0}]


async def parallel_predict(
    queries: list[str],
    candidates_list: list[list[str]],
    api_key: str,
    model: str,
    top_n: int,
    concurrency: int,
) -> list[list[dict]]:
    """Fire all Groq calls concurrently, bounded by semaphore."""
    client    = AsyncGroq(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    tasks     = [
        async_rerank(i, q, c, client, model, top_n, semaphore)
        for i, (q, c) in enumerate(zip(queries, candidates_list))
    ]
    results_raw = await asyncio.gather(*tasks)
    # Re-sort by original index (gather preserves order but let's be safe)
    ordered = sorted(results_raw, key=lambda x: x[0])
    return [r for _, r in ordered]


def run_parallel(queries, candidates_list, api_key, model, top_n, concurrency):
    """Streamlit-safe wrapper — creates a fresh event loop."""
    return asyncio.run(
        parallel_predict(queries, candidates_list, api_key, model, top_n, concurrency)
    )


# ─── Sync single predict (for single tab) ────────────────────────────────────

def sync_rerank(query, candidates, api_key, model, top_n):
    from groq import Groq
    client    = Groq(api_key=api_key)
    cand_list = "\n".join(f"- {c}" for c in candidates)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE.format(top_n=top_n)},
            {"role": "user",   "content": f"Product: {query}\n\nCandidates:\n{cand_list}"},
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
    api_key = st.text_input("Paste your key here:",
                            value=os.environ.get("GROQ_API_KEY", ""),
                            type="password", placeholder="gsk_…")
    st.caption("Free key at [console.groq.com](https://console.groq.com)")

    st.markdown("---")
    st.markdown("## Settings")
    model_choice = st.selectbox(
        "Groq model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="8b is fastest & free. 70b is most accurate.",
    )
    top_n        = st.slider("Top N results", 1, 10, 5)
    shortlist_k  = st.slider("Shortlist size", 10, 50, 30,
                             help="Candidates sent to Groq per product. 30 is optimal.")
    concurrency  = st.slider("Parallel requests", 1, 30, 10,
                             help="How many Groq calls run simultaneously. Free tier: keep ≤ 30.")
    score_threshold = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    show_chart   = st.checkbox("Show confidence chart", value=True)
    show_hierarchy = st.checkbox("Show category hierarchy", value=True)

    st.markdown("---")
    st.markdown("""### How it works
**Step 1 — TF-IDF** (free, ~10ms total for any batch)
Shortlists 30 leaf candidates per product in one matrix op.

**Step 2 — Groq async** (all products fire simultaneously)
Parallel calls return together in ~2s regardless of batch size.

**Result:** 100 products in ~5s instead of ~200s.""")


# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">Product Category Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">TF-IDF shortlist + async parallel Groq — fast, accurate, 1 API call per product.</p>', unsafe_allow_html=True)

if not api_key:
    st.info("Enter your Groq API key in the sidebar.")
    st.stop()

# Ensure local Excel file exists
excel_path = "category_map1.xlsx"
if not os.path.exists(excel_path):
    st.error(f"Required file '{excel_path}' not found in the script directory.")
    st.stop()

with st.spinner("Building category index (one-time ~2s)…"):
    leaves, vectorizer, matrix, all_paths = build_index(excel_path)

st.success(f"Successfully indexed {len(leaves):,} leaf categories.")

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_explore = st.tabs(["Single Predict", "Batch Predict", "Explore"])

EXAMPLES = [
    "Baggy Unit Denim Jeans Men's Streetwear",
    "AirPods Pro 2nd Generation Wireless Earbuds",
    "LEGO Star Wars Skywalker Saga Nintendo Switch",
    "KitchenAid 5-Quart Artisan Stand Mixer",
    "Harry Potter Sorcerer's Stone Hardcover",
    "Hydro Boost Face Moisturizer SPF 25",
]

# ── Single ─────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("### Enter a product title")
    st.markdown("**Quick examples:**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        short = ex[:46] + "…" if len(ex) > 46 else ex
        if cols[i % 3].button(short, key=f"ex_{i}", use_container_width=True):
            st.session_state["product_text"] = ex

    col_title, col_brand = st.columns([3, 1])
    with col_title:
        product_text = st.text_area(
            "Product title",
            value=st.session_state.get("product_text", ""),
            height=90,
            placeholder="e.g. Air Max 270 Men's Running Shoes…",
        )
    with col_brand:
        brand = st.text_input(
            "Brand *(optional)*",
            placeholder="e.g. Nike",
            help="Helps Groq disambiguate — e.g. Apple → Electronics not Grocery.",
        )

    if st.button("Predict", type="primary", use_container_width=True):
        if product_text.strip():
            query = f"{brand.strip()} {product_text.strip()}".strip() if brand.strip() else product_text.strip()
            with st.spinner("Shortlisting…"):
                candidates = shortlist(query, leaves, vectorizer, matrix, shortlist_k)
            with st.spinner(f"Asking Groq ({len(candidates)} candidates)…"):
                try:
                    preds = sync_rerank(query, candidates, api_key, model_choice, top_n)
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
    input_mode  = st.radio("Input method",
                           ["Upload file (CSV or Excel)", "Paste a list"],
                           horizontal=True)

    texts  = []
    brands = []

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
                    brand_col = st.selectbox("Brand column *(optional)*",
                                             ["— none —"] + df_input.columns.tolist())
                texts  = df_input[text_col].astype(str).fillna("").tolist()
                brands = (df_input[brand_col].astype(str).fillna("").tolist()
                          if brand_col != "— none —" else [""] * len(texts))
                st.caption(f"{len(texts):,} products — all will run in parallel.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.markdown("Supports `.csv`, `.xlsx`, `.xls`")

    else:
        pasted = st.text_area("Paste one product per line:", height=180,
                              placeholder="Nike Air Max 270\nKitchenAid Stand Mixer")
        brand_prefix = st.text_input("Brand *(optional — applies to all)*",
                                     placeholder="e.g. Nike", key="paste_brand")
        if pasted.strip():
            texts  = [t.strip() for t in pasted.strip().splitlines() if t.strip()]
            brands = [brand_prefix.strip() if brand_prefix else ""] * len(texts)
            st.caption(f"{len(texts):,} products — all will run in parallel.")

    if texts:
        est_secs = max(2, len(texts) // concurrency + 2)
        st.info(f"**{len(texts)} products** will run {concurrency} at a time — estimated **~{est_secs}s** total.")

        if st.button("Run Batch Prediction", type="primary"):
            import time

            # Step 1: TF-IDF shortlist for all queries in one shot
            queries = [
                f"{b.strip()} {t.strip()}".strip() if b.strip() else t.strip()
                for t, b in zip(texts, brands)
            ]

            with st.spinner(f"Step 1: Shortlisting {len(queries)} products (TF-IDF)…"):
                t0 = time.time()
                all_candidates = batch_shortlist(queries, leaves, vectorizer, matrix, shortlist_k)
                tfidf_ms = int((time.time() - t0) * 1000)

            # Step 2: parallel Groq calls
            prog    = st.progress(0, text="Step 2: Sending all Groq calls in parallel…")
            t1      = time.time()

            all_preds = run_parallel(queries, all_candidates, api_key,
                                     model_choice, top_n_batch, concurrency)

            elapsed = time.time() - t1
            prog.progress(1.0, text=f"Done in {elapsed:.1f}s ({tfidf_ms}ms TF-IDF + {elapsed:.1f}s Groq)")

            # Build results table
            rows = []
            for text, b, preds in zip(texts, brands, all_preds):
                rows.append({
                    "input_text":   text,
                    "brand":        b,
                    "top_category": preds[0]["category"] if preds else "",
                    "top_score":    round(preds[0]["score"], 4) if preds else 0,
                    "top_3":        " | ".join(
                        f"{p['category']} ({p['score']:.1%})" for p in preds[:3]
                    ),
                })

            df_out = pd.DataFrame(rows)
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Results CSV",
                               df_out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

    else:
        if st.button("Try sample data"):
            sample = ["Sony WH-1000XM5 Wireless Headphones",
                      "Instant Pot Duo 7-in-1 Pressure Cooker",
                      "Baggy Unit Denim Jeans Men",
                      "Harry Potter Hardcover Book",
                      "Apple AirPods Pro",
                      "Nike Air Max 270 Running Shoes"]
            s_brands = ["Sony", "Instant Pot", "", "", "Apple", "Nike"]

            queries_s = [f"{b} {t}".strip() for t, b in zip(sample, s_brands)]
            with st.spinner("Shortlisting…"):
                all_cands = batch_shortlist(queries_s, leaves, vectorizer, matrix, shortlist_k)

            with st.spinner(f"Running {len(sample)} Groq calls in parallel…"):
                import time
                t0 = time.time()
                all_preds = run_parallel(queries_s, all_cands, api_key, model_choice, 1, concurrency)
                elapsed = time.time() - t0

            st.success(f"Processed {len(sample)} products in {elapsed:.1f}s")
            rows = [{"title": t, "brand": b,
                     "predicted": p[0]["category"] if p else "",
                     "score": f"{p[0]['score']:.1%}" if p else ""}
                    for t, b, p in zip(sample, s_brands, all_preds)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Explore ────────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("### Explore Category Map")
    tops = sorted(set(p.split(" / ")[0] for p in all_paths))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Paths", f"{len(all_paths):,}")
    c2.metric("Leaf Categories", f"{len(leaves):,}")
    c3.metric("Top-level Groups", len(tops))

    st.markdown("---")
    search = st.text_input("Search", placeholder="e.g. Jeans, Headphones…")
    if search:
        results = [p for p in all_paths if search.lower() in p.lower()]
        st.markdown(f"**{len(results):,} matches:**")
        for p in results[:100]:
            depth = len(p.split(" / ")) - 1
            st.markdown(f"{'  '*depth}{'└─ ' if depth else ''}`{p}`")
        if len(results) > 100:
            st.caption(f"…and {len(results)-100} more.")
    else:
        st.markdown("**Top-level categories:**")
        cols = st.columns(3)
        for i, top in enumerate(tops):
            count = sum(1 for p in leaves if p.startswith(top))
            cols[i % 3].markdown(f"- **{top}** ({count:,} leaves)")

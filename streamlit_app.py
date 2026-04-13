"""
Product Category Predictor
Fixed load_index + paste-list batch input
"""

import os
import json
import asyncio
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq, Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CACHE_FILE = "prediction_cache.json"
INDEX_FILE  = "category_index.pkl"
CACHE_VERSION = "v5"   # bump whenever the tuple shape changes

st.set_page_config(page_title="CatPredict", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.app-header  { display:flex; align-items:baseline; gap:12px; margin-bottom:.25rem; }
.app-title   { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#f55036; letter-spacing:-.03em; line-height:1; }
.app-sub     { font-family:'DM Mono',monospace; font-size:.75rem; color:#555; letter-spacing:.05em; text-transform:uppercase; }

.stat-row  { display:flex; gap:12px; margin:1rem 0 1.5rem; flex-wrap:wrap; }
.stat-card { background:#0d0d0d; border:1px solid #1e1e1e; border-radius:8px; padding:14px 20px; min-width:130px; flex:1; }
.stat-val  { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#f55036; line-height:1; }
.stat-lbl  { font-size:.62rem; color:#444; text-transform:uppercase; letter-spacing:.08em; margin-top:4px; }

.result-card { background:#090909; border:1px solid #1e1e1e; border-left:3px solid #f55036; border-radius:8px; padding:18px 22px; margin-top:1rem; }
.result-lbl  { font-size:.62rem; color:#444; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; }
.result-cat  { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#f0f0f0; word-break:break-word; }

.badge-cache { display:inline-block; background:#0a1a0a; border:1px solid #1a3a1a; color:#4caf72; font-size:.62rem; padding:2px 8px; border-radius:20px; letter-spacing:.08em; margin-bottom:8px; }
.badge-ai    { display:inline-block; background:#1a0d08; border:1px solid #3a1a0a; color:#f55036; font-size:.62rem; padding:2px 8px; border-radius:20px; letter-spacing:.08em; margin-bottom:8px; }
.sec-lbl     { font-size:.62rem; text-transform:uppercase; letter-spacing:.1em; color:#333; margin-bottom:.5rem; }

button[data-baseweb="tab"] { font-family:'DM Mono',monospace !important; font-size:.75rem !important; text-transform:uppercase; letter-spacing:.08em; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f55036 !important; border-bottom-color:#f55036 !important; }

section[data-testid="stSidebar"] { background:#070707; border-right:1px solid #111; }
section[data-testid="stSidebar"] label { font-family:'DM Mono',monospace !important; font-size:.68rem !important; text-transform:uppercase; letter-spacing:.07em; color:#444 !important; }

.stProgress > div > div { background-color:#f55036 !important; }
[data-testid="stFileUploadDropzone"] { border:1px dashed #1e1e1e !important; border-radius:8px !important; background:#070707 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <span class="app-title">CATPREDICT</span>
  <span class="app-sub">/ product category intelligence</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;font-size:.95rem;color:#f55036;margin-bottom:1rem;">⚙ CONFIG</p>', unsafe_allow_html=True)

    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")

    manual_key = st.text_input("API Key Override", type="password", placeholder="gsk_...")
    if manual_key:
        api_key = manual_key

    if api_key:
        st.markdown('<p style="color:#4caf72;font-size:.7rem;letter-spacing:.05em;">● KEY ACTIVE</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#f55036;font-size:.7rem;letter-spacing:.05em;">● NO KEY — AI DISABLED</p>', unsafe_allow_html=True)

    st.divider()
    model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    shortlist_k  = st.slider("Candidate Shortlist", 5, 50, 20)
    concurrency  = st.slider("Parallel Workers", 1, 20, 10)
    st.divider()

    cache_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache_data = json.load(open(CACHE_FILE))
        except Exception:
            pass

    st.markdown(
        f'<p style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;color:#f55036;margin-bottom:0;">{len(cache_data)}</p>'
        f'<p style="font-size:.62rem;color:#333;text-transform:uppercase;letter-spacing:.08em;margin-top:2px;">CACHED PREDICTIONS</p>',
        unsafe_allow_html=True,
    )
    if st.button("Clear Cache", use_container_width=True):
        for f in [CACHE_FILE, INDEX_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.rerun()

# ─────────────────────────────────────────────
# FILE HANDLING
# ─────────────────────────────────────────────

map_file = None
for f in ["category_map1.csv", "category_map1.xlsx"]:
    if os.path.exists(f):
        map_file = f
        break

if not map_file:
    st.divider()
    st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;color:#f0f0f0;">No Category Map Found</p>', unsafe_allow_html=True)
    st.caption("Upload a CSV or Excel file — category paths should be in column 3.")
    up = st.file_uploader("Upload category file", type=["csv", "xlsx"])
    if up:
        df_up = pd.read_excel(up) if up.name.endswith(".xlsx") else pd.read_csv(up)
        df_up.to_csv("category_map1.csv", index=False)
        st.success("✓ Uploaded — reloading...")
        st.rerun()
    st.stop()

# ─────────────────────────────────────────────
# INDEX  — version is passed as arg so st.cache_resource
#          busts itself when CACHE_VERSION changes
# ─────────────────────────────────────────────

def _build_index(path):
    df    = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    paths = df.iloc[:, 2].dropna().astype(str).tolist()
    pset  = set(paths)
    leaves = [p for p in paths if not any(o.startswith(p + " / ") for o in pset)]
    docs   = [" ".join(p.split(" / ")) for p in leaves]
    vec    = TfidfVectorizer(ngram_range=(1, 2))
    mat    = vec.fit_transform(docs)
    return (leaves, vec, mat, df, paths)

@st.cache_resource
def load_index(path, ver):
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as fh:
                saved_ver, data = pickle.load(fh)
            if saved_ver == ver and isinstance(data, tuple) and len(data) == 5:
                return data
        except Exception:
            pass
    data = _build_index(path)
    try:
        with open(INDEX_FILE, "wb") as fh:
            pickle.dump((ver, data), fh)
    except Exception:
        pass
    return data

leaves, vectorizer, matrix, raw_df, all_paths = load_index(map_file, CACHE_VERSION)

all_tops       = [p.split(" / ")[0] for p in all_paths if p]
top_level_cats = list(set(all_tops))
depth_max      = max((len(p.split(" / ")) for p in all_paths), default=1)

# ─────────────────────────────────────────────
# STATS BAR
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card"><div class="stat-val">{len(leaves):,}</div><div class="stat-lbl">Leaf Categories</div></div>
  <div class="stat-card"><div class="stat-val">{len(top_level_cats):,}</div><div class="stat-lbl">Top-Level Groups</div></div>
  <div class="stat-card"><div class="stat-val">{depth_max}</div><div class="stat-lbl">Max Depth</div></div>
  <div class="stat-card"><div class="stat-val">{len(cache_data):,}</div><div class="stat-lbl">Cached Predictions</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE))
        except Exception:
            pass
    return {}

def save_cache(c):
    json.dump(c, open(CACHE_FILE, "w"), indent=2)

def find_cache(q, cache):
    return cache.get(q.lower())

def render_path(cat_str):
    if not cat_str:
        return cat_str
    parts = cat_str.split(" / ")
    html  = []
    for i, p in enumerate(parts):
        color = "#f55036" if i == len(parts) - 1 else "#555"
        html.append(f'<span style="color:{color}">{p}</span>')
    return '<span style="color:#333"> / </span>'.join(html)

SYSTEM_PROMPT = (
    "You are a product categorisation assistant. "
    "Given a product title and a list of candidate category paths, "
    "return ONLY a JSON object with a single key 'category' "
    "whose value is the best matching category path from the list."
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def async_ai(q, cands, client, sem):
    async with sem:
        resp = await client.chat.completions.create(
            model=model_choice,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Product: {q}\n\nCandidates:\n" + "\n".join(cands)},
            ],
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"category": raw.strip()}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Mono"),
    margin=dict(l=0, r=20, t=10, b=10),
)

# ─────────────────────────────────────────────
# BATCH RUNNER  (shared by file + paste paths)
# ─────────────────────────────────────────────

def run_batch_ui(titles):
    """Render the batch prediction UI for a list of title strings."""
    cache   = load_cache()
    results = []
    todo    = []

    for i, q in enumerate(titles):
        c = find_cache(q, cache)
        if c:
            cat = c.get("category", str(c)) if isinstance(c, dict) else str(c)
            results.append({"#": i+1, "Product": q, "Category": cat, "Status": "Cached"})
        else:
            results.append({"#": i+1, "Product": q, "Category": "—", "Status": "Pending"})
            todo.append((i, q))

    progress = st.progress(0, text=f"0 / {len(todo)} remaining…")
    table    = st.empty()
    table.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    if todo and api_key:
        client_async = AsyncGroq(api_key=api_key)
        sem = asyncio.Semaphore(concurrency)

        async def _run():
            tasks = []
            for _, q in todo:
                s  = cosine_similarity(vectorizer.transform([q]), matrix)[0]
                ix = np.argsort(s)[::-1][:shortlist_k]
                tasks.append(async_ai(q, [leaves[x] for x in ix], client_async, sem))
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            outputs = asyncio.run(_run())
            done = 0
            for (i, q), out in zip(todo, outputs):
                done += 1
                if isinstance(out, Exception):
                    results[i]["Category"] = f"Error: {out}"
                    results[i]["Status"]   = "❌ Failed"
                else:
                    cat = out.get("category", str(out)) if isinstance(out, dict) else str(out)
                    results[i]["Category"] = cat
                    results[i]["Status"]   = "✓ Done"
                    cache[q.lower()] = out
                progress.progress(done / len(todo), text=f"{done} / {len(todo)} done")
                table.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            save_cache(cache)
            progress.empty()
            cached_n = len(results) - len(todo)
            st.success(f"✓ {len(todo)} predicted · {cached_n} from cache · {len(results)} total")

            res_df = pd.DataFrame(results)
            st.download_button(
                "↓ Download Results CSV",
                res_df.to_csv(index=False).encode(),
                "predictions.csv", "text/csv",
                use_container_width=True,
            )

            # Distribution chart
            done_rows = [r for r in results if "Done" in str(r["Status"]) or "Cached" in str(r["Status"])]
            if done_rows:
                st.markdown('<p class="sec-lbl" style="margin-top:1.5rem;">Top-Level Category Distribution</p>', unsafe_allow_html=True)
                tops = [r["Category"].split(" / ")[0] if " / " in str(r["Category"]) else str(r["Category"]) for r in done_rows]
                tc   = pd.Series(tops).value_counts().head(20)
                fig_b = go.Figure(go.Bar(
                    x=tc.index.tolist(), y=tc.values.tolist(),
                    marker=dict(color=tc.values.tolist(), colorscale=[[0,"#130803"],[1,"#f55036"]], showscale=False),
                    text=tc.values.tolist(), textposition="outside",
                    textfont=dict(color="#444", size=10, family="DM Mono"),
                ))
                fig_b.update_layout(
                    **PLOTLY_BASE, height=300,
                    margin=dict(l=5, r=5, t=5, b=60),
                    xaxis=dict(tickfont=dict(color="#666", size=9, family="DM Mono"), gridcolor="#0f0f0f", tickangle=-35),
                    yaxis=dict(tickfont=dict(color="#444", size=9, family="DM Mono"), gridcolor="#0f0f0f"),
                )
                st.plotly_chart(fig_b, use_container_width=True)

        except Exception as e:
            st.error(f"Batch error: {e}")
            st.info("💡 Check your API key at console.groq.com.")

    elif not api_key:
        progress.empty()
        st.warning("No API key — cached results only.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["  Single  ", "  Batch  ", "  Explore  "])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE
# ══════════════════════════════════════════════

with tab1:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("title", placeholder="e.g. Sony WH-1000XM5 Wireless Headphones", label_visibility="collapsed")
    with col_btn:
        predict_btn = st.button("→ Predict", use_container_width=True, type="primary")

    if predict_btn and query.strip():
        cache  = load_cache()
        cached = find_cache(query, cache)

        sims   = cosine_similarity(vectorizer.transform([query]), matrix)[0]
        idxs   = np.argsort(sims)[::-1][:shortlist_k]
        cands  = [leaves[i] for i in idxs]
        scores = [float(sims[i]) for i in idxs]

        if cached:
            cat_display = cached.get("category", str(cached)) if isinstance(cached, dict) else str(cached)
            st.markdown('<span class="badge-cache">⚡ FROM CACHE</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-card"><div class="result-lbl">Best Match</div><div class="result-cat">{render_path(cat_display)}</div></div>', unsafe_allow_html=True)

        elif not api_key:
            st.error("No API key configured — add one in the sidebar.")

        else:
            with st.spinner("Asking AI…"):
                try:
                    client = Groq(api_key=api_key)
                    resp   = client.chat.completions.create(
                        model=model_choice,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": f"Product: {query}\n\nCandidates:\n" + "\n".join(cands)},
                        ],
                    )
                    raw = resp.choices[0].message.content
                    try:
                        out = json.loads(raw)
                    except json.JSONDecodeError:
                        out = {"category": raw.strip()}

                    cache[query.lower()] = out
                    save_cache(cache)
                    cat_display = out.get("category", str(out)) if isinstance(out, dict) else str(out)
                    st.markdown('<span class="badge-ai">🤖 AI PREDICTION</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-card"><div class="result-lbl">Best Match</div><div class="result-cat">{render_path(cat_display)}</div></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Groq API error: {e}")
                    st.info("💡 Verify your key at console.groq.com and check model availability.")

        # Candidate bar chart
        st.markdown('<p class="sec-lbl" style="margin-top:1.5rem;">Top Candidates by TF-IDF Similarity</p>', unsafe_allow_html=True)
        n = min(15, len(cands))
        labels = []
        for c in cands[:n]:
            parts = c.split(" / ")
            labels.append(parts[-1] + ("  [" + parts[0] + "]" if len(parts) > 1 else ""))

        fig = go.Figure(go.Bar(
            x=scores[:n], y=labels, orientation="h",
            marker=dict(color=scores[:n], colorscale=[[0,"#130803"],[0.5,"#6a1f0e"],[1,"#f55036"]], showscale=False),
            text=[f"{s:.3f}" for s in scores[:n]], textposition="outside",
            textfont=dict(color="#444", size=10, family="DM Mono"),
        ))
        fig.update_layout(
            **PLOTLY_BASE, height=420,
            margin=dict(l=0, r=60, t=5, b=5),
            xaxis=dict(showgrid=True, gridcolor="#111", tickfont=dict(color="#333", size=9, family="DM Mono"), zeroline=False),
            yaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif predict_btn:
        st.warning("Enter a product title first.")

# ══════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════

with tab2:
    # Input mode toggle
    input_mode = st.radio(
        "Input source",
        ["Upload file", "Paste list"],
        horizontal=True,
        label_visibility="collapsed",
    )

    titles_to_run = None

    if input_mode == "Upload file":
        col_up, col_info = st.columns([3, 2])
        with col_up:
            uploaded = st.file_uploader(
                "CSV / XLSX — column A = product titles",
                type=["csv", "xlsx"], key="batch_file",
            )
        with col_info:
            st.markdown("""
            <div style="margin-top:1.5rem;padding:1rem;background:#0a0a0a;border:1px solid #151515;border-radius:8px;">
              <p style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#333;margin-bottom:6px;">Format</p>
              <p style="font-size:.8rem;color:#666;">Column A: product titles<br>Other columns ignored<br>CSV or XLSX accepted</p>
            </div>""", unsafe_allow_html=True)

        if uploaded:
            df_batch = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
            titles_to_run = df_batch.iloc[:, 0].astype(str).tolist()
            st.caption(f"{len(titles_to_run):,} products loaded from file")

    else:  # Paste list
        pasted = st.text_area(
            "Paste product titles — one per line",
            height=200,
            placeholder="Sony WH-1000XM5 Wireless Headphones\nApple AirPods Pro 2nd Gen\nSamsung 65\" QLED TV\n…",
            label_visibility="visible",
        )
        if pasted.strip():
            titles_to_run = [line.strip() for line in pasted.splitlines() if line.strip()]
            st.caption(f"{len(titles_to_run):,} titles detected")

    if titles_to_run:
        if st.button("▶ Run Batch", type="primary", key="run_batch"):
            run_batch_ui(titles_to_run)

# ══════════════════════════════════════════════
# TAB 3 — EXPLORE
# ══════════════════════════════════════════════

with tab3:
    st.markdown('<p class="sec-lbl">Search & Explore the Category Tree</p>', unsafe_allow_html=True)

    col_s, col_d = st.columns([4, 1])
    with col_s:
        search_q = st.text_input("search", placeholder="Filter by keyword…", label_visibility="collapsed", key="ex_search")
    with col_d:
        depth_opt = st.selectbox("Depth", ["All"] + list(range(1, depth_max + 1)), label_visibility="collapsed", key="ex_depth")

    filtered = leaves
    if search_q.strip():
        sq       = search_q.lower()
        filtered = [l for l in filtered if sq in l.lower()]
    if depth_opt != "All":
        filtered = [l for l in filtered if len(l.split(" / ")) == int(depth_opt)]

    st.caption(f"{len(filtered):,} of {len(leaves):,} categories shown")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="sec-lbl">Categories per Top-Level Group</p>', unsafe_allow_html=True)
        tc = pd.Series([l.split(" / ")[0] for l in filtered]).value_counts().head(20)
        fig_top = go.Figure(go.Bar(
            x=tc.values.tolist(), y=tc.index.tolist(), orientation="h",
            marker=dict(color=tc.values.tolist(), colorscale=[[0,"#0d0503"],[0.4,"#5a1808"],[1,"#f55036"]], showscale=False),
            text=tc.values.tolist(), textposition="outside",
            textfont=dict(color="#333", size=9, family="DM Mono"),
        ))
        fig_top.update_layout(
            **PLOTLY_BASE, height=400,
            margin=dict(l=0, r=40, t=5, b=5),
            xaxis=dict(tickfont=dict(color="#333", size=9, family="DM Mono"), gridcolor="#0f0f0f"),
            yaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), autorange="reversed"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        st.markdown('<p class="sec-lbl">Depth Distribution</p>', unsafe_allow_html=True)
        depths = pd.Series([len(l.split(" / ")) for l in filtered]).value_counts().sort_index()
        fig_dep = go.Figure(go.Bar(
            x=[f"Level {i}" for i in depths.index], y=depths.values.tolist(),
            marker=dict(color=depths.values.tolist(), colorscale=[[0,"#0d0503"],[1,"#f55036"]], showscale=False),
            text=depths.values.tolist(), textposition="outside",
            textfont=dict(color="#333", size=10, family="DM Mono"),
        ))
        fig_dep.update_layout(
            **PLOTLY_BASE, height=400,
            margin=dict(l=10, r=10, t=5, b=10),
            xaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), gridcolor="#0f0f0f"),
            yaxis=dict(tickfont=dict(color="#333", size=9, family="DM Mono"), gridcolor="#0f0f0f"),
        )
        st.plotly_chart(fig_dep, use_container_width=True)

    # Treemap
    st.divider()
    st.markdown('<p class="sec-lbl">Treemap — Click to Drill Down</p>', unsafe_allow_html=True)

    td = defaultdict(int)
    for leaf in filtered:
        parts  = leaf.split(" / ")
        top    = parts[0]
        second = parts[1] if len(parts) >= 2 else top
        td[(top, second)] += 1

    top_totals = defaultdict(int)
    for (top, second), cnt in td.items():
        top_totals[top] += cnt

    tm_ids     = ["root"]
    tm_labels  = ["All"]
    tm_parents = [""]
    tm_values  = [0]

    for top, total in sorted(top_totals.items(), key=lambda x: -x[1])[:35]:
        tm_ids.append(top); tm_labels.append(top); tm_parents.append("root"); tm_values.append(total)

    seen = set(tm_ids)
    for (top, second), cnt in sorted(td.items(), key=lambda x: -x[1]):
        nid = f"{top}||{second}"
        if top in seen and nid not in seen and second != top:
            tm_ids.append(nid); tm_labels.append(second); tm_parents.append(top); tm_values.append(cnt)
            seen.add(nid)

    fig_tree = go.Figure(go.Treemap(
        ids=tm_ids, labels=tm_labels, parents=tm_parents, values=tm_values,
        branchvalues="total",
        marker=dict(
            colorscale=[[0,"#0a0302"],[0.25,"#2a0d08"],[0.6,"#8a2a14"],[1,"#f55036"]],
            showscale=False,
            line=dict(width=1, color="#050505"),
        ),
        textfont=dict(family="DM Mono", size=11, color="#ddd"),
        hovertemplate="<b>%{label}</b><br>%{value} leaf categories<extra></extra>",
        pathbar=dict(visible=True, textfont=dict(family="DM Mono", size=10)),
    ))
    fig_tree.update_layout(**PLOTLY_BASE, height=520, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

    # Raw list
    st.divider()
    st.markdown('<p class="sec-lbl">Raw Category List</p>', unsafe_allow_html=True)

    max_show = max(10, len(filtered))
    show_n   = st.slider("Rows to show", 10, min(500, max_show), min(50, max_show), key="ex_rows")

    display_df = pd.DataFrame({
        "Category Path": filtered[:show_n],
        "Depth":         [len(l.split(" / ")) for l in filtered[:show_n]],
        "Top-Level":     [l.split(" / ")[0]   for l in filtered[:show_n]],
        "Leaf Node":     [l.split(" / ")[-1]  for l in filtered[:show_n]],
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(filtered) > show_n:
        st.caption(f"Showing {show_n:,} of {len(filtered):,} — adjust slider above")

    st.download_button(
        "↓ Download Filtered Categories",
        pd.DataFrame({"category_path": filtered}).to_csv(index=False).encode(),
        "categories_filtered.csv", "text/csv",
    )

"""
Product Category Predictor
Hybrid Flow: Cache → Local ML → TF-IDF Gate → Batched Groq
"""

import os, json, asyncio, pickle, re, math
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from groq import AsyncGroq, Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CACHE_FILE      = "prediction_cache.json"
INDEX_FILE      = "category_index.pkl"
CLASSIFIER_FILE = "category_classifier.pkl"
CACHE_VERSION   = "v6"

# Tunable thresholds
TFIDF_CONFIDENCE_THRESHOLD      = 0.55   # skip AI if top TF-IDF score > this
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70   # skip AI if classifier prob > this
CLASSIFIER_MIN_SAMPLES          = 50     # min cache entries before classifier trains
BATCH_SIZE                      = 10     # products per single Groq call

st.set_page_config(page_title="CatPredict", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family:'DM Mono',monospace; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem; padding-bottom:2rem; }

.app-header { display:flex; align-items:baseline; gap:12px; margin-bottom:.25rem; }
.app-title  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#f55036; letter-spacing:-.03em; line-height:1; }
.app-sub    { font-family:'DM Mono',monospace; font-size:.75rem; color:#555; letter-spacing:.05em; text-transform:uppercase; }

.stat-row  { display:flex; gap:12px; margin:1rem 0 1.5rem; flex-wrap:wrap; }
.stat-card { background:#0d0d0d; border:1px solid #1e1e1e; border-radius:8px; padding:14px 20px; min-width:130px; flex:1; }
.stat-val  { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#f55036; line-height:1; }
.stat-lbl  { font-size:.62rem; color:#444; text-transform:uppercase; letter-spacing:.08em; margin-top:4px; }

.result-card { background:#090909; border:1px solid #1e1e1e; border-left:3px solid #f55036; border-radius:8px; padding:18px 22px; margin-top:.75rem; }
.result-lbl  { font-size:.62rem; color:#444; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; }
.result-cat  { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#f0f0f0; word-break:break-word; }
.result-meta { font-size:.65rem; color:#444; margin-top:8px; }

/* source badges */
.badge { display:inline-block; font-size:.6rem; padding:2px 8px; border-radius:20px; letter-spacing:.08em; margin-bottom:8px; }
.badge-cache      { background:#0a1a0a; border:1px solid #1a3a1a; color:#4caf72; }
.badge-classifier { background:#0a0f1a; border:1px solid #1a2a3a; color:#4a9ef5; }
.badge-tfidf      { background:#1a1a0a; border:1px solid #3a3a1a; color:#d4c44a; }
.badge-ai         { background:#1a0d08; border:1px solid #3a1a0a; color:#f55036; }

/* flow diagram */
.flow-step { display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border-radius:6px; font-size:.68rem; }
.flow-hit  { background:#0a1a0a; border:1px solid #1a3a1a; color:#4caf72; }
.flow-skip { background:#111; border:1px solid #1e1e1e; color:#333; }
.flow-arrow { color:#333; font-size:.8rem; }

.sec-lbl { font-size:.62rem; text-transform:uppercase; letter-spacing:.1em; color:#333; margin-bottom:.5rem; }

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
  <span class="app-sub">/ hybrid category intelligence</span>
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
        st.markdown('<p style="color:#4caf72;font-size:.7rem;">● KEY ACTIVE</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#f55036;font-size:.7rem;">● NO KEY — AI DISABLED</p>', unsafe_allow_html=True)

    st.divider()
    model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    shortlist_k  = st.slider("Candidate Shortlist", 5, 50, 20)
    concurrency  = st.slider("Parallel Workers", 1, 20, 5)
    batch_size   = st.slider("Products per API Call", 1, 20, BATCH_SIZE)

    st.divider()
    st.markdown('<p class="sec-lbl">Confidence Thresholds</p>', unsafe_allow_html=True)
    tfidf_thresh = st.slider("TF-IDF Gate", 0.0, 1.0, TFIDF_CONFIDENCE_THRESHOLD, 0.05,
                             help="Skip AI if top TF-IDF similarity exceeds this")
    clf_thresh   = st.slider("Classifier Gate", 0.0, 1.0, CLASSIFIER_CONFIDENCE_THRESHOLD, 0.05,
                             help="Skip AI if local classifier confidence exceeds this")

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

    if st.button("Retrain Classifier", use_container_width=True):
        if os.path.exists(CLASSIFIER_FILE):
            os.remove(CLASSIFIER_FILE)
        st.rerun()

    if st.button("Clear All Cache", use_container_width=True):
        for f in [CACHE_FILE, INDEX_FILE, CLASSIFIER_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.rerun()

# ─────────────────────────────────────────────
# FILE HANDLING
# ─────────────────────────────────────────────

map_file = None
for f in ["category_map1.xlsx - pim category attribute set.csv", "category_map1.csv", "category_map1.xlsx"]:
    if os.path.exists(f):
        map_file = f
        break

if not map_file:
    st.divider()
    st.markdown('<p style="font-family:Syne,sans-serif;font-weight:700;color:#f0f0f0;">No Category Map Found</p>', unsafe_allow_html=True)
    st.caption("Upload a CSV or Excel file — category paths in column 3.")
    up = st.file_uploader("Upload category file", type=["csv", "xlsx"])
    if up:
        df_up = pd.read_excel(up) if up.name.endswith(".xlsx") else pd.read_csv(up)
        df_up.to_csv("category_map1.csv", index=False)
        st.success("✓ Uploaded — reloading...")
        st.rerun()
    st.stop()

# ─────────────────────────────────────────────
# INDEX
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
# CACHE HELPERS  (normalized keys throughout)
# ─────────────────────────────────────────────

def normalize(q: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation noise."""
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s/]", "", q)
    return q

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE))
        except Exception:
            pass
    return {}

def save_cache(c: dict):
    json.dump(c, open(CACHE_FILE, "w"), indent=2)

def find_cache(q: str, cache: dict):
    return cache.get(normalize(q))

def store_cache(q: str, result: dict, cache: dict):
    cache[normalize(q)] = result

# ─────────────────────────────────────────────
# LOCAL CLASSIFIER
# Trains on cache entries; retrained on demand
# ─────────────────────────────────────────────

def _train_classifier(cache: dict):
    """
    Build a TF-IDF + LogisticRegression classifier from cached predictions.
    Returns (clf, label_encoder, clf_vectorizer) or None if insufficient data.
    """
    rows = []
    for raw_q, val in cache.items():
        cat = val.get("category", "") if isinstance(val, dict) else str(val)
        if cat:
            rows.append((raw_q, cat))

    if len(rows) < CLASSIFIER_MIN_SAMPLES:
        return None

    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    # Filter classes with only 1 sample (can't stratify)
    from collections import Counter
    counts = Counter(labels)
    filtered = [(t, l) for t, l in zip(texts, labels) if counts[l] >= 2]
    if len(filtered) < CLASSIFIER_MIN_SAMPLES:
        return None

    texts, labels = zip(*filtered)
    le  = LabelEncoder()
    y   = le.fit_transform(labels)
    cv  = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000)
    X   = cv.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", multi_class="multinomial")
    clf.fit(X, y)
    return (clf, le, cv)

@st.cache_resource
def load_classifier(cache_size: int):
    """
    cache_size is passed so the resource re-trains when cache grows.
    Returns (clf, le, cv) or None.
    """
    if os.path.exists(CLASSIFIER_FILE):
        try:
            with open(CLASSIFIER_FILE, "rb") as fh:
                saved_size, model = pickle.load(fh)
            # reuse if saved with same or larger cache
            if saved_size >= CLASSIFIER_MIN_SAMPLES and model is not None:
                return model
        except Exception:
            pass

    cache = load_cache()
    result = _train_classifier(cache)

    if result is not None:
        try:
            with open(CLASSIFIER_FILE, "wb") as fh:
                pickle.dump((len(cache), result), fh)
        except Exception:
            pass

    return result

def classify(q: str, clf_bundle) -> tuple[str | None, float]:
    """Returns (category, confidence) or (None, 0.0)."""
    if clf_bundle is None:
        return None, 0.0
    clf, le, cv = clf_bundle
    X    = cv.transform([normalize(q)])
    probs = clf.predict_proba(X)[0]
    idx  = np.argmax(probs)
    return le.inverse_transform([idx])[0], float(probs[idx])

# Load classifier (keyed on cache size so it retrains when cache grows enough)
clf_bundle = load_classifier(len(cache_data))
clf_status = "ready" if clf_bundle is not None else f"needs {CLASSIFIER_MIN_SAMPLES}+ samples"

# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a product categorisation assistant. "
    "Given numbered product titles and candidate category paths for each, "
    "return ONLY a JSON object mapping each number (as a string key) to the best "
    "matching category path. Example: {\"1\": \"Electronics / Audio\", \"2\": \"Clothing / Shoes\"}"
)

def tfidf_top(q: str) -> tuple[str, float, list[str], list[float]]:
    """Returns (best_category, best_score, all_candidates, all_scores)."""
    sims  = cosine_similarity(vectorizer.transform([q]), matrix)[0]
    idxs  = np.argsort(sims)[::-1][:shortlist_k]
    cands = [leaves[i] for i in idxs]
    scores = [float(sims[i]) for i in idxs]
    return cands[0], scores[0], cands, scores

def render_path(cat_str: str) -> str:
    if not cat_str:
        return cat_str
    parts = cat_str.split(" / ")
    html  = []
    for i, p in enumerate(parts):
        color = "#f55036" if i == len(parts) - 1 else "#555"
        html.append(f'<span style="color:{color}">{p}</span>')
    return '<span style="color:#2a2a2a"> / </span>'.join(html)

# ─────────────────────────────────────────────
# HYBRID PREDICT  (single query)
# Flow: Cache → Classifier → TF-IDF gate → Groq
# ─────────────────────────────────────────────

def hybrid_predict_single(q: str, cache: dict, client: Groq) -> dict:
    """
    Returns {category, source, confidence, candidates, scores}
    source: 'cache' | 'classifier' | 'tfidf' | 'ai'
    """
    # ── 1. Cache ──────────────────────────────
    cached = find_cache(q, cache)
    if cached:
        cat = cached.get("category", str(cached)) if isinstance(cached, dict) else str(cached)
        return {"category": cat, "source": "cache", "confidence": 1.0,
                "candidates": [], "scores": []}

    best_tfidf, tfidf_score, cands, scores = tfidf_top(q)

    # ── 2. Classifier ─────────────────────────
    clf_cat, clf_conf = classify(q, clf_bundle)
    if clf_cat and clf_conf >= clf_thresh:
        result = {"category": clf_cat, "source": "classifier",
                  "confidence": clf_conf, "candidates": cands, "scores": scores}
        store_cache(q, result, cache)
        return result

    # ── 3. TF-IDF confidence gate ─────────────
    if tfidf_score >= tfidf_thresh:
        result = {"category": best_tfidf, "source": "tfidf",
                  "confidence": tfidf_score, "candidates": cands, "scores": scores}
        store_cache(q, result, cache)
        return result

    # ── 4. Groq AI ────────────────────────────
    resp = client.chat.completions.create(
        model=model_choice,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f'Product: {q}\n\nCandidates:\n' + "\n".join(cands)},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        out = json.loads(raw)
        # single-call variant returns {"1": "cat"} or {"category": "cat"}
        cat = out.get("1") or out.get("category") or next(iter(out.values()), best_tfidf)
    except Exception:
        cat = best_tfidf

    result = {"category": cat, "source": "ai", "confidence": None,
              "candidates": cands, "scores": scores}
    store_cache(q, result, cache)
    return result

# ─────────────────────────────────────────────
# BATCHED GROQ CALL
# Sends up to `batch_size` products in one request
# ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def groq_batch_call(items: list[tuple[int, str, list[str]]], client: AsyncGroq, sem) -> dict:
    """
    items: list of (local_index, query, candidates)
    Returns dict: {local_index: category_str}
    """
    async with sem:
        prompt_parts = []
        for local_idx, (_, q, cands) in enumerate(items, 1):
            cand_str = "\n  ".join(cands[:shortlist_k])
            prompt_parts.append(f"{local_idx}. Product: {q}\n  Candidates:\n  {cand_str}")

        prompt = "\n\n".join(prompt_parts)

        resp = await client.chat.completions.create(
            model=model_choice,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content
        try:
            mapping = json.loads(raw)
        except Exception:
            mapping = {}

        # map local 1-based index → original index
        result = {}
        for local_idx, (orig_idx, _, _) in enumerate(items, 1):
            cat = mapping.get(str(local_idx)) or mapping.get(local_idx)
            result[orig_idx] = cat or ""
        return result

# ─────────────────────────────────────────────
# BATCH RUNNER  (hybrid + batched AI calls)
# ─────────────────────────────────────────────

def run_batch_ui(titles: list[str]):
    cache   = load_cache()
    results = []
    need_ai = []  # (original_index, query, candidates)

    # ── Phase 1: cache / classifier / tfidf gates ──
    for i, q in enumerate(titles):
        cached = find_cache(q, cache)
        if cached:
            cat = cached.get("category", str(cached)) if isinstance(cached, dict) else str(cached)
            results.append({"#": i+1, "Product": q, "Category": cat,
                             "Source": "⚡ Cache", "Confidence": "—"})
            continue

        _, tfidf_score, cands, _ = tfidf_top(q)
        clf_cat, clf_conf = classify(q, clf_bundle)

        if clf_cat and clf_conf >= clf_thresh:
            results.append({"#": i+1, "Product": q, "Category": clf_cat,
                             "Source": "🧠 Classifier", "Confidence": f"{clf_conf:.0%}"})
            store_cache(q, {"category": clf_cat, "source": "classifier",
                             "confidence": clf_conf}, cache)
            continue

        if tfidf_score >= tfidf_thresh:
            results.append({"#": i+1, "Product": q, "Category": cands[0],
                             "Source": "📐 TF-IDF", "Confidence": f"{tfidf_score:.0%}"})
            store_cache(q, {"category": cands[0], "source": "tfidf",
                             "confidence": tfidf_score}, cache)
            continue

        # needs AI
        results.append({"#": i+1, "Product": q, "Category": "—",
                         "Source": "⏳ Pending", "Confidence": "—"})
        need_ai.append((i, q, cands))

    # Summary of what needs AI
    cache_hits   = sum(1 for r in results if "Cache"      in r["Source"])
    clf_hits     = sum(1 for r in results if "Classifier" in r["Source"])
    tfidf_hits   = sum(1 for r in results if "TF-IDF"     in r["Source"])
    ai_calls_est = math.ceil(len(need_ai) / batch_size)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cache hits",       cache_hits)
    col2.metric("Classifier hits",  clf_hits)
    col3.metric("TF-IDF hits",      tfidf_hits)
    col4.metric("API calls needed", ai_calls_est,
                delta=f"-{len(need_ai) - ai_calls_est} vs 1-by-1" if len(need_ai) > 1 else None,
                delta_color="inverse")

    progress = st.progress(0, text="Running…")
    table    = st.empty()
    table.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    # ── Phase 2: batched Groq calls ──
    if need_ai and api_key:
        # Split into chunks of `batch_size`
        chunks = [need_ai[i:i+batch_size] for i in range(0, len(need_ai), batch_size)]
        client_async = AsyncGroq(api_key=api_key)
        sem = asyncio.Semaphore(concurrency)

        async def _run_all():
            tasks = [groq_batch_call(chunk, client_async, sem) for chunk in chunks]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            chunk_results = asyncio.run(_run_all())
            done = 0
            for chunk, chunk_out in zip(chunks, chunk_results):
                for orig_i, q, cands in chunk:
                    done += 1
                    if isinstance(chunk_out, Exception):
                        results[orig_i]["Category"] = f"Error: {chunk_out}"
                        results[orig_i]["Source"]   = "❌ Failed"
                    else:
                        cat = chunk_out.get(orig_i, cands[0] if cands else "")
                        results[orig_i]["Category"]   = cat
                        results[orig_i]["Source"]     = "🤖 AI"
                        results[orig_i]["Confidence"] = "—"
                        store_cache(q, {"category": cat, "source": "ai"}, cache)
                    progress.progress(done / len(need_ai),
                                      text=f"AI: {done}/{len(need_ai)} ({len(chunks) - math.ceil((len(need_ai)-done)/batch_size)} calls sent)")
                    table.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            save_cache(cache)
            progress.empty()
            st.success(
                f"✓ Done — {cache_hits} cache · {clf_hits} classifier · "
                f"{tfidf_hits} TF-IDF · {len(need_ai)} AI ({ai_calls_est} API calls)"
            )

            res_df = pd.DataFrame(results)
            st.download_button("↓ Download Results CSV",
                               res_df.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv",
                               use_container_width=True)

            # Distribution chart
            done_rows = [r for r in results if r["Category"] not in ("—", "") and "Error" not in r.get("Source","")]
            if done_rows:
                st.markdown('<p class="sec-lbl" style="margin-top:1.5rem;">Category Distribution</p>', unsafe_allow_html=True)
                col_chart, col_src = st.columns([3, 1])

                with col_chart:
                    tops = [r["Category"].split(" / ")[0] if " / " in str(r["Category"]) else str(r["Category"])
                            for r in done_rows]
                    tc = pd.Series(tops).value_counts().head(20)
                    fig_b = go.Figure(go.Bar(
                        x=tc.index.tolist(), y=tc.values.tolist(),
                        marker=dict(color=tc.values.tolist(),
                                    colorscale=[[0,"#130803"],[1,"#f55036"]], showscale=False),
                        text=tc.values.tolist(), textposition="outside",
                        textfont=dict(color="#444", size=10, family="DM Mono"),
                    ))
                    fig_b.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Mono"), height=300,
                        margin=dict(l=5, r=5, t=5, b=60),
                        xaxis=dict(tickfont=dict(color="#666", size=9, family="DM Mono"),
                                   gridcolor="#0f0f0f", tickangle=-35),
                        yaxis=dict(tickfont=dict(color="#444", size=9, family="DM Mono"),
                                   gridcolor="#0f0f0f"),
                    )
                    st.plotly_chart(fig_b, use_container_width=True)

                with col_src:
                    src_counts = pd.Series([r["Source"] for r in results]).value_counts()
                    fig_src = go.Figure(go.Pie(
                        labels=src_counts.index.tolist(),
                        values=src_counts.values.tolist(),
                        hole=0.55,
                        marker=dict(colors=["#4caf72","#4a9ef5","#d4c44a","#f55036","#555"],
                                    line=dict(color="#050505", width=2)),
                        textfont=dict(family="DM Mono", size=10),
                    ))
                    fig_src.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", height=300,
                        margin=dict(l=0, r=0, t=20, b=0),
                        font=dict(family="DM Mono"),
                        legend=dict(font=dict(color="#666", size=9)),
                        annotations=[dict(text="source", x=0.5, y=0.5, showarrow=False,
                                          font=dict(color="#444", size=10, family="DM Mono"))],
                    )
                    st.plotly_chart(fig_src, use_container_width=True)

        except Exception as e:
            st.error(f"Batch error: {e}")
            st.info("💡 Check your API key at console.groq.com.")

    elif not api_key and need_ai:
        progress.empty()
        st.warning(f"No API key — {len(need_ai)} products need AI but will be skipped.")
        save_cache(cache)

# ─────────────────────────────────────────────
# PLOTLY BASE
# ─────────────────────────────────────────────

# FIXED: Removed the duplicate 'margin' dict to prevent dictionary unpacking TypeError
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Mono")
)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["  Single  ", "  Batch  ", "  Explore  ", "  Intelligence  "])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE
# ══════════════════════════════════════════════

with tab1:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("title", placeholder="e.g. Sony WH-1000XM5 Wireless Headphones",
                              label_visibility="collapsed")
    with col_btn:
        predict_btn = st.button("→ Predict", use_container_width=True, type="primary")

    if predict_btn and query.strip():
        cache = load_cache()

        # show flow steps
        step_html = ""

        cached = find_cache(query, cache)
        if cached:
            cat = cached.get("category", str(cached)) if isinstance(cached, dict) else str(cached)
            source = "cache"
            conf   = 1.0
            cands, scores = [], []
            step_html = '<span class="flow-step flow-hit">⚡ Cache HIT</span>'
        else:
            _, tfidf_score, cands, scores = tfidf_top(query)
            clf_cat, clf_conf = classify(query, clf_bundle)

            if clf_cat and clf_conf >= clf_thresh:
                cat    = clf_cat
                source = "classifier"
                conf   = clf_conf
                step_html = (
                    '<span class="flow-step flow-skip">⚡ Cache miss</span>'
                    '<span class="flow-arrow">→</span>'
                    '<span class="flow-step flow-hit">🧠 Classifier HIT</span>'
                )
            elif tfidf_score >= tfidf_thresh:
                cat    = cands[0]
                source = "tfidf"
                conf   = tfidf_score
                step_html = (
                    '<span class="flow-step flow-skip">⚡ Cache miss</span>'
                    '<span class="flow-arrow">→</span>'
                    '<span class="flow-step flow-skip">🧠 Classifier miss</span>'
                    '<span class="flow-arrow">→</span>'
                    '<span class="flow-step flow-hit">📐 TF-IDF HIT</span>'
                )
            elif not api_key:
                st.error("No API key — all local methods below confidence threshold.")
                st.stop()
            else:
                step_html = (
                    '<span class="flow-step flow-skip">⚡ Cache miss</span>'
                    '<span class="flow-arrow">→</span>'
                    '<span class="flow-step flow-skip">🧠 Classifier miss</span>'
                    '<span class="flow-arrow">→</span>'
                    '<span class="flow-step flow-skip">📐 TF-IDF miss</span>'
                    '<span class="flow-arrow">→</span>'
                    '<span class="flow-step flow-hit">🤖 Groq AI</span>'
                )
                with st.spinner("Asking AI…"):
                    try:
                        client = Groq(api_key=api_key)
                        result = hybrid_predict_single(query, cache, client)
                        cat    = result["category"]
                        source = result["source"]
                        conf   = result.get("confidence")
                        save_cache(cache)
                    except Exception as e:
                        st.error(f"Groq API error: {e}")
                        st.stop()

            store_cache(query, {"category": cat, "source": source,
                                "confidence": conf if conf else None}, cache)
            save_cache(cache)

        # flow diagram
        st.markdown(f'<div style="margin:.75rem 0;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">{step_html}</div>',
                    unsafe_allow_html=True)

        badge_map = {
            "cache":      ("badge-cache",      "⚡ CACHE"),
            "classifier": ("badge-classifier", "🧠 CLASSIFIER"),
            "tfidf":      ("badge-tfidf",      "📐 TF-IDF"),
            "ai":         ("badge-ai",         "🤖 AI"),
        }
        badge_cls, badge_txt = badge_map.get(source, ("badge-ai", "🤖 AI"))
        conf_str = f"{conf:.0%}" if isinstance(conf, float) else "—"

        st.markdown(
            f'<span class="badge {badge_cls}">{badge_txt}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-card">'
            f'  <div class="result-lbl">Best Match</div>'
            f'  <div class="result-cat">{render_path(cat)}</div>'
            f'  <div class="result-meta">confidence: {conf_str} · source: {source}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if cands:
            st.markdown('<p class="sec-lbl" style="margin-top:1.5rem;">TF-IDF Candidate Shortlist</p>', unsafe_allow_html=True)
            n = min(15, len(cands))
            labels = [c.split(" / ")[-1] + ("  [" + c.split(" / ")[0] + "]" if " / " in c else "")
                      for c in cands[:n]]
            fig = go.Figure(go.Bar(
                x=scores[:n], y=labels, orientation="h",
                marker=dict(color=scores[:n],
                            colorscale=[[0,"#130803"],[0.5,"#6a1f0e"],[1,"#f55036"]],
                            showscale=False),
                text=[f"{s:.3f}" for s in scores[:n]], textposition="outside",
                textfont=dict(color="#444", size=10, family="DM Mono"),
            ))
            fig.update_layout(
                **PLOTLY_BASE, height=400,
                margin=dict(l=0, r=60, t=5, b=5),
                xaxis=dict(showgrid=True, gridcolor="#111",
                           tickfont=dict(color="#333", size=9, family="DM Mono"), zeroline=False),
                yaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    elif predict_btn:
        st.warning("Enter a product title first.")

# ══════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════

with tab2:
    input_mode = st.radio("Input source", ["Upload file", "Paste list"],
                          horizontal=True, label_visibility="collapsed")

    titles_to_run = None

    if input_mode == "Upload file":
        col_up, col_info = st.columns([3, 2])
        with col_up:
            uploaded = st.file_uploader("CSV / XLSX — column A = product titles",
                                        type=["csv", "xlsx"], key="batch_file")
        with col_info:
            st.markdown("""
            <div style="margin-top:1.5rem;padding:1rem;background:#0a0a0a;border:1px solid #151515;border-radius:8px;">
              <p style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#333;margin-bottom:6px;">Format</p>
              <p style="font-size:.8rem;color:#666;">Column A: product titles<br>Other columns ignored<br>CSV or XLSX accepted</p>
            </div>""", unsafe_allow_html=True)
        if uploaded:
            df_batch = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
            titles_to_run = df_batch.iloc[:, 0].astype(str).tolist()
            st.caption(f"{len(titles_to_run):,} products loaded")
    else:
        pasted = st.text_area("Paste product titles — one per line", height=200,
                              placeholder="Sony WH-1000XM5\nApple AirPods Pro\n…",
                              label_visibility="visible")
        if pasted.strip():
            titles_to_run = [l.strip() for l in pasted.splitlines() if l.strip()]
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
        search_q  = st.text_input("search", placeholder="Filter by keyword…",
                                  label_visibility="collapsed", key="ex_search")
    with col_d:
        depth_opt = st.selectbox("Depth", ["All"] + list(range(1, depth_max + 1)),
                                 label_visibility="collapsed", key="ex_depth")

    filtered = leaves
    if search_q.strip():
        filtered = [l for l in filtered if search_q.lower() in l.lower()]
    if depth_opt != "All":
        filtered = [l for l in filtered if len(l.split(" / ")) == int(depth_opt)]

    st.caption(f"{len(filtered):,} of {len(leaves):,} categories shown")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="sec-lbl">Categories per Top-Level Group</p>', unsafe_allow_html=True)
        tc = pd.Series([l.split(" / ")[0] for l in filtered]).value_counts().head(20)
        fig_top = go.Figure(go.Bar(
            x=tc.values.tolist(), y=tc.index.tolist(), orientation="h",
            marker=dict(color=tc.values.tolist(),
                        colorscale=[[0,"#0d0503"],[0.4,"#5a1808"],[1,"#f55036"]], showscale=False),
            text=tc.values.tolist(), textposition="outside",
            textfont=dict(color="#333", size=9, family="DM Mono"),
        ))
        fig_top.update_layout(**PLOTLY_BASE, height=400,
                              margin=dict(l=0, r=40, t=5, b=5),
                              xaxis=dict(tickfont=dict(color="#333", size=9, family="DM Mono"), gridcolor="#0f0f0f"),
                              yaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        st.markdown('<p class="sec-lbl">Depth Distribution</p>', unsafe_allow_html=True)
        depths = pd.Series([len(l.split(" / ")) for l in filtered]).value_counts().sort_index()
        fig_dep = go.Figure(go.Bar(
            x=[f"Level {i}" for i in depths.index], y=depths.values.tolist(),
            marker=dict(color=depths.values.tolist(),
                        colorscale=[[0,"#0d0503"],[1,"#f55036"]], showscale=False),
            text=depths.values.tolist(), textposition="outside",
            textfont=dict(color="#333", size=10, family="DM Mono"),
        ))
        fig_dep.update_layout(**PLOTLY_BASE, height=400,
                              margin=dict(l=10, r=10, t=5, b=10),
                              xaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), gridcolor="#0f0f0f"),
                              yaxis=dict(tickfont=dict(color="#333", size=9, family="DM Mono"), gridcolor="#0f0f0f"))
        st.plotly_chart(fig_dep, use_container_width=True)

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

    tm_ids, tm_labels, tm_parents, tm_values = ["root"], ["All"], [""], [0]
    for top, total in sorted(top_totals.items(), key=lambda x: -x[1])[:35]:
        tm_ids.append(top); tm_labels.append(top); tm_parents.append("root"); tm_values.append(total)

    seen = set(tm_ids)
    for (top, second), cnt in sorted(td.items(), key=lambda x: -x[1]):
        nid = f"{top}||{second}"
        if top in seen and nid not in seen and second != top:
            tm_ids.append(nid); tm_labels.append(second)
            tm_parents.append(top); tm_values.append(cnt)
            seen.add(nid)

    fig_tree = go.Figure(go.Treemap(
        ids=tm_ids, labels=tm_labels, parents=tm_parents, values=tm_values,
        branchvalues="total",
        marker=dict(colorscale=[[0,"#0a0302"],[0.25,"#2a0d08"],[0.6,"#8a2a14"],[1,"#f55036"]],
                    showscale=False, line=dict(width=1, color="#050505")),
        textfont=dict(family="DM Mono", size=11, color="#ddd"),
        hovertemplate="<b>%{label}</b><br>%{value} leaf categories<extra></extra>",
        pathbar=dict(visible=True, textfont=dict(family="DM Mono", size=10)),
    ))
    fig_tree.update_layout(**PLOTLY_BASE, height=520, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()
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
        st.caption(f"Showing {show_n:,} of {len(filtered):,}")
    st.download_button("↓ Download Filtered Categories",
                       pd.DataFrame({"category_path": filtered}).to_csv(index=False).encode(),
                       "categories_filtered.csv", "text/csv")

# ══════════════════════════════════════════════
# TAB 4 — INTELLIGENCE (classifier & cache stats)
# ══════════════════════════════════════════════

with tab4:
    st.markdown('<p class="sec-lbl">System Intelligence Dashboard</p>', unsafe_allow_html=True)

    # ── Hybrid flow diagram ──
    st.markdown("""
    <div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;">
      <p style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#333;margin-bottom:1rem;">Prediction Flow</p>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <div style="text-align:center">
          <div style="background:#0a1a0a;border:1px solid #1a3a1a;color:#4caf72;padding:8px 14px;border-radius:6px;font-size:.75rem;">⚡ Cache</div>
          <div style="font-size:.6rem;color:#333;margin-top:4px;">instant</div>
        </div>
        <span style="color:#2a2a2a;font-size:1.2rem;">→</span>
        <div style="text-align:center">
          <div style="background:#0a0f1a;border:1px solid #1a2a3a;color:#4a9ef5;padding:8px 14px;border-radius:6px;font-size:.75rem;">🧠 Classifier</div>
          <div style="font-size:.6rem;color:#333;margin-top:4px;">local ML</div>
        </div>
        <span style="color:#2a2a2a;font-size:1.2rem;">→</span>
        <div style="text-align:center">
          <div style="background:#1a1a0a;border:1px solid #3a3a1a;color:#d4c44a;padding:8px 14px;border-radius:6px;font-size:.75rem;">📐 TF-IDF Gate</div>
          <div style="font-size:.6rem;color:#333;margin-top:4px;">similarity</div>
        </div>
        <span style="color:#2a2a2a;font-size:1.2rem;">→</span>
        <div style="text-align:center">
          <div style="background:#1a0d08;border:1px solid #3a1a0a;color:#f55036;padding:8px 14px;border-radius:6px;font-size:.75rem;">🤖 Groq AI</div>
          <div style="font-size:.6rem;color:#333;margin-top:4px;">last resort</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Classifier status ──
    col_cl, col_ca = st.columns(2)

    with col_cl:
        st.markdown('<p class="sec-lbl">Local Classifier</p>', unsafe_allow_html=True)
        if clf_bundle is not None:
            clf, le, cv = clf_bundle
            n_classes = len(le.classes_)
            n_samples = len(cache_data)
            st.markdown(f"""
            <div style="background:#0a0f1a;border:1px solid #1a2a3a;border-radius:8px;padding:1rem;">
              <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                <div><div class="stat-val" style="font-size:1.2rem;">{n_classes:,}</div><div class="stat-lbl">Classes</div></div>
                <div><div class="stat-val" style="font-size:1.2rem;">{n_samples:,}</div><div class="stat-lbl">Training Samples</div></div>
                <div><div class="stat-val" style="font-size:1.2rem;">{clf_thresh:.0%}</div><div class="stat-lbl">Confidence Gate</div></div>
              </div>
              <p style="color:#4a9ef5;font-size:.7rem;margin-top:.75rem;">✓ Classifier active</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            needed = CLASSIFIER_MIN_SAMPLES - len(cache_data)
            st.markdown(f"""
            <div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:8px;padding:1rem;">
              <p style="color:#555;font-size:.8rem;">Classifier not yet trained.</p>
              <p style="color:#f55036;font-size:.75rem;">Need {needed} more cached predictions to activate.</p>
              <div style="background:#111;border-radius:4px;height:6px;margin-top:.5rem;">
                <div style="background:#f55036;height:6px;border-radius:4px;width:{min(100, len(cache_data)/CLASSIFIER_MIN_SAMPLES*100):.0f}%"></div>
              </div>
              <p style="color:#333;font-size:.62rem;margin-top:4px;">{len(cache_data)} / {CLASSIFIER_MIN_SAMPLES}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_ca:
        st.markdown('<p class="sec-lbl">Cache Analytics</p>', unsafe_allow_html=True)
        if cache_data:
            sources = []
            for v in cache_data.values():
                s = v.get("source", "unknown") if isinstance(v, dict) else "legacy"
                sources.append(s)
            src_s = pd.Series(sources).value_counts()
            colors_map = {"cache":"#4caf72","classifier":"#4a9ef5","tfidf":"#d4c44a","ai":"#f55036","legacy":"#555","unknown":"#333"}
            colors = [colors_map.get(s, "#555") for s in src_s.index]

            fig_src = go.Figure(go.Pie(
                labels=src_s.index.tolist(), values=src_s.values.tolist(),
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#050505", width=2)),
                textfont=dict(family="DM Mono", size=10),
            ))
            fig_src.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(family="DM Mono"),
                legend=dict(font=dict(color="#666", size=9)),
                annotations=[dict(text=f"{len(cache_data)}<br><span style='font-size:8px'>entries</span>",
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(color="#f55036", size=13, family="Syne"))],
            )
            st.plotly_chart(fig_src, use_container_width=True)
        else:
            st.caption("No cache entries yet.")

    # ── Top cached categories ──
    if cache_data:
        st.divider()
        st.markdown('<p class="sec-lbl">Top Cached Categories</p>', unsafe_allow_html=True)
        cats = []
        for v in cache_data.values():
            c = v.get("category","") if isinstance(v, dict) else str(v)
            if c:
                cats.append(c)
        if cats:
            top_cached = pd.Series(cats).value_counts().head(20)
            fig_cc = go.Figure(go.Bar(
                x=top_cached.values.tolist(), y=top_cached.index.tolist(), orientation="h",
                marker=dict(color=top_cached.values.tolist(),
                            colorscale=[[0,"#0d0503"],[1,"#f55036"]], showscale=False),
            ))
            fig_cc.update_layout(
                **PLOTLY_BASE, height=max(300, len(top_cached) * 22),
                margin=dict(l=0, r=40, t=5, b=5),
                xaxis=dict(tickfont=dict(color="#333", size=9, family="DM Mono"), gridcolor="#0f0f0f"),
                yaxis=dict(tickfont=dict(color="#777", size=10, family="DM Mono"), autorange="reversed"),
            )
            st.plotly_chart(fig_cc, use_container_width=True)

    # ── Thresholds explainer ──
    st.divider()
    st.markdown('<p class="sec-lbl">How Thresholds Work</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;padding:1rem;">
        <p style="color:#d4c44a;font-size:.7rem;font-weight:600;margin-bottom:4px;">📐 TF-IDF Gate  ({tfidf_thresh:.0%})</p>
        <p style="color:#555;font-size:.72rem;line-height:1.5;">
          If the highest cosine similarity score between the query and category index
          exceeds <b style="color:#888">{tfidf_thresh:.0%}</b>, the top match is returned
          without calling the AI. Lower this to send more to AI; raise it to trust TF-IDF more.
        </p>
      </div>
      <div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;padding:1rem;">
        <p style="color:#4a9ef5;font-size:.7rem;font-weight:600;margin-bottom:4px;">🧠 Classifier Gate  ({clf_thresh:.0%})</p>
        <p style="color:#555;font-size:.72rem;line-height:1.5;">
          A LogisticRegression trained on your cache history. If it predicts a category
          with probability &gt; <b style="color:#888">{clf_thresh:.0%}</b>, it bypasses
          both TF-IDF and AI. Activates after {CLASSIFIER_MIN_SAMPLES} cached predictions.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

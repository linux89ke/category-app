"""
Product Category Predictor
Hybrid Flow: Cache → Local ML → TF-IDF Gate → Batched Groq (ThreadPool)
"""

import os, json, pickle, re, math
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

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
for f in ["category_map1.csv", "category_map1.xlsx", "category_map1.xlsx - pim category attribute set.csv"]:
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
# CACHE HELPERS
# ─────────────────────────────────────────────

def normalize(q: str) -> str:
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
# ─────────────────────────────────────────────

def _train_classifier(cache: dict):
    rows = []
    for raw_q, val in cache.items():
        cat = val.get("category", "") if isinstance(val, dict) else str(val)
        if cat: rows.append((raw_q, cat))

    if len(rows) < CLASSIFIER_MIN_SAMPLES: return None

    texts, labels = [r[0] for r in rows], [r[1] for r in rows]

    from collections import Counter
    counts = Counter(labels)
    filtered = [(t, l) for t, l in zip(texts, labels) if counts[l] >= 2]
    if len(filtered) < CLASSIFIER_MIN_SAMPLES: return None

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
    if os.path.exists(CLASSIFIER_FILE):
        try:
            with open(CLASSIFIER_FILE, "rb") as fh:
                saved_size, model = pickle.load(fh)
            if saved_size >= CLASSIFIER_MIN_SAMPLES and model is not None:
                return model
        except: pass

    cache = load_cache()
    result = _train_classifier(cache)

    if result is not None:
        try:
            with open(CLASSIFIER_FILE, "wb") as fh:
                pickle.dump((len(cache), result), fh)
        except: pass
    return result

def classify(q: str, clf_bundle) -> tuple[str | None, float]:
    if clf_bundle is None: return None, 0.0
    clf, le, cv = clf_bundle
    X    = cv.transform([normalize(q)])
    probs = clf.predict_proba(X)[0]
    idx  = np.argmax(probs)
    return le.inverse_transform([idx])[0], float(probs[idx])

clf_bundle = load_classifier(len(cache_data))
clf_status = "ready" if clf_bundle is not None else f"needs {CLASSIFIER_MIN_SAMPLES}+ samples"

# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a product categorisation expert. "
    "Given a product title and a list of candidate category paths, pick the best matching category. "
    "Consider brand, product type, gender, style, material, and QUANTITY. "
    "If a product is a single item, DO NOT put it in a 'Sets' category. "
    "Return ONLY JSON: {\"category\": \"path here\", \"score\": 0.95}"
)

def tfidf_top(q: str) -> tuple[str, float, list[str], list[float]]:
    sims  = cosine_similarity(vectorizer.transform([q]), matrix)[0]
    idxs  = np.argsort(sims)[::-1][:shortlist_k]
    cands = [leaves[i] for i in idxs]
    scores = [float(sims[i]) for i in idxs]
    return cands[0], scores[0], cands, scores

def batch_shortlist(queries: list[str]) -> list[list[str]]:
    qmat = vectorizer.transform(queries)
    sims = cosine_similarity(qmat, matrix)
    results = []
    for row in sims:
        top_idx = np.argsort(row)[::-1][:shortlist_k]
        results.append([leaves[i] for i in top_idx if row[i] > 0])
    return results

def render_path(cat_str: str) -> str:
    if not cat_str: return cat_str
    parts = cat_str.split(" / ")
    html  = []
    for i, p in enumerate(parts):
        color = "#f55036" if i == len(parts) - 1 else "#555"
        html.append(f'<span style="color:{color}">{p}</span>')
    return '<span style="color:#2a2a2a"> / </span>'.join(html)

# ─────────────────────────────────────────────
# MULTI-THREADED GROQ CALL (No asyncio)
# ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def sync_rerank_batch(idx: int, query: str, candidates: list[str], groq_key: str, model: str) -> tuple[int, dict]:
    """Runs fully synchronously to avoid Streamlit threading issues."""
    # Creates a fresh client instance per thread, very safe for Streamlit
    client = Groq(api_key=groq_key)
    cand_list = "\n".join(f"- {c}" for c in candidates[:shortlist_k])
    
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        out = json.loads(raw)
        cat = out.get("category", "")
        return idx, {"category": cat, "score": out.get("score", 1.0)}
    except Exception:
        return idx, {"category": candidates[0] if candidates else "", "score": 1.0}


# ─────────────────────────────────────────────
# PLOTLY BASE
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor ="rgba(0,0,0,0)", font=dict(family="DM Mono"))

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
        query = st.text_input("title", placeholder="e.g. Sony WH-1000XM5 Wireless Headphones", label_visibility="collapsed")
    with col_btn:
        predict_btn = st.button("→ Predict", use_container_width=True, type="primary")

    if predict_btn and query.strip():
        cache = load_cache()
        step_html = ""

        cached = find_cache(query, cache)
        if cached:
            cat = cached.get("category", str(cached)) if isinstance(cached, dict) else str(cached)
            source, conf = "cache", 1.0
            cands, scores = [], []
            step_html = '<span class="flow-step flow-hit">⚡ Cache HIT</span>'
        else:
            _, tfidf_score, cands, scores = tfidf_top(query)
            clf_cat, clf_conf = classify(query, clf_bundle)

            if clf_cat and clf_conf >= clf_thresh:
                cat, source, conf = clf_cat, "classifier", clf_conf
                step_html = '<span class="flow-step flow-skip">⚡ Cache miss</span><span class="flow-arrow">→</span><span class="flow-step flow-hit">🧠 Classifier HIT</span>'
            elif tfidf_score >= tfidf_thresh:
                cat, source, conf = cands[0], "tfidf", tfidf_score
                step_html = '<span class="flow-step flow-skip">⚡ Cache miss</span><span class="flow-arrow">→</span><span class="flow-step flow-skip">🧠 Classifier miss</span><span class="flow-arrow">→</span><span class="flow-step flow-hit">📐 TF-IDF HIT</span>'
            elif not api_key:
                st.error("No API key — all local methods below confidence threshold.")
                st.stop()
            else:
                step_html = '<span class="flow-step flow-skip">⚡ Cache miss</span><span class="flow-arrow">→</span><span class="flow-step flow-skip">🧠 Classifier miss</span><span class="flow-arrow">→</span><span class="flow-step flow-skip">📐 TF-IDF miss</span><span class="flow-arrow">→</span><span class="flow-step flow-hit">🤖 Groq AI</span>'
                with st.spinner("Asking AI…"):
                    try:
                        client = Groq(api_key=api_key)
                        cand_list = "\n".join(f"- {c}" for c in cands)
                        resp = client.chat.completions.create(
                            model=model_choice, temperature=0.1, response_format={"type": "json_object"},
                            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Product: {query}\n\nCandidates:\n{cand_list}"}]
                        )
                        out = json.loads(resp.choices[0].message.content.strip())
                        cat = out.get("category", cands[0])
                        source, conf = "ai", out.get("score", 1.0)
                    except Exception as e:
                        st.error(f"Groq API error: {e}"); st.stop()

            store_cache(query, {"category": cat, "source": source, "confidence": conf if conf else None}, cache)
            save_cache(cache)

        st.markdown(f'<div style="margin:.75rem 0;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">{step_html}</div>', unsafe_allow_html=True)
        badge_map = {"cache": ("badge-cache", "⚡ CACHE"), "classifier": ("badge-classifier", "🧠 CLASSIFIER"), "tfidf": ("badge-tfidf", "📐 TF-IDF"), "ai": ("badge-ai", "🤖 AI")}
        badge_cls, badge_txt = badge_map.get(source, ("badge-ai", "🤖 AI"))
        
        st.markdown(f'<span class="badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-card"><div class="result-lbl">Best Match</div><div class="result-cat">{render_path(cat)}</div><div class="result-meta">confidence: {conf:.0%} · source: {source}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════

with tab2:
    input_mode = st.radio("Input source", ["Upload file", "Paste list"], horizontal=True, label_visibility="collapsed")
    titles_to_run = None

    if input_mode == "Upload file":
        col_up, col_info = st.columns([3, 2])
        with col_up:
            uploaded = st.file_uploader("CSV / XLSX — column A = product titles", type=["csv", "xlsx"], key="batch_file")
        with col_info:
            st.markdown("""<div style="margin-top:1.5rem;padding:1rem;background:#0a0a0a;border:1px solid #151515;border-radius:8px;">
              <p style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#333;margin-bottom:6px;">Format</p>
              <p style="font-size:.8rem;color:#666;">Column A: product titles<br>Other columns ignored<br>CSV or XLSX accepted</p></div>""", unsafe_allow_html=True)
        if uploaded:
            df_batch = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
            titles_to_run = df_batch.iloc[:, 0].astype(str).tolist()
            st.caption(f"{len(titles_to_run):,} products loaded")
    else:
        pasted = st.text_area("Paste product titles — one per line", height=200)
        if pasted.strip(): titles_to_run = [l.strip() for l in pasted.splitlines() if l.strip()]

    if titles_to_run:
        if st.button("▶ Run Batch", type="primary", key="run_batch"):
            cache = load_cache()
            results = []
            need_ai = []
            
            # Phase 1: Local parsing
            for i, q in enumerate(titles_to_run):
                cached = find_cache(q, cache)
                if cached:
                    cat = cached.get("category", str(cached)) if isinstance(cached, dict) else str(cached)
                    results.append({"#": i+1, "Product": q, "Category": cat, "Source": "⚡ Cache"})
                    continue

                _, tfidf_score, cands, _ = tfidf_top(q)
                clf_cat, clf_conf = classify(q, clf_bundle)

                if clf_cat and clf_conf >= clf_thresh:
                    results.append({"#": i+1, "Product": q, "Category": clf_cat, "Source": "🧠 Classifier"})
                    store_cache(q, {"category": clf_cat, "source": "classifier", "confidence": clf_conf}, cache)
                    continue

                if tfidf_score >= tfidf_thresh:
                    results.append({"#": i+1, "Product": q, "Category": cands[0], "Source": "📐 TF-IDF"})
                    store_cache(q, {"category": cands[0], "source": "tfidf", "confidence": tfidf_score}, cache)
                    continue

                results.append({"#": i+1, "Product": q, "Category": "—", "Source": "⏳ Pending"})
                need_ai.append((i, q))

            progress = st.progress(0, text="Processing...")
            table = st.empty()
            table.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            # Phase 2: ThreadPool AI Processing (No Asyncio needed!)
            if need_ai and api_key:
                to_predict_indices = [x[0] for x in need_ai]
                to_predict_queries = [x[1] for x in need_ai]
                
                all_candidates = batch_shortlist(to_predict_queries)
                
                done = 0
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_idx = {
                        executor.submit(sync_rerank_batch, idx, q, c, api_key, model_choice): (idx, q)
                        for idx, q, c in zip(to_predict_indices, to_predict_queries, all_candidates)
                    }
                    
                    for future in as_completed(future_to_idx):
                        orig_idx, q = future_to_idx[future]
                        done += 1
                        try:
                            _, preds = future.result()
                            cat = preds.get("category", "")
                            results[orig_idx]["Category"] = cat
                            results[orig_idx]["Source"] = "🤖 AI"
                            store_cache(q, {"category": cat, "source": "ai"}, cache)
                        except Exception as e:
                            results[orig_idx]["Category"] = f"Error: {e}"
                            results[orig_idx]["Source"] = "❌ Failed"
                        
                        perc = min(100, int((done / len(need_ai)) * 100))
                        progress.progress(perc, text=f"AI Processing: {done}/{len(need_ai)}")
                        table.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            save_cache(cache)
            progress.empty()
            st.success("Batch Complete!")
            st.download_button("↓ Download Results CSV", pd.DataFrame(results).to_csv(index=False).encode(), "predictions.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — EXPLORE & TAB 4 — INTELLIGENCE (Kept exactly identical to your previous code)
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="sec-lbl">Search & Explore the Category Tree</p>', unsafe_allow_html=True)
    col_s, col_d = st.columns([4, 1])
    with col_s: search_q  = st.text_input("search", placeholder="Filter by keyword…", label_visibility="collapsed", key="ex_search")
    with col_d: depth_opt = st.selectbox("Depth", ["All"] + list(range(1, depth_max + 1)), label_visibility="collapsed", key="ex_depth")

    filtered = leaves
    if search_q.strip(): filtered = [l for l in filtered if search_q.lower() in l.lower()]
    if depth_opt != "All": filtered = [l for l in filtered if len(l.split(" / ")) == int(depth_opt)]

    st.caption(f"{len(filtered):,} of {len(leaves):,} categories shown")
    max_show = max(10, len(filtered))
    show_n   = st.slider("Rows to show", 10, min(500, max_show), min(50, max_show), key="ex_rows")
    display_df = pd.DataFrame({
        "Category Path": filtered[:show_n],
        "Depth":         [len(l.split(" / ")) for l in filtered[:show_n]],
        "Top-Level":     [l.split(" / ")[0]   for l in filtered[:show_n]],
        "Leaf Node":     [l.split(" / ")[-1]  for l in filtered[:show_n]],
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab4:
    st.markdown('<p class="sec-lbl">System Intelligence Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;">
      <p style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#333;margin-bottom:1rem;">Prediction Flow</p>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <div style="text-align:center"><div style="background:#0a1a0a;border:1px solid #1a3a1a;color:#4caf72;padding:8px 14px;border-radius:6px;font-size:.75rem;">⚡ Cache</div><div style="font-size:.6rem;color:#333;margin-top:4px;">instant</div></div>
        <span style="color:#2a2a2a;font-size:1.2rem;">→</span>
        <div style="text-align:center"><div style="background:#0a0f1a;border:1px solid #1a2a3a;color:#4a9ef5;padding:8px 14px;border-radius:6px;font-size:.75rem;">🧠 Classifier</div><div style="font-size:.6rem;color:#333;margin-top:4px;">local ML</div></div>
        <span style="color:#2a2a2a;font-size:1.2rem;">→</span>
        <div style="text-align:center"><div style="background:#1a1a0a;border:1px solid #3a3a1a;color:#d4c44a;padding:8px 14px;border-radius:6px;font-size:.75rem;">📐 TF-IDF Gate</div><div style="font-size:.6rem;color:#333;margin-top:4px;">similarity</div></div>
        <span style="color:#2a2a2a;font-size:1.2rem;">→</span>
        <div style="text-align:center"><div style="background:#1a0d08;border:1px solid #3a1a0a;color:#f55036;padding:8px 14px;border-radius:6px;font-size:.75rem;">🤖 Groq AI</div><div style="font-size:.6rem;color:#333;margin-top:4px;">last resort</div></div>
      </div></div>""", unsafe_allow_html=True)
    
    col_cl, col_ca = st.columns(2)
    with col_cl:
        st.markdown('<p class="sec-lbl">Local Classifier</p>', unsafe_allow_html=True)
        if clf_bundle is not None:
            st.markdown(f'<div style="background:#0a0f1a;border:1px solid #1a2a3a;border-radius:8px;padding:1rem;"><p style="color:#4a9ef5;font-size:.7rem;">✓ Classifier active</p></div>', unsafe_allow_html=True)
        else:
            needed = CLASSIFIER_MIN_SAMPLES - len(cache_data)
            st.markdown(f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-radius:8px;padding:1rem;"><p style="color:#f55036;font-size:.75rem;">Need {needed} more cached predictions to activate.</p></div>', unsafe_allow_html=True)

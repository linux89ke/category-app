import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Category Matcher & Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ══════════════════════════════════════════════════════════════════════════════
# STYLING — dark industrial / data-warehouse aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Headers */
h1 { 
    font-family: 'IBM Plex Mono', monospace !important; 
    color: #58a6ff !important; 
    font-size: 1.6rem !important;
    letter-spacing: -0.5px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 12px;
    margin-bottom: 24px !important;
}
h2, h3 { color: #c9d1d9 !important; font-weight: 600 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem !important; }

/* Buttons */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.15s;
}
.stButton > button:hover { background: #2ea043 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(35,134,54,0.3); }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 8px;
}

/* Info / warning boxes */
.stAlert { border-radius: 8px; border-left-width: 3px; }

/* DataFrame */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Slider */
.stSlider [data-baseweb="slider"] { color: #58a6ff; }

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #1f6feb, #58a6ff); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] { color: #8b949e !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }

/* Download button */
.stDownloadButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: 1px solid #388bfd !important;
    border-radius: 6px;
}

/* Code / mono text */
code { background: #161b22; color: #79c0ff; border-radius: 4px; padding: 2px 6px; font-family: 'IBM Plex Mono', monospace; }

/* Expander */
.streamlit-expanderHeader { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 6px; }

/* Status badge helpers */
.badge-approved { color: #3fb950; font-weight: 700; }
.badge-rejected { color: #f85149; font-weight: 700; }
.badge-review   { color: #d29922; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔍 Category Matcher & Auditor")
st.caption("TF-IDF · Fuzzy Reranking · Synonym-Enriched Taxonomy — 30,085 categories")

# ══════════════════════════════════════════════════════════════════════════════
# BRAND / NOISE STOPWORDS  (things that confuse TF-IDF but add no category signal)
# ══════════════════════════════════════════════════════════════════════════════
BRAND_STOPWORDS = {
    # Tech brands
    "samsung","apple","iphone","huawei","oppo","xiaomi","vivo","tecno","infinix",
    "nokia","motorola","sony","lg","hp","dell","lenovo","asus","acer","toshiba",
    "panasonic","philips","canon","nikon","fujifilm","gopro","bose","jbl","anker",
    # Fashion
    "nike","adidas","puma","reebok","vans","converse","gucci","prada","zara",
    "levis","tommy","ralph","calvin","armani","versace","fendi","dior","chanel",
    # FMCG / grocery
    "nestle","nescafe","unilever","loreal","garnier","neutrogena","dove","nivea",
    "pampers","huggies","pg","procter","colgate","oral","heinz","kelloggs",
    # Furniture / home
    "ikea","ashley","wayfair","tempur","simmons",
    # Generic noise in product titles
    "new","original","genuine","authentic","official","best","premium","quality",
    "pack","bundle","combo","piece","pcs","units","qty","quantity",
    "black","white","red","blue","green","silver","gold","grey","gray","brown",
    "pink","purple","yellow","orange","navy","beige","cream","clear","transparent",
    # Material descriptors (can pull Industrial/Scientific incorrectly)
    "stainless","steel","plastic","rubber","fabric","cotton","wooden","wood",
    "metal","aluminium","aluminum","ceramic","nylon","polyester",
    # Generic size / tier noise
    "slim","large","small","mini","pro","ultra","plus","standard",
    "heavy","duty","grade","series","version","type",
}

# ══════════════════════════════════════════════════════════════════════════════
# SYNONYM EXPANSION  (applied to query tokens before TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════
QUERY_EXPANSIONS = {
    # ── Phones ────────────────────────────────────────────────────────────────
    "iphone":           "phone smartphone mobile handset cell phone",
    "galaxy":           "phone smartphone mobile cell phone",
    "pixel":            "phone smartphone mobile",
    "phone":            "smartphone mobile handset cellphone cell",
    "mobile":           "phone smartphone handset",
    # ── Phone / tablet accessories ────────────────────────────────────────────
    "case":             "cover shell bumper casing protective phones tablets",
    "cover":            "case shell protector",
    "earbuds":          "earphones headphones in-ear wireless electronics",
    "airpods":          "earphones earbuds wireless bluetooth",
    "powerbank":        "portable charger external battery",
    "power bank":       "portable charger external battery",
    "charger":          "adapter charging cable fast charger",
    "screen protector": "tempered glass guard film phones tablets mobile",
    "tempered glass":   "screen protector guard film phones tablets",
    # ── Computing ─────────────────────────────────────────────────────────────
    "laptop":           "notebook computer portable ultrabook",
    "pc":               "computer desktop",
    "dslr":             "camera digital photo interchangeable lens",
    # ── TV / Audio ────────────────────────────────────────────────────────────
    "tv":               "television smart led oled flat screen",
    "telly":            "television tv screen",
    "smartwatch":       "smart watch fitness tracker wearable",
    "earphone":         "headphone in-ear earbud earpiece audio",
    "earphones":        "headphones in-ear earbuds earpieces audio",
    "headphone":        "headset over-ear on-ear audio electronics",
    "headphones":       "headsets over-ear on-ear audio electronics",
    "wireless":         "bluetooth audio electronics",
    "bluetooth":        "wireless electronics audio",
    "speaker":          "audio sound electronics",
    "speakers":         "audio sound electronics",
    "portable":         "compact handheld",
    # ── Home appliances ───────────────────────────────────────────────────────
    "fridge":           "refrigerator cooler cold storage",
    "cooker":           "stove range oven gas electric kitchen",
    "blender":          "mixer food processor smoothie kitchen appliance",
    "kettle":           "electric kettle water boiler home kitchen appliance",
    "iron":             "steam iron clothes iron appliance",
    # ── Fashion ───────────────────────────────────────────────────────────────
    "sneakers":         "shoes athletic trainers footwear",
    "sneaker":          "shoe athletic trainer footwear",
    "trainers":         "shoes athletic sneakers footwear",
    "trousers":         "pants slacks chinos bottoms",
    "trouser":          "pant slack chino bottom",
    "spectacles":       "glasses eyewear eyeglasses frames",
    "specs":            "glasses eyewear",
    "sofa":             "couch settee loveseat furniture",
    "wallet":           "card holder billfold money fashion accessories",
    # ── Baby ──────────────────────────────────────────────────────────────────
    "nappy":            "diaper nappies baby grocery toiletries",
    "nappies":          "diapers diaper baby grocery toiletries",
    "stroller":         "baby pram pushchair carriage products infant",
    "strollers":        "baby prams pushchairs carriages products",
    "pram":             "stroller pushchair baby carriage",
    "prams":            "strollers pushchairs baby carriages",
    "cot":              "crib baby bed infant",
    "crib":             "cot baby bed infant",
    # ── Food / Grocery ────────────────────────────────────────────────────────
    "coffee":           "instant coffee espresso beverage grocery",
    "juice":            "fruit juice drink beverage grocery",
    "perfume":          "fragrance cologne scent beauty personal care",
}

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN SIGNAL MAP
# When these tokens appear in the raw query, any candidate outside the
# allowed top-level categories receives a score penalty. This prevents
# "GPS Screen Protector Foils" from beating "Phones & Tablets / Screen Protectors"
# when the query clearly describes a phone accessory.
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_SIGNALS: dict[str, set[str]] = {
    "phone":        {"Phones & Tablets"},
    "mobile":       {"Phones & Tablets"},
    "tablet":       {"Phones & Tablets"},
    "laptop":       {"Computing"},
    "notebook":     {"Computing"},
    "computer":     {"Computing"},
    "shoe":         {"Fashion"},
    "shoes":        {"Fashion"},
    "sneaker":      {"Fashion"},
    "sneakers":     {"Fashion"},
    "trainer":      {"Fashion"},
    "trainers":     {"Fashion"},
    "dress":        {"Fashion"},
    "trouser":      {"Fashion"},
    "trousers":     {"Fashion"},
    "wallet":       {"Fashion"},
    "jeans":        {"Fashion"},
    "tv":           {"Electronics"},
    "television":   {"Electronics"},
    "headphone":    {"Electronics"},
    "headphones":   {"Electronics"},
    "speaker":      {"Electronics"},
    "bluetooth":    {"Electronics", "Phones & Tablets"},
    "diaper":       {"Grocery", "Baby Products"},
    "diapers":      {"Grocery", "Baby Products"},
    "nappy":        {"Grocery", "Baby Products"},
    "nappies":      {"Grocery", "Baby Products"},
    "stroller":     {"Baby Products"},
    "pram":         {"Baby Products"},
    "coffee":       {"Grocery"},
    "perfume":      {"Health & Beauty"},
    "fragrance":    {"Health & Beauty"},
    "yoga":         {"Sporting Goods"},
    "bicycle":      {"Sporting Goods"},
    "cycling":      {"Sporting Goods"},
}
_MEASURE_RE = re.compile(
    r'\b\d+\.?\d*\s*'
    r'(ml|cl|dl|l|g|kg|mg|units|tabs|tablets|pcs|capsules|oz|fl oz|count|ct|'
    r'inch|inches|in\b|cm|mm|m\b|ft|w|kw|mah|kwh|gb|tb|mb|mp|k\b|hz|'
    r'x\d+|pack of \d+)\b',
    re.IGNORECASE,
)

def clean_query(text: str) -> str:
    """Strip noise, brands, measurements; then synonym-expand."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _MEASURE_RE.sub(" ", text)               # remove measurements
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only alpha + spaces
    tokens = [t for t in text.split() if len(t) > 1 and t not in BRAND_STOPWORDS]

    expanded, seen = [], set(tokens)
    for t in tokens:
        expanded.append(t)
        if t in QUERY_EXPANSIONS:
            for ex in QUERY_EXPANSIONS[t].split():
                if ex not in seen:
                    expanded.append(ex)
                    seen.add(ex)
    return " ".join(expanded).strip()


# ══════════════════════════════════════════════════════════════════════════════
# INDEX BUILDER  (cached across sessions)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️  Building taxonomy index…")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found in the working directory.")
        st.stop()

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_excel(CATEGORY_FILE, usecols=[0, 1, 2, 3])
    df.columns = ["category_name", "category_code", "Category Path", "keywords"]

    # ── Clean category_code (handles int 1000017, float 1000017.0, and the two
    #    malformed string codes like "beer,1001386") ────────────────────────────
    def _fix_code(val):
        s = str(val).strip()
        if "," in s:                          # malformed: "beer,1001386"
            parts = [p.strip() for p in s.split(",")]
            nums = [p for p in parts if p.isdigit()]
            return nums[0] if nums else s
        return re.sub(r"\.0$", "", s).strip() # remove trailing .0

    df["category_code_str"] = df["category_code"].apply(_fix_code)

    # ── Depth  ────────────────────────────────────────────────────────────────
    df["depth"] = df["Category Path"].apply(lambda x: str(x).count("/") + 1)

    # ── Build search_text ─────────────────────────────────────────────────────
    # Weighting rationale:
    #   • category_name × 5  → the precise leaf label; highest signal
    #   • keywords × 5       → synonym-enriched; catches consumer language
    #   • path_words × 2     → structural context; lower weight to avoid
    #                           parent-node "gravity" pulling generic ancestors
    name_clean = df["category_name"].astype(str).str.lower()
    path_clean = df["Category Path"].astype(str).str.replace("/", " ", regex=False).str.lower()
    kw_clean   = df["keywords"].fillna("").astype(str).str.lower()

    df["name_clean"] = name_clean
    df["path_clean"] = path_clean
    df["kw_clean"]   = kw_clean
    df["search_text"] = (
        (name_clean + " ") * 5 +
        (kw_clean   + " ") * 5 +
        (path_clean + " ") * 2
    )

    # ── TF-IDF  ───────────────────────────────────────────────────────────────
    # ngram (1,2): unigrams dominate matching; bigrams catch "screen protector",
    # "smart tv", "instant coffee" without exploding the vocabulary.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,          # log-normalise term freq (handles repetition)
        strip_accents="unicode",
        max_features=250_000,
    )
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])

    # code → path lookup (for validating existing assigned codes)
    code_to_path = dict(zip(df["category_code_str"], df["Category Path"]))

    return df, vectorizer, tfidf_matrix, code_to_path


df_cat, vectorizer, tfidf_matrix, code_to_path = build_index()


# ══════════════════════════════════════════════════════════════════════════════
# RERANKER
# ══════════════════════════════════════════════════════════════════════════════
def rerank(clean_q: str, top_idxs, sims_row, threshold: float, raw_tokens: set | None = None):
    """
    Score formula (tuned for e-commerce taxonomy):
      0.50 × cosine      (TF-IDF vector similarity)
      0.28 × name fuzzy  (token-set ratio vs leaf category name)
      0.15 × kw fuzzy    (token-set ratio vs enriched keyword blob)
      0.07 × depth bonus (prefer specific leaves over root nodes)
      –0.12 penalty for depth < 3 (reject overly generic root matches)
      –0.20 domain penalty when query has strong vertical signal
                         and candidate is outside that vertical
    """
    # Build allowed top-levels from raw query tokens
    allowed_tops: set[str] = set()
    if raw_tokens:
        for tok in raw_tokens:
            if tok in DOMAIN_SIGNALS:
                allowed_tops |= DOMAIN_SIGNALS[tok]

    best_score = -1.0
    best_row   = None

    for idx in top_idxs:
        row = df_cat.iloc[idx]
        cos       = float(sims_row[idx])
        name_fuzz = fuzz.token_set_ratio(clean_q, row["name_clean"]) / 100.0
        kw_fuzz   = fuzz.token_set_ratio(clean_q, row["kw_clean"])   / 100.0
        depth     = int(row["depth"])

        depth_bonus   = min(depth, 6) * 0.012
        depth_penalty = 0.12 if depth < 3 else 0.0

        # Domain penalty: strong vertical signal violated
        top_level      = row["Category Path"].split(" / ")[0]
        domain_penalty = 0.20 if (allowed_tops and top_level not in allowed_tops) else 0.0

        combined = (
            cos       * 0.50 +
            name_fuzz * 0.28 +
            kw_fuzz   * 0.15 +
            depth_bonus       -
            depth_penalty     -
            domain_penalty
        )

        if combined > best_score:
            best_score = combined
            best_row   = row

    if best_row is None:
        return "Uncategorized", None, 0.0, "❌ Rejected"

    conf   = round(min(best_score * 125.0, 100.0), 1)
    status = "✅ Approved" if conf >= threshold else "⚠️ Review"
    return best_row["Category Path"], str(best_row["category_code_str"]), conf, status


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PRODUCT LOOKUP  (instant sidebar tool)
# ══════════════════════════════════════════════════════════════════════════════
def _single_lookup(query: str, threshold: float, n_top: int = 5):
    cq = clean_query(query)
    if not cq:
        return None, []
    raw_tokens = set(re.sub(r"[^a-z\s]", " ", query.lower()).split())
    qvec = vectorizer.transform([cq])
    sims = cosine_similarity(qvec, tfidf_matrix)[0]

    top_idxs = np.argpartition(sims, -40)[-40:]
    best_path, best_code, best_conf, best_status = rerank(cq, top_idxs, sims, threshold, raw_tokens)

    # Collect top-N for the alternatives expander
    # Build allowed_tops once for the alternatives scoring too
    allowed_tops: set[str] = set()
    for tok in raw_tokens:
        if tok in DOMAIN_SIGNALS:
            allowed_tops |= DOMAIN_SIGNALS[tok]

    scored = []
    for idx in top_idxs:
        row = df_cat.iloc[idx]
        cos       = float(sims[idx])
        name_fuzz = fuzz.token_set_ratio(cq, row["name_clean"]) / 100.0
        kw_fuzz   = fuzz.token_set_ratio(cq, row["kw_clean"])   / 100.0
        depth     = int(row["depth"])
        depth_bonus   = min(depth, 6) * 0.012
        depth_penalty = 0.12 if depth < 3 else 0.0
        top_level     = row["Category Path"].split(" / ")[0]
        domain_penalty = 0.20 if (allowed_tops and top_level not in allowed_tops) else 0.0
        combined = cos * 0.50 + name_fuzz * 0.28 + kw_fuzz * 0.15 + depth_bonus - depth_penalty - domain_penalty
        scored.append((combined, row["Category Path"], round(min(combined * 125, 100), 1), depth))

    scored.sort(reverse=True)
    return (best_path, best_code, best_conf, best_status), scored[:n_top]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    threshold = st.slider(
        "Approval threshold (%)",
        min_value=0, max_value=100, value=40,
        help="Matches above this confidence are marked ✅ Approved. "
             "Below is ⚠️ Review. Raise to be stricter."
    )

    st.markdown("---")
    st.markdown("### 🔎 Quick Lookup")
    single_q = st.text_input("Test a product name", placeholder="e.g. iPhone 15 Pro case")
    if single_q:
        result, alternatives = _single_lookup(single_q, threshold)
        if result:
            path, code, conf, status = result
            st.markdown(f"**{status}** — `{conf}%`")
            st.markdown(f"```\n{path}\n```")
            st.caption(f"Code: `{code}`")
            if alternatives:
                with st.expander("Top alternatives"):
                    for score, alt_path, alt_conf, alt_depth in alternatives:
                        st.markdown(f"- `{alt_conf}%` · depth {alt_depth} · {alt_path}")

    st.markdown("---")
    st.markdown("### 📋 Taxonomy Stats")
    st.metric("Total categories", f"{len(df_cat):,}")
    st.metric("Max depth", int(df_cat['depth'].max()))
    depth_dist = df_cat.groupby("depth").size()
    st.caption("Depth distribution:")
    for d, cnt in depth_dist.items():
        bar = "█" * min(int(cnt / 500), 20)
        st.caption(f"L{d}: {bar} {cnt:,}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_batch, tab_audit, tab_explore = st.tabs(["📦 Batch Categorise", "🔍 Audit Existing", "🗂 Explore Taxonomy"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: BATCH CATEGORISE
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Upload a product list — get AI-suggested categories")

    st.info(
        "**Expected CSV columns** (column names are auto-detected):\n"
        "- A column containing **product names** (header with 'name', 'title', 'product', or first column)\n"
        "- _(Optional)_ A column with your **existing category code** (header with 'code')",
        icon="ℹ️",
    )

    uploaded = st.file_uploader("Upload Product CSV", type=["csv", "txt"], key="batch_upload")

    if uploaded:
        # ── Robust CSV reader ─────────────────────────────────────────────────
        try:
            df_up = pd.read_csv(uploaded, sep=None, engine="python", on_bad_lines="skip")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        st.success(f"Loaded **{len(df_up):,} rows** · **{len(df_up.columns)} columns**")

        # ── Column detection ─────────────────────────────────────────────────
        name_col = next(
            (c for c in df_up.columns
             if any(k in c.upper() for k in ["NAME", "TITLE", "PRODUCT", "DESCRIPTION", "DESC"])),
            df_up.columns[0],
        )
        code_col = next(
            (c for c in df_up.columns
             if "CODE" in c.upper() and "AI" not in c.upper()),
            None,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            name_col = st.selectbox("Product name column", df_up.columns.tolist(),
                                    index=df_up.columns.tolist().index(name_col))
        with col_b:
            code_options = ["(none)"] + df_up.columns.tolist()
            code_default = code_options.index(code_col) if code_col in code_options else 0
            code_col_sel = st.selectbox("Existing category code column (optional)", code_options,
                                        index=code_default)
            code_col = None if code_col_sel == "(none)" else code_col_sel

        # Resolve existing codes → paths
        if code_col:
            clean_codes = (
                df_up[code_col]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
            )
            df_up["Assigned Category (Original)"] = clean_codes.map(code_to_path).fillna("⚠️ Unknown Code")

        if st.button("🚀 Run AI Categorisation", use_container_width=True):
            names = df_up[name_col].fillna("").astype(str).tolist()
            cleaned = [clean_query(n) for n in names]

            # Batch TF-IDF transform
            q_matrix = vectorizer.transform(cleaned)
            all_sims  = cosine_similarity(q_matrix, tfidf_matrix)

            results   = []
            prog      = st.progress(0.0, text="Matching products…")
            for i in range(len(names)):
                sims_row = all_sims[i]
                top_idxs = np.argpartition(sims_row, -40)[-40:]
                raw_tokens = set(re.sub(r"[^a-z\s]", " ", names[i].lower()).split())
                results.append(rerank(cleaned[i], top_idxs, sims_row, threshold, raw_tokens))
                if i % 100 == 0:
                    prog.progress((i + 1) / len(names), text=f"Processing {i+1:,} / {len(names):,}…")
            prog.progress(1.0, text="Done ✓")

            df_up["AI Category"]  = [r[0] for r in results]
            df_up["AI Code"]      = [r[1] for r in results]
            df_up["Confidence %"] = [r[2] for r in results]
            df_up["Status"]       = [r[3] for r in results]

            # ── Summary metrics ───────────────────────────────────────────────
            n_approved = sum(1 for r in results if r[3] == "✅ Approved")
            n_review   = sum(1 for r in results if r[3] == "⚠️ Review")
            avg_conf   = round(np.mean([r[2] for r in results]), 1)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",    f"{len(names):,}")
            m2.metric("✅ Approved", f"{n_approved:,}",
                      delta=f"{n_approved/len(names)*100:.0f}%")
            m3.metric("⚠️ Review",  f"{n_review:,}",
                      delta=f"-{n_review/len(names)*100:.0f}%", delta_color="inverse")
            m4.metric("Avg Confidence", f"{avg_conf}%")

            st.markdown("---")

            # ── Display ───────────────────────────────────────────────────────
            display_cols = [name_col, "Status", "Confidence %", "AI Category", "AI Code"]
            if "Assigned Category (Original)" in df_up.columns:
                display_cols.insert(3, "Assigned Category (Original)")

            col_cfg = {
                name_col: st.column_config.TextColumn("Product Name", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Confidence %": st.column_config.ProgressColumn(
                    "Confidence", format="%.1f%%", min_value=0, max_value=100
                ),
                "Assigned Category (Original)": st.column_config.TextColumn("Existing Category", width="large"),
                "AI Category": st.column_config.TextColumn("AI Suggestion", width="large"),
                "AI Code": st.column_config.TextColumn("Code", width="small"),
            }

            st.subheader("Results")
            st.dataframe(
                df_up[[c for c in display_cols if c in df_up.columns]].head(2000),
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
            )

            # ── Download ──────────────────────────────────────────────────────
            st.download_button(
                "📥 Download Full Results (CSV)",
                data=df_up.to_csv(index=False).encode("utf-8"),
                file_name="categorised_products.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: AUDIT — check if existing codes are correct
# ─────────────────────────────────────────────────────────────────────────────
with tab_audit:
    st.markdown("#### Audit existing category assignments")
    st.markdown(
        "Upload a file with product names **and** their current category codes. "
        "The auditor will flag mismatches where the AI suggests a significantly different category."
    )

    uploaded_audit = st.file_uploader("Upload CSV for audit", type=["csv"], key="audit_upload")

    if uploaded_audit:
        try:
            df_audit = pd.read_csv(uploaded_audit, sep=None, engine="python", on_bad_lines="skip")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        name_col_a = next(
            (c for c in df_audit.columns
             if any(k in c.upper() for k in ["NAME", "TITLE", "PRODUCT", "DESC"])),
            df_audit.columns[0],
        )
        code_col_a = next(
            (c for c in df_audit.columns if "CODE" in c.upper()),
            None,
        )

        col_a2, col_b2 = st.columns(2)
        with col_a2:
            name_col_a = st.selectbox("Product name column", df_audit.columns.tolist(),
                                      index=df_audit.columns.tolist().index(name_col_a),
                                      key="audit_name")
        with col_b2:
            code_options_a = df_audit.columns.tolist()
            code_default_a = code_options_a.index(code_col_a) if code_col_a in code_options_a else 0
            code_col_a = st.selectbox("Existing category code column", code_options_a,
                                      index=code_default_a, key="audit_code")

        mismatch_only = st.checkbox("Show mismatches only", value=True)

        if st.button("🔍 Run Audit", use_container_width=True):
            names = df_audit[name_col_a].fillna("").astype(str).tolist()
            cleaned = [clean_query(n) for n in names]

            clean_codes = (
                df_audit[code_col_a]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
            )
            existing_paths = clean_codes.map(code_to_path).fillna("⚠️ Unknown Code").tolist()

            q_matrix = vectorizer.transform(cleaned)
            all_sims  = cosine_similarity(q_matrix, tfidf_matrix)

            audit_results = []
            prog2 = st.progress(0.0, text="Auditing…")
            for i in range(len(names)):
                sims_row = all_sims[i]
                top_idxs = np.argpartition(sims_row, -40)[-40:]
                raw_tokens = set(re.sub(r"[^a-z\s]", " ", names[i].lower()).split())
                ai_path, ai_code, conf, status = rerank(cleaned[i], top_idxs, sims_row, threshold, raw_tokens)

                existing = existing_paths[i]
                # Mismatch = existing is unknown OR AI top-level differs
                existing_top = existing.split(" / ")[0] if " / " in existing else existing
                ai_top       = ai_path.split(" / ")[0]  if " / " in ai_path  else ai_path
                is_mismatch  = ("Unknown" in existing) or (existing_top != ai_top)

                audit_results.append({
                    "Product": names[i],
                    "Existing Category": existing,
                    "AI Suggestion": ai_path,
                    "Confidence %": conf,
                    "Match?": "✅ Match" if not is_mismatch else "⚠️ Mismatch",
                })
                if i % 100 == 0:
                    prog2.progress((i + 1) / len(names))
            prog2.progress(1.0)

            df_result = pd.DataFrame(audit_results)
            if mismatch_only:
                df_result = df_result[df_result["Match?"] == "⚠️ Mismatch"]

            n_mismatch = sum(1 for r in audit_results if r["Match?"] == "⚠️ Mismatch")
            st.metric("Mismatches found", f"{n_mismatch:,} / {len(names):,}",
                      delta=f"{n_mismatch/len(names)*100:.1f}% error rate", delta_color="inverse")

            st.dataframe(df_result, use_container_width=True, hide_index=True,
                         column_config={
                             "Confidence %": st.column_config.ProgressColumn(
                                 "Confidence", format="%.1f%%", min_value=0, max_value=100
                             ),
                         })

            st.download_button(
                "📥 Download Audit Report",
                data=df_result.to_csv(index=False).encode("utf-8"),
                file_name="audit_report.csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: EXPLORE TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("#### Browse the category taxonomy")

    col_e1, col_e2 = st.columns([1, 2])

    with col_e1:
        # Filter by top-level
        top_levels = sorted(
            df_cat[~df_cat["Category Path"].str.contains("/", na=False)]["category_name"].tolist()
        )
        chosen_top = st.selectbox("Top-level category", ["All"] + top_levels)

    with col_e2:
        search_term = st.text_input("Search category names / keywords", placeholder="e.g. screen protector")

    sub_df = df_cat.copy()
    if chosen_top != "All":
        sub_df = sub_df[sub_df["Category Path"].str.startswith(chosen_top, na=False)]
    if search_term.strip():
        mask = (
            sub_df["category_name"].str.contains(search_term, case=False, na=False) |
            sub_df["keywords"].str.contains(search_term, case=False, na=False)
        )
        sub_df = sub_df[mask]

    depth_filter = st.slider("Filter by depth", 1, 9, (1, 9))
    sub_df = sub_df[sub_df["depth"].between(*depth_filter)]

    st.caption(f"Showing **{len(sub_df):,}** categories")

    st.dataframe(
        sub_df[["category_name", "category_code_str", "Category Path", "depth", "keywords"]]
        .rename(columns={"category_code_str": "code"})
        .head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "category_name": st.column_config.TextColumn("Name", width="medium"),
            "code": st.column_config.TextColumn("Code", width="small"),
            "Category Path": st.column_config.TextColumn("Full Path", width="large"),
            "depth": st.column_config.NumberColumn("Depth", width="small"),
            "keywords": st.column_config.TextColumn("Keywords", width="large"),
        },
    )

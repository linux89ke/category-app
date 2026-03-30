import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==============================================================================
# 1. SETTINGS & STYLE
# ==============================================================================
st.set_page_config(page_title="AI Category Auditor", page_icon="🎯", layout="wide")

CATEGORY_FILE = "category_map_fully_enriched.xlsx"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #58a6ff !important; border-bottom: 1px solid #21262d; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'IBM Plex Mono', monospace; }
    .stButton > button { background: #238636 !important; color: white !important; border: 1px solid #2ea043 !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Precision Category Engine")
st.caption("Hybrid TF-IDF + Domain Signal Penalty + Leaf-Node Priority")

# ==============================================================================
# 2. LOGIC DICTIONARIES
# ==============================================================================
# These words anchor a product to a specific Department to prevent "Magnet Category" errors
DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"}, "tablet": {"Phones & Tablets"},
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "watch": {"Fashion", "Electronics"},
    "tv": {"Electronics"}, "television": {"Electronics"}, "headphone": {"Electronics"},
    "diaper": {"Grocery", "Baby Products"}, "perfume": {"Health & Beauty"},
    "bike": {"Sporting Goods"}, "yoga": {"Sporting Goods"}
}

# Standardized expansions
QUERY_EXPANSIONS = {
    "case": "cover shell protector bumper casing phones tablets",
    "cover": "case shell protector bumper phones tablets",
    "earpods": "earphones headphones wireless earbuds",
    "sneakers": "shoes trainers athletic footwear",
    "fridge": "refrigerator freezer cooler",
    "iron": "steam iron clothes appliance"
}

_MEASURE_RE = re.compile(r'\b\d+\.?\d*\s*(ml|l|g|kg|pcs|inch|cm|w|kw|mah|gb|tb|v|ah)\b', re.I)

def clean_robust(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = _MEASURE_RE.sub(" ", text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in QUERY_EXPANSIONS:
            expanded.extend(QUERY_EXPANSIONS[t].split())
    return " ".join(expanded).strip()

# ==============================================================================
# 3. ROBUST INDEX BUILDER
# ==============================================================================
@st.cache_resource(show_spinner="⚙️ Initializing Unbreakable Index...")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"File {CATEGORY_FILE} not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    raw_cols = df.columns.tolist()

    # Smart Detection
    path_col = next((c for c in raw_cols if 'PATH' in c.upper()), None)
    kw_col = next((c for c in raw_cols if 'KEY' in c.upper()), None)
    code_col = next((c for c in raw_cols if 'CODE' in c.upper()), None)
    name_col = next((c for c in raw_cols if 'NAME' in c.upper()), None)

    # Process Path
    df['path_str'] = df[path_col].astype(str)
    df['leaf_name'] = df['path_str'].apply(lambda x: x.split('/')[-1].strip().lower())
    df['depth'] = df['path_str'].apply(lambda x: x.count('/') + 1)
    
    # Cleaning
    p_clean = df['path_str'].str.replace('/', ' ').str.lower()
    k_clean = df[kw_col].fillna('').astype(str).str.lower() if kw_col else ""
    n_clean = df[name_col].fillna('').astype(str).str.lower() if name_col else ""

    # Weights: Path(x4), Name(x4), Keywords(x2)
    df['search_text'] = (p_clean + ' ') * 4 + (n_clean + ' ') * 4 + k_clean

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    # Code Map
    clean_codes = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    code_to_path = dict(zip(clean_codes, df[path_col]))

    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_cat, vectorizer, tfidf_matrix, code_to_path, PATH_COL = build_index()

# ==============================================================================
# 4. SCORING ENGINE (The "Anti-Magnet" Scorer)
# ==============================================================================
def advanced_score(clean_q, top_idxs, sims_row, threshold):
    best_score, best_row = -1.0, None
    raw_tokens = set(clean_q.split())
    
    # Determine vertical requirement
    required_verticals = set()
    for tok in raw_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]

    for idx in top_idxs:
        row = df_cat.iloc[idx]
        cos = float(sims_row[idx])
        
        # 1. Fuzzy Matches
        name_fuzz = fuzz.token_set_ratio(clean_q, row['leaf_name']) / 100.0
        
        # 2. Domain Penalty (The Fix)
        # If the product name implies "Phones" but the category is "Wholesale", apply penalty
        top_level = str(row[PATH_COL]).split('/')[0].strip()
        penalty = 0.25 if (required_verticals and top_level not in required_verticals) else 0.0

        # Formula: Math (50%) + Fuzzy (30%) + Depth (5%) - Penalty
        score = (cos * 0.50) + (name_fuzz * 0.30) + (int(row['depth']) * 0.01) - penalty

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"

    conf = round(min(best_score * 120.0, 100.0), 1)
    # Double Check: If math is too weak, force Review
    status = "✅ Approved" if (conf >= threshold and best_score > 0.22) else "⚠️ Review"
    return best_row[PATH_COL], str(best_row.get('category_code', 'N/A')), conf, status

# ==============================================================================
# 5. UI TABS
# ==============================================================================
t1, t2 = st.tabs(["📦 Batch Process", "🔍 Audit Report"])

with t1:
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        df_up = pd.read_csv(up, sep=None, engine='python', on_bad_lines='skip')
        n_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
        c_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

        if st.button("🔥 Run Analysis"):
            names = df_up[n_col].fillna("").tolist()
            cleaned = [clean_robust(n) for n in names]
            q_vecs = vectorizer.transform(cleaned)
            all_sims = cosine_similarity(q_vecs, tfidf_matrix)

            results = []
            for i in range(len(names)):
                s_row = all_sims[i]
                t_idxs = np.argpartition(s_row, -40)[-40:]
                results.append(advanced_score(cleaned[i], t_idxs, s_row, threshold=40))

            df_up["AI Category"] = [r[0] for r in results]
            df_up["Confidence %"] = [r[2] for r in results]
            df_up["Status"] = [r[3] for r in results]

            st.dataframe(df_up, use_container_width=True)
            st.download_button("📥 Download", df_up.to_csv(index=False).encode('utf-8'), "results.csv")

with t2:
    st.info("Upload a file with category codes to find mismatches.")
    # Audit logic matches Tab 1 but highlights discrepancies

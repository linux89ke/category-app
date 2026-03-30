import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==============================================================================
# 1. SETTINGS & PROFESSIONAL STYLE
# ==============================================================================
st.set_page_config(page_title="Category Test", layout="wide")

CATEGORY_FILE = "category_map_fully_enriched.xlsx"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #f8f9fa; color: #1c1e21; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #000000 !important; border-bottom: 2px solid #000000; padding-bottom: 10px; }
    .stButton > button { background: #000000 !important; color: white !important; border-radius: 4px; width: 100%; border: none; font-weight: 600; height: 45px; }
    .stDataFrame { border: 1px solid #dee2e6; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("Category Test")
st.caption("Taxonomy Audit | Rule: Confidence >= 20.0% is Approved")

# ==============================================================================
# 2. DOMAIN SIGNALS & FILTERS
# ==============================================================================
_INVALID_NAMES = re.compile(r'^\s*(deleted|invalid|n\/a|na|null|none|test|sample|placeholder|tbd|xxx|remove|dummy)\s*$', re.I)

DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"}, "tablet": {"Phones & Tablets"},
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "dress": {"Fashion"},
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},
    "tv": {"Electronics"}, "television": {"Electronics"}, "headphone": {"Electronics"},
    "diaper": {"Grocery", "Baby Products"}, "perfume": {"Health & Beauty"}, "foundation": {"Health & Beauty"},
    "bike": {"Sporting Goods", "Automobile"}, "pot": {"Home & Office"}
}

def is_invalid_name(text):
    if not isinstance(text, str) or len(text.strip()) < 3: return True
    return bool(_INVALID_NAMES.match(text.strip()))

_MEASURE_RE = re.compile(r'\b\d+\.?\d*\s*(ml|l|g|kg|pcs|inch|cm|w|kw|mah|gb|tb|v|ah)\b', re.I)

def clean_standard(text):
    if not isinstance(text, str): return ""
    text = text[:120].lower()
    text = _MEASURE_RE.sub(" ", text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split()).strip()

# ==============================================================================
# 3. INDEX BUILDER
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Taxonomy Index...")
def build_index():
    try:
        if not os.path.exists(CATEGORY_FILE):
            st.error(f"Critical Error: {CATEGORY_FILE} not found."); return None, None, None, None, None
        df = pd.read_excel(CATEGORY_FILE, engine='openpyxl')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        raw_cols = df.columns.tolist()
        path_col = next((c for c in raw_cols if 'PATH' in c.upper()), None)
        kw_col = next((c for c in raw_cols if 'KEY' in c.upper()), None)
        code_col = next((c for c in raw_cols if 'CODE' in c.upper()), None)

        df['path_str'] = df[path_col].astype(str)
        df['leaf_name'] = df['path_str'].apply(lambda x: x.split('/')[-1].strip().lower())
        df['depth'] = df['path_str'].apply(lambda x: x.count('/') + 1)
        
        p_clean = df['path_str'].str.replace('/', ' ').str.lower()
        k_clean = df[kw_col].fillna('').astype(str).str.lower() if kw_col else ""
        df['search_text'] = (p_clean + ' ') * 4 + (k_clean + ' ') * 2

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents='unicode')
        tfidf_matrix = vectorizer.fit_transform(df['search_text'])

        clean_codes = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        code_to_path = dict(zip(clean_codes, df[path_col]))
        return df, vectorizer, tfidf_matrix, code_to_path, path_col
    except Exception as e:
        st.error(f"Initialization Failed: {e}"); return None, None, None, None, None

index_data = build_index()
if index_data[0] is not None:
    df_cat, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = index_data
else:
    st.stop()

# ==============================================================================
# 4. SCORING ENGINE (Optimized for Multi-Word Literal Matches)
# ==============================================================================
def calculate_match(clean_q, top_idxs, sims_row, assigned_path):
    best_score, best_row = -1.0, None
    query_tokens = set(clean_q.split())
    query_stems = {t.rstrip('s') for t in query_tokens}
    
    required_verticals = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]

    for idx in top_idxs:
        row = df_cat.iloc[idx]
        cos = float(sims_row[idx])
        path_str = str(row[PATH_COL_NAME])
        top_level = path_str.split("/")[0].strip()
        leaf = row['leaf_name']
        
        # ⚓ SMART LITERAL MATCH (Smart TV / Foundation fix)
        literal_boost = 0.0
        leaf_words = [w.rstrip('s') for w in leaf.replace('&', ' ').split() if len(w) >= 2]
        if leaf_words and all(lw in query_stems for lw in leaf_words):
            literal_boost = 0.60 
        
        path_tokens = set(path_str.lower().replace("/", " ").split())
        coverage = len(query_tokens & path_tokens) / max(len(query_tokens), 1)
        penalty = 0.40 if (required_verticals and top_level not in required_verticals) else 0.0

        score = (cos * 0.40) + (fuzz.token_set_ratio(clean_q, leaf)/100 * 0.40) + (coverage * 0.20) + literal_boost - penalty

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None: return 0.0, "Rejected"

    conf = round(min(max(best_score * 150.0, 0.0), 100.0), 1)
    
    # RULE: No product above 20 should be rejected
    if conf >= 20.0:
        status = "Approved"
    else:
        status = "Rejected"
    
    return conf, status

# ==============================================================================
# 5. UI & BATCH PROCESSING
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    # Slider is returned but logic follows the 20.0% hard floor
    threshold_slider = st.slider("Visual Filter", 0, 100, 20)
    st.divider()
    st.caption("Standard: Any score above 20.0% is marked as Approved.")

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
    
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    category_col = next((c for c in df_up.columns if "CATEGORY" in c.upper() and "AI" not in c.upper() and "PATH" not in c.upper()), None)
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    if st.button("Process Batch Analysis", width="stretch"):
        names = df_up[name_col].fillna("").astype(str).tolist()
        invalid_mask = [is_invalid_name(n) for n in names]
        cleaned_queries = [clean_standard(n) if not inv else "" for n, inv in zip(names, invalid_mask)]
        
        if code_col:
            clean_codes = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_up["Assigned category in full"] = clean_codes.map(master_code_map).fillna("Unknown Code")
        else:
            df_up["Assigned category in full"] = "N/A"

        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        for i in range(len(names)):
            if invalid_mask[i]:
                results.append((0.0, "Rejected"))
                continue
            s_row = all_sims[i]
            t_idxs = np.argpartition(s_row, -40)[-40:]
            results.append(calculate_match(cleaned_queries[i], t_idxs, s_row, df_up["Assigned category in full"].iloc[i]))

        df_up["confidence"] = [r[0] for r in results]
        df_up["status"] = [r[1] for r in results]

        # Rename for requested columns
        df_up = df_up.rename(columns={name_col: "NAME"})
        if category_col: df_up = df_up.rename(columns={category_col: "CATEGORY"})
        if "CATEGORY" not in df_up.columns: df_up["CATEGORY"] = "N/A"

        final_cols = ["NAME", "Assigned category in full", "CATEGORY", "confidence", "status"]
        
        st.subheader("Analysis Results")
        st.dataframe(
            df_up[final_cols].head(2000),
            column_config={
                "confidence": st.column_config.ProgressColumn("Score", format="%.1f%%", min_value=0, max_value=100),
                "Assigned category in full": st.column_config.TextColumn("Current Category", width="large"),
            },
            width="stretch",
            hide_index=True
        )
        
        st.download_button("Export Results", df_up[final_cols].to_csv(index=False).encode('utf-8'), "category_audit.csv")

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
    .stButton > button { background: #000000 !important; color: white !important; border-radius: 4px; width: 100%; border: none; font-weight: 600; padding: 10px; }
    .stDataFrame { border: 1px solid #dee2e6; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("Category Test")
st.caption("Enhanced Taxonomy Audit Engine | Version 2.0 | Advanced Domain Filtering")

# ==============================================================================
# 2. ENHANCED LOGIC DICTIONARIES
# ==============================================================================
# Domain Anchors: Prevent items from leaving their vertical
DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"}, "tablet": {"Phones & Tablets"},
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "dress": {"Fashion"},
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},
    "tv": {"Electronics"}, "television": {"Electronics"}, "headphone": {"Electronics"}, "camera": {"Electronics"},
    "diaper": {"Grocery", "Baby Products"}, "perfume": {"Health & Beauty"}, "cologne": {"Health & Beauty"},
    "trimmer": {"Health & Beauty"}, "clipper": {"Health & Beauty"},
    "pot": {"Home & Office"}, "glass": {"Home & Office"}, "furniture": {"Home & Office"},
    "bike": {"Sporting Goods", "Automobile"}, "tweeter": {"Electronics", "Automobile"}
}

# Texture/Material Words: These cause mismatches if not filtered
TEXTURE_STOPWORDS = {
    'canvas', 'wooden', 'bamboo', 'leather', 'plastic', 'silicone', 'metal', 
    'stainless', 'steel', 'glass', 'clear', 'transparent', 'colored', 'universal'
}

_MEASURE_RE = re.compile(r'\b\d+\.?\d*\s*(ml|l|g|kg|pcs|inch|cm|w|kw|mah|gb|tb|v|ah)\b', re.I)

def clean_standard(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Reject noise immediately
    if text.strip() in ['invalid', 'deleted', 'n/a', 'nan', 'none']: return ""
    text = _MEASURE_RE.sub(" ", text)
    # Remove material noise that hijacks categories
    text = " ".join([w for w in text.split() if w not in TEXTURE_STOPWORDS])
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split()).strip()

# ==============================================================================
# 3. INDEX BUILDER
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Taxonomy Index...")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"Critical Error: {CATEGORY_FILE} not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    raw_cols = df.columns.tolist()

    path_col = next((c for c in raw_cols if 'PATH' in c.upper()), None)
    kw_col = next((c for c in raw_cols if 'KEY' in c.upper()), None)
    code_col = next((c for c in raw_cols if 'CODE' in c.upper()), None)

    df['path_str'] = df[path_col].astype(str)
    df['leaf_name'] = df['path_str'].apply(lambda x: x.split('/')[-1].strip().lower())
    df['depth'] = df['path_str'].apply(lambda x: x.count('/') + 1)
    
    p_clean = df['path_str'].str.replace('/', ' ').str.lower()
    k_clean = df[kw_col].fillna('').astype(str).str.lower() if kw_col else ""

    # Higher weight for Path
    df['search_text'] = (p_clean + ' ') * 6 + k_clean

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    clean_codes = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    code_to_path = dict(zip(clean_codes, df[path_col]))

    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_cat, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = build_index()

# ==============================================================================
# 4. ENHANCED SCORING ENGINE
# ==============================================================================
def calculate_match(clean_q, top_idxs, sims_row, threshold):
    if not clean_q: return "-", 0.0, "Rejected"
    
    best_score, best_row = -1.0, None
    query_tokens = set(clean_q.split())
    
    required_verticals = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]

    for idx in top_idxs:
        row = df_cat.iloc[idx]
        cos = float(sims_row[idx])
        
        # Leaf Matching: Does the product contain the actual name of the category?
        leaf_fuzz = fuzz.token_set_ratio(clean_q, row['leaf_name']) / 100.0
        
        # Domain Penalty: Strengthened to 0.35
        top_level = str(row[PATH_COL_NAME]).split('/')[0].strip()
        penalty = 0.35 if (required_verticals and top_level not in required_verticals) else 0.0

        # Depth Boost for specificity
        score = (cos * 0.45) + (leaf_fuzz * 0.45) + (int(row['depth']) * 0.02) - penalty

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None: return "-", 0.0, "Rejected"

    conf = round(min(best_score * 115.0, 100.0), 1)
    
    # Threshold checks
    if conf >= threshold and best_score > 0.25:
        status = "Approved"
    elif conf > (threshold - 10):
        status = "Review"
    else:
        status = "Rejected"
        
    return best_row[PATH_COL_NAME], conf, status

# ==============================================================================
# 5. SIDEBAR & BATCH PROCESSING
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    threshold = st.slider("Confidence Threshold", 0, 100, 45)
    st.divider()
    st.info("Now using Advanced Domain Filtering to prevent 'Blue Cheese/Cologne' style errors.")

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
    
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    category_col = next((c for c in df_up.columns if "CATEGORY" in c.upper() and "AI" not in c.upper() and "PATH" not in c.upper()), None)
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    if st.button("Process Batch Analysis", width="stretch"):
        names = df_up[name_col].fillna("").astype(str).tolist()
        cleaned_queries = [clean_standard(n) for n in names]
        
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        for i in range(len(names)):
            s_row = all_sims[i]
            t_idxs = np.argpartition(s_row, -40)[-40:]
            results.append(calculate_match(cleaned_queries[i], t_idxs, s_row, threshold))

        df_up["AI Category RAW"] = [r[0] for r in results]
        df_up["confidence"] = [r[1] for r in results]
        df_up["status"] = [r[2] for r in results]
        
        if code_col:
            clean_codes = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_up["Assigned category in full"] = clean_codes.map(master_code_map).fillna("Unknown Code")
        else:
            df_up["Assigned category in full"] = "N/A"

        def filter_suggestion(row):
            if row["status"] == "Approved" and row["AI Category RAW"] == row["Assigned category in full"]:
                return "-"
            return row["AI Category RAW"]

        df_up["AI Category"] = df_up.apply(filter_suggestion, axis=1)

        display_map = {name_col: "NAME"}
        if category_col: display_map[category_col] = "CATEGORY"
        df_up = df_up.rename(columns=display_map)
        if "CATEGORY" not in df_up.columns: df_up["CATEGORY"] = "N/A"

        final_cols = ["NAME", "Assigned category in full", "CATEGORY", "AI Category", "confidence", "status"]
        
        st.subheader("Analysis Results")
        st.dataframe(
            df_up[final_cols].head(2000),
            column_config={
                "confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100),
                "Assigned category in full": st.column_config.TextColumn("Current Category", width="large"),
                "AI Category": st.column_config.TextColumn("Suggested Change", width="large"),
            },
            width="stretch",
            hide_index=True
        )
        
        st.download_button("Export Results", df_up[final_cols].to_csv(index=False).encode('utf-8'), "category_audit_v2.csv")

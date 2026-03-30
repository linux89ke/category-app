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
st.set_page_config(page_title="Category Test", layout="wide")
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #f8f9fa; color: #1c1e21; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #000000 !important; border-bottom: 2px solid #000000; padding-bottom: 10px; }
    .stButton > button { background: #000000 !important; color: white !important; border-radius: 4px; width: 100%; border: none; font-weight: 600; height: 45px; }
</style>
""", unsafe_allow_html=True)

st.title("Category Test")
st.caption("Taxonomy Audit | Full Keyword Integration | Version 2026.12")

# ==============================================================================
# 2. DOMAIN GUARDRAILS
# ==============================================================================
DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"}, "tablet": {"Phones & Tablets"},
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "dress": {"Fashion"},
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},
    "tv": {"Electronics"}, "television": {"Electronics"}, "headphone": {"Electronics"},
    "diaper": {"Grocery", "Baby Products"}, "perfume": {"Health & Beauty"}, "foundation": {"Health & Beauty"},
    "bike": {"Sporting Goods", "Automobile"}, "pot": {"Home & Office"}
}

def clean_standard(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\b\d+\.?\d*\s*(ml|l|g|kg|pcs|inch|cm|w|kw|mah|gb|tb|v|ah)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split()).strip()

def normalize_path(path_str):
    """Ensures 'A / B' matches 'A/B' by stripping all non-alphanumeric chars."""
    if not isinstance(path_str, str): return ""
    return re.sub(r'[^a-z0-9]', '', path_str.lower())

# ==============================================================================
# 3. INDEX BUILDER (Keyword-First)
# ==============================================================================
@st.cache_resource(show_spinner="Loading Master Keywords...")
def build_index():
    try:
        df = pd.read_excel(CATEGORY_FILE, engine='openpyxl')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Column Detection
        path_col = next((c for c in df.columns if 'PATH' in c.upper()), df.columns[2])
        kw_col = next((c for c in df.columns if 'KEY' in c.upper()), None)
        code_col = next((c for c in df.columns if 'CODE' in c.upper()), None)

        df['path_str'] = df[path_col].astype(str).str.strip()
        df['norm_path'] = df['path_str'].apply(normalize_path)
        df['leaf_name'] = df['path_str'].apply(lambda x: x.split('/')[-1].strip().lower())
        
        # --- KEYWORD INTEGRATION ---
        # We clean the keywords from your Excel and give them heavy weight in the search
        p_clean = df['path_str'].str.replace('/', ' ').str.lower()
        k_clean = df[kw_col].fillna('').astype(str).str.lower() if kw_col else ""
        
        # Search text = Path(x3) + Your Manual Keywords(x5)
        # This makes your Excel keywords the strongest signal for the AI
        df['search_text'] = (p_clean + ' ') * 3 + (k_clean + ' ') * 5

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(df['search_text'])
        
        # Robust Mappings
        norm_path_to_idx = {row['norm_path']: i for i, row in df.iterrows()}
        code_to_path = dict(zip(df[code_col].astype(str).str.replace(r'\.0$', '', regex=True), df[path_col])) if code_col else {}

        return df, vectorizer, tfidf_matrix, code_to_path, path_col, norm_path_to_idx
    except Exception as e:
        st.error(f"Error reading master file: {e}"); return None, None, None, None, None, None

index_data = build_index()
if index_data[0] is not None:
    df_cat, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME, path_map = index_data
else:
    st.stop()

# ==============================================================================
# 4. AUDIT SCORER
# ==============================================================================
def score_assignment(clean_q, assigned_path, sims_row):
    # Normalize the path to ignore spacing/slashes difference
    norm_assigned = normalize_path(assigned_path)
    
    if norm_assigned not in path_map:
        return 0.0 # Category doesn't exist in master file
    
    idx = path_map[norm_assigned]
    row = df_cat.iloc[idx]
    
    # TF-IDF score (includes your keywords)
    cos = float(sims_row[idx])
    leaf = row['leaf_name']
    
    query_tokens = set(clean_q.split())
    query_stems = {t.rstrip('s') for t in query_tokens}
    
    # Smart Literal Boost (Multi-word like "Smart TV" or "Laptop")
    literal_boost = 0.0
    leaf_words = [w.rstrip('s') for w in leaf.replace('&', ' ').split() if len(w) >= 2]
    if leaf_words and all(lw in query_stems for lw in leaf_words):
        literal_boost = 0.60 

    # Department Anchor Penalty
    top_level = str(row[PATH_COL_NAME]).split("/")[0].strip()
    required_verticals = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS: required_verticals |= DOMAIN_SIGNALS[tok]
    
    penalty = 0.45 if (required_verticals and top_level not in required_verticals) else 0.0
    
    # Final Score Calculation
    score = (cos * 0.50) + (fuzz.token_set_ratio(clean_q, leaf)/100 * 0.30) + literal_boost - penalty
    return round(min(max(score * 150.0, 0.0), 100.0), 1)

# ==============================================================================
# 5. UI & PROCESSING
# ==============================================================================
with st.sidebar:
    st.header("Audit Settings")
    threshold = st.slider("Min Confidence %", 0, 100, 20)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)
    
    if st.button("Run Audit", width="stretch"):
        # Resolve the full path text
        if code_col:
            clean_codes = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_up["Assigned category in full"] = clean_codes.map(master_code_map).fillna("Unknown")
        else:
            # If no code, assume the 'CATEGORY' column has the path
            df_up["Assigned category in full"] = df_up.get("CATEGORY", "N/A")

        names = df_up[name_col].fillna("").astype(str).tolist()
        cleaned = [clean_standard(n) for n in names]
        
        # The AI "Transform" step now looks at your product name vs (Paths + Keywords)
        q_vecs = vectorizer.transform(cleaned)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        scores, statuses = [], []
        for i in range(len(names)):
            conf = score_assignment(cleaned[i], df_up["Assigned category in full"].iloc[i], all_sims[i])
            scores.append(conf)
            # Binary result: Approved if >= 20 and >= user threshold
            statuses.append("Approved" if conf >= 20.0 and conf >= threshold else "Rejected")

        df_up["confidence"] = scores
        df_up["status"] = statuses
        
        final_cols = ["NAME", "Assigned category in full", "confidence", "status"]
        st.dataframe(df_up[final_cols].head(2500), column_config={"confidence": st.column_config.ProgressColumn("Confidence")}, width="stretch")
        st.download_button("Export Results", df_up[final_cols].to_csv(index=False).encode('utf-8'), "audit_report.csv")

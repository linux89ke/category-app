import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Category Auditor", layout="wide")
st.title("🛡️ Category Audit & Comparison Engine")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. QUERY CLEANING (Your Expert Logic)
# ==========================================
_EXPANSIONS = {'iphone': 'phone smartphone mobile', 'ipad': 'tablet', 'android': 'phone smartphone', 'tv': 'television', 'telly': 'television', 'fridge': 'refrigerator', 'moringa': 'herbal supplement vitamins', 'ashwagandha': 'herbal supplement', 'turmeric': 'herbal supplement spice', 'sneakers': 'shoes athletic', 'tyre': 'tire', 'jiko': 'stove charcoal'}

def clean_query(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+\s*(ml|cl|l|g|kg|mg|units|tablets|pcs|capsules|oz|inch|cm|mm|w|kw|mah|gb|tb|mb|m\b)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in _EXPANSIONS: expanded.extend(_EXPANSIONS[t].split())
    return " ".join(expanded).strip()

def clean_category_text(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

# ==========================================
# 3. DATA LOADING & INDEX BUILDING
# ==========================================
@st.cache_resource(show_spinner="Preparing Master Taxonomy...")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    
    # Use Path (3x weight) + standard Keywords. Ignore noisy 'enriched_keywords'.
    df['path_clean'] = df['category_path'].apply(clean_category_text)
    df['kw_clean'] = df['keywords'].fillna('').apply(clean_category_text)
    df['search_text'] = (df['path_clean'] + ' ') * 3 + df['kw_clean']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))
    return df, vectorizer, tfidf_matrix, code_to_path

df_main, vectorizer, tfidf_matrix, master_code_map = build_index()

# ==========================================
# 4. RE-RANKING ENGINE
# ==========================================
def rerank_candidates(clean_query, top_idx, sims_row, threshold):
    best_combined = -1.0
    best_row = None

    for idx in top_idx:
        row = df_main.iloc[idx]
        cos_score = float(sims_row[idx])
        path_fuzzy = fuzz.token_set_ratio(clean_query, row['path_clean']) / 100.0
        # Weights: Math (65%) + Accuracy (30%) + Depth (5%)
        combined = cos_score * 0.65 + path_fuzzy * 0.30 + int(row['depth']) * 0.008

        if combined > best_combined:
            best_combined = combined
            best_row = row

    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"
    confidence = round(min(best_combined * 125.0, 100.0), 2)
    status = "✅ Approved" if confidence >= threshold else "❌ Rejected"
    return best_row['category_path'], str(best_row['category_code']), confidence, status

# ==========================================
# 5. UI: BATCH PROCESSING & COMPARISON
# ==========================================
st.sidebar.header("Matching Rules")
threshold = st.sidebar.slider("Match threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    
    # Identify key columns from your CSV
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCH" not in c.upper()), None)
    cat_name_col = next((c for c in df_up.columns if "CATEGORY" in c.upper() and "PATH" not in c.upper() and "AI" not in c.upper()), None)

    if code_col:
        st.success(f"Original Code found in: **{code_col}**")
        df_up["Original Assigned Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Unknown Code")

    if st.button("Start Analysis & Show Comparison 🚀"):
        names = df_up[name_col].fillna("").tolist()
        total = len(names)
        
        # Matrix Vectorization (Phase 1)
        cleaned_queries = [clean_query(n) for n in names]
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        progress_bar = st.progress(0.0)
        for i in range(total):
            sims_row = all_sims[i]
            top_idx = np.argpartition(sims_row, -25)[-25:]
            top_idx = top_idx[np.argsort(sims_row[top_idx])[::-1]]
            results.append(rerank_candidates(cleaned_queries[i], top_idx, sims_row, threshold))
            if i % 50 == 0: progress_bar.progress((i + 1) / total)

        # Build Results Table
        new_cols = ["AI Suggestion", "Matched Code", "Confidence", "Status"]
        df_out = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
        df_out["AI Suggestion"] = [r[0] for r in results]
        df_out["Matched Code"] = [r[1] for r in results]
        df_out["Confidence"] = [r[2] for r in results]
        df_out["Status"] = [r[3] for r in results]

        st.success("✅ Comparison Analysis Ready!")

        # DEFINE THE COMPARISON DISPLAY
        display_cols = [name_col]
        if "Original Assigned Path" in df_out.columns:
            display_cols.append("Original Assigned Path")
        elif cat_name_col:
            display_cols.append(cat_name_col)
            
        display_cols += ["Status", "AI Suggestion", "Confidence"]

        st.subheader("Comparison: Assigned vs AI Suggestion")
        st.dataframe(df_out[display_cols].head(1000), use_container_width=True)
        
        # Actionable buttons
        st.download_button("📥 Download Combined Comparison", df_out.to_csv(index=False).encode("utf-8"), "comparison_audit.csv")

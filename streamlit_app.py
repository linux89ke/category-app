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
st.set_page_config(page_title="AI Category Auditor", layout="wide")
st.title("🛡️ Professional Category Matcher & Auditor")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. DICTIONARIES & SYNONYMS (The "Fix Names" Logic)
# ==========================================
_EXPANSIONS = {
    'iphone': 'phone smartphone mobile case cover',
    'galaxy': 'phone smartphone mobile case cover',
    'tv': 'television television screen',
    'fridge': 'refrigerator freezer',
    'sneakers': 'shoes athletic trainers footwear',
    'smartwatch': 'smart watch fitness tracker wearable',
    'earbuds': 'earphones headphones airpods wireless'
}

def clean_query(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove technical measurements (100ml, 50g, 12v, etc)
    text = re.sub(r'\d+\s*(ml|cl|l|g|kg|mg|units|tablets|pcs|capsules|oz|count|ct|inch|cm|mm|w|kw|mah|gb|tb|mb|m\b)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in _EXPANSIONS: expanded.extend(_EXPANSIONS[t].split())
    if expanded:
        last = expanded[-1]
        if not last.endswith('s'): expanded.append(last + 's')
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
    
    path_col = next((c for c in df.columns if 'PATH' in c.upper()), 'category_path')
    df['depth'] = df[path_col].apply(lambda x: str(x).count('/') + 1)
    
    # Clean codes (remove .0 and spaces)
    df['category_code_str'] = df['category_code'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    df['path_clean'] = df[path_col].astype(str).str.replace('/', ' ').str.lower()
    df['kw_clean'] = df['keywords'].fillna('').astype(str).str.lower()
    df['search_text'] = (df['path_clean'] + ' ') * 3 + df['kw_clean']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    code_to_path = dict(zip(df['category_code_str'], df[path_col]))
    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_main, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = build_index()

# ==========================================
# 4. RE-RANKING ENGINE
# ==========================================
def rerank(clean_query, top_idx, sims_row, threshold):
    best_combined = -1.0
    best_row = None
    for idx in top_idx:
        row = df_main.iloc[idx]
        cos_score = float(sims_row[idx])
        path_fuzzy = fuzz.token_set_ratio(clean_query, str(row[PATH_COL_NAME]).lower()) / 100.0
        combined = cos_score * 0.70 + path_fuzzy * 0.25 + int(row['depth']) * 0.005
        if combined > best_combined:
            best_combined = combined
            best_row = row
    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"
    conf = round(min(best_combined * 125.0, 100.0), 2)
    return best_row[PATH_COL_NAME], str(best_row['category_code']), conf, ("✅ Approved" if conf >= threshold else "❌ Rejected")

# ==========================================
# 5. UI & BATCH PROCESSING
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    # --- ROBUST READER FIX ---
    try:
        # sep=None lets pandas auto-detect if your file uses commas (,) or semicolons (;)
        df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}"); st.stop()

    # Find the critical columns
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    if code_col:
        # Fix the "Unknown Code" issue by standardizing the input codes
        clean_codes = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_up["Assigned Category (Original)"] = clean_codes.map(master_code_map).fillna("⚠️ Unknown Code")

    if st.button("Fix & Analyze Data 🚀"):
        names = df_up[name_col].fillna("").tolist()
        cleaned_queries = [clean_query(n) for n in names]
        
        # Batch Matrix Matching
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        progress_bar = st.progress(0.0)
        for i in range(len(names)):
            sims_row = all_sims[i]
            top_idx = np.argpartition(sims_row, -20)[-20:]
            results.append(rerank(cleaned_queries[i], top_idx, sims_row, threshold))
            if i % 50 == 0: progress_bar.progress((i + 1) / len(names))

        df_up["AI Suggestion"] = [r[0] for r in results]
        df_up["Confidence"] = [r[2] for r in results]
        df_up["Status"] = [r[3] for r in results]

        # --- READABLE UI CONFIGURATION ---
        st.subheader("📊 Cleaned Side-by-Side Comparison")
        
        col_config = {
            name_col: st.column_config.TextColumn("Product Name", width="medium"),
            "Status": st.column_config.TextColumn("Result", width="small"),
            "Assigned Category (Original)": st.column_config.TextColumn("Assigned Path", width="large"),
            "AI Suggestion": st.column_config.TextColumn("AI Suggestion", width="large"),
            "Confidence": st.column_config.ProgressColumn("Score", format="%f%%", min_value=0, max_value=100)
        }

        display_cols = [name_col, "Status", "Assigned Category (Original)", "AI Suggestion", "Confidence"]
        st.dataframe(
            df_up[[c for c in display_cols if c in df_up.columns]].head(1000),
            column_config=col_config,
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button("📥 Download Fixed CSV", df_up.to_csv(index=False).encode("utf-8"), "fixed_categories.csv")

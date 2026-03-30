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
st.set_page_config(page_title="Universal Category Matcher", layout="wide")
st.title("🚀 Universal Category Matching Engine")

CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. GENERIC CLEANING (Universal for all Categories)
# ==========================================
_FILLER = {
    'free', 'shipping', 'best', 'price', 'new', 'promo', 'original', 'authentic',
    'pack', 'dozen', 'strong', 'instant', 'buy', 'sale', 'the', 'with', 'and',
    'for', 'set', 'kit', 'compatible', 'high', 'quality', 'premium', 'of', 'in'
}

def clean_universal(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove technical noise (measurements)
    text = re.sub(r'\d+\s*(ml|cl|l|g|kg|mg|units|pcs|inch|cm|mm|w|kw|mah|gb|tb|mb|m|ah|v)\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in _FILLER and len(t) > 1]
    return " ".join(filtered).strip()

# ==========================================
# 3. DATA LOADING & KEYWORD INDEXING
# ==========================================
@st.cache_resource(show_spinner="Building Keyword Index…")
def build_universal_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found. Check your file name."); st.stop()

    # Load the Excel Master File
    df = pd.read_excel(CATEGORY_FILE)
    
    # Clean up column names (lowercase and no spaces)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Detect required columns robustly
    path_col = next((c for c in df.columns if 'PATH' in c.upper()), 'category_path')
    kw_col = next((c for c in df.columns if 'KEY' in c.upper()), 'keywords')
    code_col = next((c for c in df.columns if 'CODE' in c.upper()), 'category_code')

    # Ensure keywords column exists even if empty
    if kw_col not in df.columns:
        df[kw_col] = ""

    # Process Path & Leaf Node (The most specific word)
    df['path_processed'] = df[path_col].astype(str)
    df['depth'] = df['path_processed'].apply(lambda x: str(x).count('/') + 1)
    df['leaf_node'] = df['path_processed'].apply(lambda x: str(x).split('/')[-1].strip().lower())
    
    # --- KEYWORD LOGIC ---
    # We repeat the Path 4x and add the Keywords 1x.
    # This ensures keywords help but don't "take over" the whole match.
    df['path_clean'] = df['path_processed'].str.replace('/', ' ').str.lower()
    df['search_text'] = (df['path_clean'] + ' ') * 4 + df[kw_col].fillna('').astype(str).str.lower()

    # Create the TF-IDF Matrix
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    # Map codes to paths
    df['clean_code'] = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    code_to_path = dict(zip(df['clean_code'], df[path_col]))

    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_main, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = build_universal_index()

# ==========================================
# 4. UNIVERSAL MATCHING ENGINE
# ==========================================
def universal_match_logic(clean_query, top_idx, sims_row, threshold):
    best_score = -1.0
    best_row = None
    query_tokens = set(clean_query.split())

    for idx in top_idx:
        row = df_main.iloc[idx]
        cos_score = float(sims_row[idx])
        
        # 1. Fuzzy overlap check
        path_str = str(row['path_processed']).lower()
        token_score = fuzz.token_set_ratio(clean_query, path_str) / 100.0
        
        # 2. Specific Leaf Boost (e.g. 'Sneakers' or 'Cover')
        leaf_boost = 0.20 if row['leaf_node'] in query_tokens else 0.0
        
        # 3. Depth Bias
        depth_nudge = int(row['depth']) * 0.01

        # Final Weighted Score
        combined = cos_score * 0.50 + token_score * 0.20 + leaf_boost + depth_nudge

        if combined > best_score:
            best_score = combined
            best_row = row

    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"
    
    conf = round(min(best_score * 110.0, 100.0), 2)
    status = "✅ Approved" if conf >= threshold else "❌ Rejected"
    return best_row[PATH_COL_NAME], str(best_row['clean_code']), conf, status

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Sensitivity Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0, 100, 38)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    # Use auto-delimiter detection
    df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
    
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    if code_col:
        clean_codes = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_up["Assigned Category (Original)"] = clean_codes.map(master_code_map).fillna("⚠️ Unknown Code")

    if st.button("Run Global Keyword Match 🚀"):
        names = df_up[name_col].fillna("").tolist()
        cleaned_queries = [clean_universal(n) for n in names]
        
        # High-Speed Matrix Math
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        progress_bar = st.progress(0.0)
        for i in range(len(names)):
            sims_row = all_sims[i]
            top_idx = np.argpartition(sims_row, -30)[-30:]
            results.append(universal_match_logic(cleaned_queries[i], top_idx, sims_row, threshold))
            if i % 50 == 0: progress_bar.progress((i + 1) / len(names))

        df_up["AI Suggestion"] = [r[0] for r in results]
        df_up["Confidence Score"] = [r[2] for r in results]
        df_up["Status"] = [r[3] for r in results]

        # --- UI DISPLAY ---
        st.subheader("📊 Category Audit Results")
        col_config = {
            name_col: st.column_config.TextColumn("Product", width="medium"),
            "Status": st.column_config.TextColumn("Result", width="small"),
            "Assigned Category (Original)": st.column_config.TextColumn("Current Category", width="large"),
            "AI Suggestion": st.column_config.TextColumn("AI Suggestion", width="large"),
            "Confidence Score": st.column_config.ProgressColumn("Match", format="%f%%", min_value=0, max_value=100)
        }

        st.dataframe(
            df_up[[name_col, "Status", "Assigned Category (Original)", "AI Suggestion", "Confidence Score"]].head(1000), 
            column_config=col_config, 
            use_container_width=True, 
            hide_index=True
        )
        
        st.download_button("📥 Download Results", df_up.to_csv(index=False).encode("utf-8"), "audit_results.csv")

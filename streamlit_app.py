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
st.set_page_config(page_title="Professional Category Matcher", layout="wide")
st.title("🛡️ Professional Category Matcher & Auditor")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. DICTIONARIES & NOISE FILTERS
# ==========================================
_BRANDS = {'samsung', 'apple', 'sony', 'lg', 'huawei', 'xiaomi', 'nokia', 'oppo', 'tecno', 'itel', 'infinix', 'toyota', 'honda', 'ford', 'nissan', 'mazda', 'pampers', 'huggies', 'molfix', 'nike', 'adidas', 'puma', 'reebok', 'fila', 'hp', 'dell', 'lenovo', 'acer', 'asus', 'hisense', 'whirlpool', 'nescafe', 'unilever', 'nestle', 'loreal', 'colgate', 'dettol', 'gillette', 'philips', 'intex', 'wink', 'sadoer', 'lattafa', 'paris', 'redmi', 'realme', 'oneplus'}
_TECH_NOISE = {'i3', 'i5', 'i7', 'i9', 'qled', 'oled', 'uhd', 'fhd', 'hd', '4k', '8k', 'gen', 'monocrystalline', 'polycrystalline', 'ultra', 'plus', 'mini', 'lite', 'turbo', 'series', 'edition', 'model', 'version', 'tpu', 'magsafe', 'nfc', 'matte'}
_FILLER = {'free', 'shipping', 'best', 'price', 'new', 'promo', 'original', 'authentic', 'pack', 'dozen', 'strong', 'instant', 'buy', 'sale', 'the', 'with', 'and', 'for', 'set', 'kit', 'compatible'}
_ALL_QUERY_NOISE = _BRANDS | _TECH_NOISE | _FILLER

_EXPANSIONS = {
    'iphone': 'phone smartphone mobile case cover',
    'ipad': 'tablet',
    'android': 'phone smartphone',
    'galaxy': 'phone smartphone mobile',
    'tv': 'television television screen',
    'fridge': 'refrigerator freezer',
    'sneakers': 'shoes athletic trainers footwear',
    'smartwatch': 'smart watch fitness tracker wearable',
    'earbuds': 'earphones headphones airpods wireless'
}

def clean_query(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+\s*(ml|cl|l|g|kg|mg|units|tablets|pcs|capsules|oz|count|ct|inch|cm|mm|w|kw|mah|gb|tb|mb|m\b)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if t not in _ALL_QUERY_NOISE and len(t) > 1]
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in _EXPANSIONS: expanded.extend(_EXPANSIONS[t].split())
    if expanded:
        last = expanded[-1]
        if not last.endswith('s'): expanded.append(last + 's')
    return " ".join(expanded).strip() or re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

# ==========================================
# 3. DATA LOADING & INDEX BUILDING
# ==========================================
@st.cache_resource(show_spinner="Building High-Speed Index…")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    path_col = next((c for c in df.columns if 'PATH' in c.upper()), 'category_path')
    df['depth'] = df[path_col].apply(lambda x: str(x).count('/') + 1)

    df['path_clean'] = df[path_col].astype(str).str.replace('/', ' ').str.lower()
    df['kw_clean'] = df['keywords'].fillna('').astype(str).str.lower()
    df['search_text'] = (df['path_clean'] + ' ') * 3 + df['kw_clean']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    # --- FIX 1: Robust Code Mapping ---
    # Force codes to be stripped strings (removes .0 and whitespace)
    df['category_code_str'] = df['category_code'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    code_to_path = dict(zip(df['category_code_str'], df[path_col]))
    
    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_main, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = build_index()

# ==========================================
# 4. MATRIX MATCHING & RE-RANKING
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
    status = "✅ Approved" if conf >= threshold else "❌ Rejected"
    return best_row[PATH_COL_NAME], str(best_row['category_code']), conf, status

# ==========================================
# 5. UI & BATCH PROCESSING
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    # Identify the Code column in the uploaded CSV
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCH" not in c.upper()), None)

    if code_col:
        # --- FIX 2: Standardize Uploaded Codes ---
        # Normalize the uploaded code column to match our master dictionary
        df_up_clean_codes = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_up["Assigned Path (Original)"] = df_up_clean_codes.map(master_code_map).fillna("⚠️ Unknown Code")

    if st.button("Start Analysis 🚀"):
        names = df_up[name_col].fillna("").tolist()
        cleaned_queries = [clean_query(n) for n in names]
        
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        progress_bar = st.progress(0.0)
        for i in range(len(names)):
            sims_row = all_sims[i]
            top_idx = np.argpartition(sims_row, -25)[-25:]
            results.append(rerank(cleaned_queries[i], top_idx, sims_row, threshold))
            if i % 50 == 0: progress_bar.progress((i + 1) / len(names))

        df_up["AI Suggestion"] = [r[0] for r in results]
        df_up["Confidence"] = [r[2] for r in results]
        df_up["Status"] = [r[3] for r in results]

        st.subheader("📊 Comparison Table")
        
        col_config = {
            name_col: st.column_config.TextColumn("Product Name", width="medium"),
            "Status": st.column_config.TextColumn("Result", width="small"),
            "Assigned Path (Original)": st.column_config.TextColumn("Current Category", width="large"),
            "AI Suggestion": st.column_config.TextColumn("AI Suggested Category", width="large"),
            "Confidence": st.column_config.ProgressColumn("Match Score", format="%f%%", min_value=0, max_value=100)
        }

        final_display_cols = [name_col, "Status", "Assigned Path (Original)", "AI Suggestion", "Confidence"]
        st.dataframe(
            df_up[[c for c in final_display_cols if c in df_up.columns]].head(1000),
            column_config=col_config,
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button("📥 Download Results", df_up.to_csv(index=False).encode("utf-8"), "audit_results.csv")

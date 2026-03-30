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
st.set_page_config(page_title="High-Speed Category Matcher", layout="wide")
st.title("⚡ High-Speed Category Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. QUERY CLEANING & EXPANSIONS (Your Logic)
# ==========================================
_BRANDS = {'samsung', 'apple', 'sony', 'lg', 'huawei', 'xiaomi', 'nokia', 'oppo', 'tecno', 'itel', 'infinix', 'toyota', 'honda', 'ford', 'nissan', 'mazda', 'pampers', 'huggies', 'molfix', 'nike', 'adidas', 'puma', 'reebok', 'fila', 'hp', 'dell', 'lenovo', 'acer', 'asus', 'hisense', 'whirlpool', 'nescafe', 'unilever', 'nestle', 'loreal', 'colgate', 'dettol', 'gillette', 'philips'}
_TECH_NOISE = {'i3', 'i5', 'i7', 'i9', 'qled', 'oled', 'uhd', 'fhd', 'hd', '4k', '8k', 'gen', 'monocrystalline', 'polycrystalline', 'pro', 'max', 'ultra', 'plus', 'mini', 'lite', 'turbo', 'series', 'edition', 'model', 'version'}
_FILLER = {'free', 'shipping', 'best', 'price', 'new', 'promo', 'original', 'authentic', 'pack', 'dozen', 'strong', 'instant', 'buy', 'sale', 'the', 'with', 'and', 'for', 'set', 'kit'}
_ALL_QUERY_NOISE = _BRANDS | _TECH_NOISE | _FILLER

_EXPANSIONS = {'iphone': 'phone smartphone mobile', 'ipad': 'tablet', 'android': 'phone smartphone', 'tv': 'television', 'telly': 'television', 'fridge': 'refrigerator', 'moringa': 'herbal supplement vitamins', 'ashwagandha': 'herbal supplement', 'turmeric': 'herbal supplement spice', 'spirulina': 'herbal supplement', 'neem': 'herbal supplement', 'blender': 'blender mixer kitchen', 'earbuds': 'earphones headphones', 'airpods': 'earphones headphones wireless', 'sneakers': 'shoes athletic', 'trainers': 'shoes athletic', 'tyre': 'tire', 'tyres': 'tires', 'jerrycan': 'container storage', 'sufuria': 'cookware pot', 'jiko': 'stove charcoal', 'mkeka': 'mat floor'}

def clean_query(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+\s*(ml|cl|l|g|kg|mg|units|tablets|pcs|capsules|oz|count|ct|inch|cm|mm|w|kw|mah|gb|tb|mb|m\b)', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if t not in _ALL_QUERY_NOISE and len(t) > 1]
    
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in _EXPANSIONS: expanded.extend(_EXPANSIONS[t].split())

    if expanded:
        last = expanded[-1]
        if not last.endswith('s'): expanded.append(last + 's')
        elif len(last) > 4: expanded.append(last[:-1])

    result = " ".join(expanded).strip()
    return result if result else re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

def clean_category_text(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

# ==========================================
# 3. DATA LOADING & INDEX BUILDING
# ==========================================
@st.cache_resource(show_spinner="Building search index…")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)

    # Use Path (3x weight) and Keywords ONLY. Ignore enriched_keywords noise.
    df['path_clean'] = df['category_path'].apply(clean_category_text)
    df['kw_clean'] = df['keywords'].fillna('').apply(clean_category_text)
    df['search_text'] = (df['path_clean'] + ' ') * 3 + df['kw_clean']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=80000, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))
    return df, vectorizer, tfidf_matrix, code_to_path

df_main, vectorizer, tfidf_matrix, master_code_map = build_index()

# ==========================================
# 4. RE-RANKING ENGINE (Section 4)
# ==========================================
def rerank_candidates(clean_query, top_idx, sims_row, threshold):
    best_combined = -1.0
    best_row = None

    for idx in top_idx:
        row = df_main.iloc[idx]
        cos_score = float(sims_row[idx])
        path_fuzzy = fuzz.token_set_ratio(clean_query, row['path_clean']) / 100.0
        
        # Scoring logic: Cosine similarity + Fuzzy Precision + Depth Bonus
        combined = cos_score * 0.65 + path_fuzzy * 0.30 + int(row['depth']) * 0.008

        if combined > best_combined:
            best_combined = combined
            best_row = row

    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"
    
    confidence = round(min(best_combined * 125.0, 100.0), 2)
    status = "✅ Approved" if confidence >= threshold else "❌ Rejected"
    return best_row['category_path'], str(best_row['category_code']), confidence, status

# ==========================================
# 5. UI: BATCH PROCESSING (The Speed Fix)
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
        if len(df_up.columns) == 1:
            uploaded_file.seek(0)
            df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}"); st.stop()

    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCH" not in c.upper()), None)

    if code_col:
        df_up["Assigned Full Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Not in taxonomy")

    if st.button("Start Matrix-Optimized Analysis 🚀"):
        # Load Overrides
        conn = sqlite3.connect(DB_PATH)
        try:
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
        except: learned_dict = {}
        conn.close()

        names = df_up[name_col].fillna("").tolist()
        total = len(names)
        
        st.write("### ⚙️ Phase 1: Cleaning & Vectorizing...")
        cleaned_queries = [clean_query(n) for n in names]
        
        # MATRIX SPEED FIX: Transform all items at once
        q_vecs = vectorizer.transform(cleaned_queries)
        
        st.write("### ⚙️ Phase 2: Matrix Cosine Similarity...")
        # Calculate all similarities in one giant matrix operation
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        st.write("### ⚙️ Phase 3: Fuzzy Re-Ranking...")
        results = []
        progress_bar = st.progress(0.0)
        
        for i in range(total):
            orig_q = names[i].lower().strip()
            
            # 1. Manual Override Check
            if orig_q in learned_dict:
                results.append((*learned_dict[orig_q], 100.0, "✅ Approved"))
            else:
                # 2. Get top 25 candidates from pre-calculated sims matrix
                sims_row = all_sims[i]
                top_idx = np.argpartition(sims_row, -25)[-25:]
                top_idx = top_idx[np.argsort(sims_row[top_idx])[::-1]]
                
                # 3. Perform Fuzzy re-rank on just those 25
                results.append(rerank_candidates(cleaned_queries[i], top_idx, sims_row, threshold))

            if i % 50 == 0 or i == total - 1:
                progress_bar.progress((i + 1) / total)

        # Build Output
        new_cols = ["AI Category", "Matched Code", "Confidence", "Status"]
        df_out = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
        df_out["AI Category"] = [r[0] for r in results]
        df_out["Matched Code"] = [r[1] for r in results]
        df_out["Confidence"] = [r[2] for r in results]
        df_out["Status"] = [r[3] for r in results]

        st.success("✅ Analysis Complete!")
        st.dataframe(df_out[[name_col, "Status", "AI Category", "Confidence"]].head(500), use_container_width=True)
        st.download_button("📥 Download Results", df_out.to_csv(index=False).encode("utf-8"), "matrix_matches.csv")

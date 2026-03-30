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
st.set_page_config(page_title="Category Matcher Pro", layout="wide")
st.title("🛡️ Professional Category Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. DICTIONARIES & NOISE FILTERS (Your Logic)
# ==========================================
_BRANDS = {
    'samsung', 'apple', 'sony', 'lg', 'huawei', 'xiaomi', 'nokia', 'oppo',
    'tecno', 'itel', 'infinix', 'toyota', 'honda', 'ford', 'nissan', 'mazda',
    'pampers', 'huggies', 'molfix', 'nike', 'adidas', 'puma', 'reebok', 'fila',
    'hp', 'dell', 'lenovo', 'acer', 'asus', 'hisense', 'whirlpool', 'nescafe',
    'unilever', 'nestle', 'loreal', 'colgate', 'dettol', 'gillette', 'philips',
    'intex', 'wink', 'sadoer', 'lattafa', 'paris', 'redmi', 'realme', 'oneplus',
}
_TECH_NOISE = {
    'i3', 'i5', 'i7', 'i9', 'qled', 'oled', 'uhd', 'fhd', 'hd', '4k', '8k',
    'gen', 'monocrystalline', 'polycrystalline', 'ultra', 'plus',
    'mini', 'lite', 'turbo', 'series', 'edition', 'model', 'version',
    'tpu', 'magsafe', 'nfc', 'matte', 
}
_FILLER = {
    'free', 'shipping', 'best', 'price', 'new', 'promo', 'original', 'authentic',
    'pack', 'dozen', 'strong', 'instant', 'buy', 'sale', 'the', 'with', 'and',
    'for', 'set', 'kit', 'compatible',
}
_ALL_QUERY_NOISE = _BRANDS | _TECH_NOISE | _FILLER

_EXPANSIONS = {
    'iphone': 'phone smartphone mobile case',
    'ipad': 'tablet',
    'android': 'phone smartphone',
    'galaxy': 'phone smartphone mobile',
    'tv': 'television',
    'telly': 'television',
    'fridge': 'refrigerator',
    'moringa': 'herbal supplement vitamins',
    'ashwagandha': 'herbal supplement',
    'turmeric': 'herbal supplement spice',
    'sneakers': 'shoes athletic',
    'trainers': 'shoes athletic',
    'smartwatch': 'smart watch fitness tracker wearable',
}

_SYNTHETIC_CATEGORY_KW = {
    'Phones & Tablets / Accessories / Cases': 'phone cover protective tpu silicone flip magsafe smartphone mobile case cover',
    'Phones & Tablets / Accessories / Cases / Cases': 'phone cover protective tpu silicone flip magsafe smartphone mobile case cover',
    'Phones & Tablets / Accessories / Screen Protectors': 'phone screen glass tempered protector smartphone mobile privacy film',
    'Phones & Tablets / Accessories / USB Sync & Charging / Chargers & Adapters': 'phone charger adapter usb fast charging watt samsung galaxy wall plug',
    'Electronics / Headphones / Earbud Headphones': 'wireless earbuds airpods earphones bluetooth in ear true wireless',
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
        elif len(last) > 4: expanded.append(last[:-1])
    
    return " ".join(expanded).strip() or re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

def clean_category_text(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

# ==========================================
# 3. DATA LOADING & INDEX BUILDING
# ==========================================
@st.cache_resource(show_spinner="Building High-Precision Index…")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)

    # Building search text with Synthetic injection
    df['path_clean'] = df['category_path'].apply(clean_category_text)
    df['kw_clean'] = df['keywords'].fillna('').apply(clean_category_text)
    df['synthetic'] = df['category_path'].map(_SYNTHETIC_CATEGORY_KW).fillna('')
    df['search_text'] = (df['path_clean'] + ' ') * 3 + df['kw_clean'] + ' ' + df['synthetic']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=80000)
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))
    return df, vectorizer, tfidf_matrix, code_to_path

df_main, vectorizer, tfidf_matrix, master_code_map = build_index()

# ==========================================
# 4. RE-RANKING LOGIC
# ==========================================
def rerank(clean_query, top_idx, sims_row, threshold):
    best_combined = -1.0
    best_row = None
    for idx in top_idx:
        row = df_main.iloc[idx]
        cos_score = float(sims_row[idx])
        path_fuzzy = fuzz.token_set_ratio(clean_query, row['path_clean']) / 100.0
        combined = cos_score * 0.65 + path_fuzzy * 0.30 + int(row['depth']) * 0.008
        if combined > best_combined:
            best_combined = combined
            best_row = row
    
    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"
    conf = round(min(best_combined * 125.0, 100.0), 2)
    return best_row['category_path'], str(best_row['category_code']), conf, ("✅ Approved" if conf >= threshold else "❌ Rejected")

# ==========================================
# 5. UI & BATCH
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCH" not in c.upper()), None)

    if code_col:
        df_up["Assigned Full Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Unknown Code")

    if st.button("Start AI Analysis 🚀"):
        names = df_up[name_col].fillna("").tolist()
        
        # Phase 1: Matrix Transform (Fast)
        cleaned_queries = [clean_query(n) for n in names]
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        # Phase 2: Scoring
        results = []
        progress_bar = st.progress(0.0)
        for i in range(len(names)):
            sims_row = all_sims[i]
            top_idx = np.argpartition(sims_row, -25)[-25:]
            top_idx = top_idx[np.argsort(sims_row[top_idx])[::-1]]
            results.append(rerank(cleaned_queries[i], top_idx, sims_row, threshold))
            if i % 50 == 0: progress_bar.progress((i + 1) / len(names))

        # Build Output
        df_up["AI Category"] = [r[0] for r in results]
        df_up["Confidence"] = [r[2] for r in results]
        df_up["Status"] = [r[3] for r in results]

        st.dataframe(df_up[[name_col, "Status", "AI Category", "Confidence"]].head(1000), use_container_width=True)
        st.download_button("📥 Download Results", df_up.to_csv(index=False).encode("utf-8"), "results.csv")

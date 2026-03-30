import streamlit as st
import pandas as pd
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Precision Category Matcher", layout="wide")
st.title("🛡️ Precision Category Matcher & Auditor")

# Filename for your new CSV
CATEGORY_FILE = "category_map_fully_enriched.xlsx - Sheet1.csv"

# ==========================================
# 2. SYNONYMS & KEYWORD EXPANSIONS
# ==========================================
_EXPANSIONS = {
    # Phone Accessories (The "Phone Cover/Case" Fix)
    'cover': 'case protector shell bumper skin tpu silicone',
    'covers': 'cases protectors shells bumpers skins tpu silicone',
    'iphone': 'phone smartphone mobile case cover apple',
    'galaxy': 'phone smartphone mobile case cover samsung',
    'magsafe': 'magnetic wireless charging case cover',
    
    # Audio
    'airpods': 'earphones headphones wireless earbuds pods',
    'earbuds': 'earphones headphones bluetooth in-ear pods',
    'buds': 'earphones headphones bluetooth earbuds',
    
    # General Tech
    'tv': 'television telly screen smart-tv',
    'fridge': 'refrigerator cooling freezer',
    'sneakers': 'shoes athletic trainers footwear',
    'tyre': 'tire rubber wheel',
    'jiko': 'stove charcoal burner cooker',
    'sufuria': 'pot cookware pan kitchenware',
    'smartwatch': 'watch fitness tracker wearable',
}

# This "Injects" keywords into the Taxonomy at build time
# to help the AI find specific paths even if the master file is sparse.
_SYNTHETIC_INJECTION = {
    'Phones & Tablets / Accessories / Cases': 'phone cover protective tpu silicone flip magsafe back cover skin shell',
    'Phones & Tablets / Accessories / Cases / Cases': 'phone cover protective tpu silicone flip magsafe back cover skin shell',
    'Phones & Tablets / Accessories / Screen Protectors': 'glass tempered privacy film guard screen protector',
    'Electronics / Headphones / Earbud Headphones': 'wireless earbuds airpods pods bluetooth in-ear true wireless headset',
    'Computing / Laptops / Traditional Laptops / Notebooks': 'computer pc macbook laptop notebook laptop',
}

def clean_query(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove technical measurements (100ml, 50g, 12v, etc)
    text = re.sub(r'\d+\s*(ml|cl|l|g|kg|mg|units|pcs|inch|cm|mm|w|kw|mah|gb|tb|m\b)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in _EXPANSIONS:
            expanded.extend(_EXPANSIONS[t].split())
            
    # Add plurals for better recall
    if expanded:
        last = expanded[-1]
        if not last.endswith('s'): expanded.append(last + 's')
    
    return " ".join(expanded).strip()

# ==========================================
# 3. INDEX BUILDING (Using Keywords Column)
# ==========================================
@st.cache_resource(show_spinner="Building Clean Search Index…")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found."); st.stop()

    df = pd.read_csv(CATEGORY_FILE)
    # Map CSV columns to internal names
    df.columns = df.columns.str.lower().str.strip()
    # We expect: category_name, category_code, category path, keywords
    
    path_col = 'category path' if 'category path' in df.columns else 'category_path'
    
    df['depth'] = df[path_col].apply(lambda x: str(x).count('/') + 1)
    
    # BUILD SEARCH TEXT: Path + Base Keywords + Synthetic Synonyms
    df['search_text'] = (
        df[path_col].astype(str) + " " + 
        df['keywords'].fillna("").astype(str) + " " + 
        df[path_col].map(_SYNTHETIC_INJECTION).fillna("")
    ).str.lower()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    code_to_path = dict(zip(df['category_code'].astype(str), df[path_col]))
    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_main, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = build_index()

# ==========================================
# 4. MATRIX MATCHING ENGINE
# ==========================================
def perform_rerank(clean_q, top_idx, sims_row, threshold):
    best_score = -1.0
    best_row = None
    for idx in top_idx:
        row = df_main.iloc[idx]
        cos_score = float(sims_row[idx])
        # Fuzzy check against the path itself
        path_fuzzy = fuzz.token_set_ratio(clean_q, str(row[PATH_COL_NAME]).lower()) / 100.0
        
        # Combined weight (Math + Fuzzy Precision + Depth Nudge)
        combined = cos_score * 0.70 + path_fuzzy * 0.25 + int(row['depth']) * 0.005
        
        if combined > best_score:
            best_score = combined
            best_row = row
            
    if best_row is None: return "Uncategorized", None, 0.0, "❌ Rejected"
    conf = round(min(best_score * 125.0, 100.0), 2)
    return best_row[PATH_COL_NAME], str(best_row['category_code']), conf, ("✅ Approved" if conf >= threshold else "❌ Rejected")

# ==========================================
# 5. UI & READABILITY FIXES
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match threshold", 0, 100, 40)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCH" not in c.upper()), None)

    if code_col:
        df_up["Assigned Path (Original)"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Unknown")

    if st.button("Start High-Precision Analysis 🚀"):
        names = df_up[name_col].fillna("").tolist()
        cleaned_queries = [clean_query(n) for n in names]
        
        # Matrix Step
        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)
        
        results = []
        progress_bar = st.progress(0.0)
        for i in range(len(names)):
            sims_row = all_sims[i]
            top_idx = np.argpartition(sims_row, -20)[-20:]
            results.append(perform_rerank(cleaned_queries[i], top_idx, sims_row, threshold))
            if i % 50 == 0: progress_bar.progress((i + 1) / len(names))

        # Update Dataframe
        df_up["AI Suggestion"] = [r[0] for r in results]
        df_up["Confidence"] = [r[2] for r in results]
        df_up["Status"] = [r[3] for r in results]

        # --- READABILITY CONFIGURATION ---
        st.subheader("📊 Comparison Analysis")
        
        col_config = {
            name_col: st.column_config.TextColumn("Product", width="medium"),
            "Status": st.column_config.TextColumn("Result", width="small"),
            "Assigned Path (Original)": st.column_config.TextColumn("Assigned Category", width="large"),
            "AI Suggestion": st.column_config.TextColumn("AI Suggestion", width="large"),
            "Confidence": st.column_config.ProgressColumn("Score", format="%f%%", min_value=0, max_value=100)
        }

        # Show table
        st.dataframe(
            df_up[[c for c in col_config.keys() if c in df_up.columns]].head(1000),
            column_config=col_config,
            use_container_width=True,
            hide_index=True
        )
        
        st.download_button("📥 Download Results", df_up.to_csv(index=False).encode("utf-8"), "precision_audit.csv")

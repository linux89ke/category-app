import streamlit as st
import pandas as pd
import os
import re
from rapidfuzz import fuzz, process, utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Precision Matcher v3", layout="wide")
st.title("🎯 High-Precision Category Engine")

# Exact filename as requested
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. FILE CHECKER & LOADER
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ File '{CATEGORY_FILE}' not found in the directory.")
        st.stop()
        
    # Read Excel directly
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # CRITICAL: We ignore 'enriched_keywords' because they contain too much noise.
    # We build a 'Clean Search' using only the Path and the base Keywords.
    df['search_text'] = (df['category_path'].astype(str) + " " + df['category_name'].astype(str)).str.lower()
    
    # Extract the last part of the path (e.g., 'Sneakers') for specificity boosting
    df['specific_item'] = df['category_path'].apply(lambda x: str(x).split('/')[-1].lower())
    
    # Master Code Map for display
    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))
    
    return df, code_to_path

df_main, master_code_map = load_data()
SEARCH_SPACE = df_main['search_text'].tolist()

# ==========================================
# 3. NOISE CLEANING (Removes the 'horrible' words)
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove technical measurements
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|v|w|ah|mah|gb|tb)\b', '', text)
    # Remove marketing junk that causes false matches
    junk = r'\b(buy|sale|for|original|authentic|pack|set|kit|tools|best|price|new|promo|shipping|quality|genuine|stylish)\b'
    text = re.sub(junk, '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())

# ==========================================
# 4. PRECISION MATCHING ENGINE
# ==========================================
@lru_cache(maxsize=10000)
def get_match(clean_query):
    # We use Token Set Ratio - it ignores word order and is very strict about keyword matching
    candidates = process.extract(clean_query, SEARCH_SPACE, scorer=fuzz.token_set_ratio, limit=10)
    
    best_final_score = -1
    winner = ("Uncategorized", 0, 0.0)

    for match_str, base_score, idx in candidates:
        row = df_main.iloc[idx]
        final_score = base_score
        
        # SPECIFICITY BOOST: If the actual category name is in the product title, 
        # give it a +25 point lead so it beats random keyword matches.
        if row['specific_item'] in clean_query:
            final_score += 25
            
        if final_score > best_final_score:
            best_final_score = final_score
            winner = (row['category_path'], row['category_code'], round(min(base_score, 100), 2))
            
    return winner

def process_row(name, threshold, idx):
    clean_q = clean_product_name(name)
    if not clean_q:
        return (idx, "Uncategorized", None, 0.0, "❌ Rejected")
        
    path, code, score = get_match(clean_q)
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    return (idx, path, code, score, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 45)

uploaded_file = st.file_uploader("Upload Your Product CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    col_name = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCHED" not in c.upper()), None)
    
    if code_col:
        df_up["Assigned Full Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Code Not Found")

    if st.button("Run High-Precision Analysis 🚀"):
        names = df_up[col_name].tolist()
        results = {}
        
        progress_bar = st.progress(0.0)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, n, threshold, i): i for i, n in enumerate(names)}
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results[row_idx] = {
                    "Status": status, 
                    "AI Matched Category": path, 
                    "Confidence Score": score
                }
                if i % 50 == 0:
                    progress_bar.progress((i + 1) / len(names))

        res_df = pd.DataFrame.from_dict(results, orient='index')
        df_up_clean = df_up.drop(columns=[c for c in res_df.columns if c in df_up.columns])
        final_df = pd.concat([df_up_clean, res_df], axis=1)

        st.subheader("Results Comparison")
        display_cols = [col_name]
        if "Assigned Full Path" in final_df.columns: display_cols.append("Assigned Full Path")
        display_cols += ["Status", "AI Matched Category", "Confidence Score"]
        
        st.dataframe(final_df[display_cols].head(500), use_container_width=True)
        st.download_button("📥 Download Results", final_df.to_csv(index=False).encode('utf-8'), "final_precision_results.csv")

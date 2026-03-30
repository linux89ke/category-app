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
st.set_page_config(page_title="Clean Signal Matcher", layout="wide")
st.title("🎯 High-Precision Category Engine")

CATEGORY_FILE = "category_map_fully_enriched.xlsx - Sheet1.csv" # Using the uploaded CSV

# ==========================================
# 2. BETTER NOISE CLEANING
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements/quantities
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units|v|w|ah|mah)\b', '', text)
    # Remove symbols but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove high-frequency "junk" words that cause false matches
    junk = r'\b(buy|sale|for|original|authentic|pack|set|kit|tools|best|price|new|promo|shipping|quality)\b'
    text = re.sub(junk, '', text)
    return " ".join(text.split())

# ==========================================
# 3. DATA LOADING (IGNORE ENRICHED JUNK)
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error("Master file missing.")
        st.stop()
        
    df = pd.read_csv(CATEGORY_FILE)
    # Standardize names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # FOCUS: Only use the Category Path and the simple keywords. 
    # IGNORE 'enriched_keywords' as they are causing the errors.
    df['clean_path'] = df['category_path'].fillna("").str.replace("/", " ")
    
    # Extract the "Specific Name" (the last part of the path)
    df['specific_name'] = df['category_path'].apply(lambda x: str(x).split('/')[-1])
    
    # The search string is now ONLY the path. Clean and simple.
    df['search_text'] = (df['category_path'].astype(str) + " " + df['keywords'].fillna("")).str.lower()
    
    return df

df_main = load_data()
SEARCH_SPACE = df_main['search_text'].tolist()

# ==========================================
# 4. PRECISION MATCHING LOGIC
# ==========================================
@lru_cache(maxsize=10000)
def get_match(clean_query):
    # Use Weighted Ratio (better for e-commerce)
    # It prioritizes matching the core nouns over the word order.
    candidates = process.extract(
        clean_query, 
        SEARCH_SPACE, 
        scorer=fuzz.WRatio, 
        limit=5
    )
    
    best_final_score = -1
    winner = ("Uncategorized", 0, 0.0)

    for match_str, base_score, idx in candidates:
        row = df_main.iloc[idx]
        
        # --- SMART BOOSTS ---
        # 1. Specificity Boost: If the specific name (e.g. "Speakers") is in the title, boost it.
        if row['specific_name'].lower() in clean_query:
            base_score += 15
            
        if base_score > best_final_score:
            best_final_score = base_score
            winner = (row['category_path'], row['category_code'], round(min(base_score, 100), 2))
            
    return winner

def process_row(name, threshold, idx):
    clean_q = clean_product_name(name)
    path, code, score = get_match(clean_q)
    
    # Strict Threshold: If score is under 45, it's a "Reject"
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    return (idx, path, code, score, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Matching Rules")
# Raised default threshold to 50 for higher quality
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 50)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    col_name = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    
    if st.button("Run Precision Analysis 🚀"):
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

        # Merge and Display
        res_df = pd.DataFrame.from_dict(results, orient='index')
        df_up_clean = df_up.drop(columns=[c for c in res_df.columns if c in df_up.columns])
        final_df = pd.concat([df_up_clean, res_df], axis=1)

        st.dataframe(final_df[[col_name, "Status", "AI Matched Category", "Confidence Score"]].head(200))
        st.download_button("Download", final_df.to_csv(index=False).encode('utf-8'), "clean_results.csv")

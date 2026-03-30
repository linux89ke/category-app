import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from rapidfuzz import fuzz, process, utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Ultra-Fast Deep Matcher", layout="wide")
st.title("⚡ High-Performance Category Engine")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. OPTIMIZED NOISE CLEANING
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements/quantities
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    # Fast regex for special chars
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove marketing fluff
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return utils.default_process(text) # RapidFuzz's built-in optimizer

# ==========================================
# 3. DATA LOADING (Pre-Processed for Speed)
# ==========================================
@st.cache_data
def load_optimized_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ Master file missing.")
        st.stop()
        
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Pre-calculate constants
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    # Store as dictionary for O(1) lookup speed
    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))
    
    return df, code_to_path

df_main, master_code_map = load_optimized_data()
# Pre-extract list to avoid dataframe overhead inside the loop
CATEGORY_LIST = df_main['search_text'].tolist()

# ==========================================
# 4. TWO-PASS SEARCH ENGINE (Speed + Accuracy)
# ==========================================
@lru_cache(maxsize=10000)
def get_best_match(clean_query):
    # PASS 1: Quick Filter (Narrow 10,000 down to 50 using fast Simple Ratio)
    # This is the "Speed Secret"
    quick_candidates = process.extract(
        clean_query, 
        CATEGORY_LIST, 
        scorer=fuzz.ratio, 
        limit=50
    )
    
    # PASS 2: Deep Match (Only check the 50 candidates using the complex scorer)
    candidate_indices = [c[2] for c in quick_candidates]
    refined_search_space = [CATEGORY_LIST[i] for i in candidate_indices]
    
    final_candidates = process.extract(
        clean_query, 
        refined_search_space, 
        scorer=fuzz.token_set_ratio, 
        limit=5
    )
    
    best_final_score = -1
    best_match_data = ("Uncategorized", None, 0.0)

    for match_str, base_score, local_idx in final_candidates:
        # Map local_idx back to the original global index
        global_idx = candidate_indices[local_idx]
        row = df_main.iloc[global_idx]
        
        # Depth Bonus logic
        depth_bonus = row['depth'] * 2.0 
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            best_match_data = (row['category_path'], row['category_code'], round(min(base_score, 100), 2))
            
    return best_match_data

def process_row(name, threshold, learned_dict, idx):
    q = str(name).lower().strip()
    # 1. DB Check
    if learned_dict and q in learned_dict:
        return (idx, learned_dict[q][0], learned_dict[q][1], 100.0, "✅ Approved")
    
    # 2. Math Check
    clean_q = clean_product_name(name)
    path, code, score = get_best_match(clean_q)
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    return (idx, path, code, score, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file)
    col_name = "NAME" if "NAME" in df_up.columns else df_up.columns[0]
    
    if st.button("Run High-Speed Analysis 🚀"):
        # Load memory
        conn = sqlite3.connect(DB_PATH)
        try:
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
        except: learned_dict = {}
        conn.close()

        names = df_up[col_name].tolist()
        total = len(names)
        results = {}
        
        # UI Setup
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # Parallel Execution (Unconstrained)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, name, threshold, learned_dict, i): i for i, name in enumerate(names)}
            
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results[row_idx] = {"Status": status, "AI Matched Category": path, "Matched Code": code, "Confidence Score": score}
                
                # Update UI every 50 items (Reduced lag)
                if i % 50 == 0 or i == total - 1:
                    pct = (i + 1) / total
                    progress_bar.progress(pct)
                    status_text.text(f"Processed {i+1}/{total} products...")

        # Build Output
        res_df = pd.DataFrame.from_dict(results, orient='index')
        # Clean columns to avoid "Duplicate Column" error
        df_up = df_up.drop(columns=[c for c in res_df.columns if c in df_up.columns])
        final_df = pd.concat([df_up, res_df], axis=1)

        st.dataframe(final_df.head(500))
        st.download_button("Download", final_df.to_csv(index=False).encode('utf-8'), "results.csv")

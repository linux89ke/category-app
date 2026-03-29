import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Pro Category Matcher", layout="wide")
st.title("🧠 Deep-Path Category Engine")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. NOISE CLEANING (Feature 6)
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements and quantity noise
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    # Remove symbols and promo fluff
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return " ".join(text.split())

# ==========================================
# 3. DATA LOADING WITH DEPTH CALCULATION
# ==========================================
@st.cache_data
def load_hierarchical_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found.")
        st.stop()
        
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Calculate Depth: Automobile / Car Care / Wax = 3
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    
    # Identify Top Level Department
    df['tlc'] = df['category_path'].apply(lambda x: str(x).split('/')[0].strip())
    
    # Merge for search
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    tlc_list = df['tlc'].unique().tolist()
    return df, tlc_list

df_main, tlc_list = load_hierarchical_data()

# ==========================================
# 4. CORE ENGINE WITH DEPTH BONUS
# ==========================================
def match_single_item(product_name, threshold, learned_dict, row_index):
    original_query = str(product_name).lower().strip()
    clean_query = clean_product_name(product_name)
    
    # 1. Check Learning Database
    if learned_dict and original_query in learned_dict:
        return (row_index, learned_dict[original_query][0], learned_dict[original_query][1], 100.0, "✅ Approved")

    # 2. Step 1: Find Department (TLC)
    best_tlc = process.extractOne(clean_query, tlc_list, scorer=fuzz.token_set_ratio)
    winning_tlc = best_tlc[0]
    
    # 3. Step 2: Deep Match with Depth Bonus
    sub_df = df_main[df_main['tlc'] == winning_tlc].copy()
    
    # Get top 3 candidates instead of 1 to compare depth
    candidates = process.extract(clean_query, sub_df['search_text'].tolist(), scorer=fuzz.token_set_ratio, limit=5)
    
    best_final_score = -1
    best_row = None

    for match_str, base_score, local_idx in candidates:
        candidate_row = sub_df.iloc[local_idx]
        
        # --- DEPTH BONUS LOGIC ---
        # Add 1.5 points for every level of depth to break ties and favor specific categories
        depth_bonus = candidate_row['depth'] * 1.5
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            # We cap the score at 100 for display, but keep the raw score for status
            display_score = round(min(base_score, 100), 2)
            best_row = (candidate_row['category_path'], candidate_row['category_code'], display_score)

    status = "✅ Approved" if best_row[2] >= threshold else "❌ Rejected"
    
    return (row_index, *best_row, status)

# ==========================================
# 5. UI: BATCH MATCHING
# ==========================================
st.sidebar.header("Matching Rules")
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    
    # Auto-find NAME and any existing Category column
    col_name = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    orig_cat_col = next((c for c in df_up.columns if "CAT" in c.upper() and "MATCHED" not in c.upper()), None)
    
    st.info(f"Analyzing product names from column: **{col_name}**")
    if orig_cat_col:
        st.success(f"Detected assigned category column: **{orig_cat_col}**")

    if st.button("Start AI Matching 🚀"):
        conn = sqlite3.connect(DB_PATH); l_df = pd.read_sql("SELECT * FROM feedback", conn); conn.close()
        learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}

        names = df_up[col_name].tolist()
        results_map = {}
        
        progress_bar = st.progress(0); status_text = st.empty()

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(match_single_item, name, threshold, learned_dict, i): i for i, name in enumerate(names)}
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results_map[row_idx] = {
                    "Status": status,
                    "Matched Category": path,
                    "Matched Code": code,
                    "Confidence Score": score
                }
                if i % 10 == 0:
                    progress_bar.progress((i + 1) / len(names))
                    status_text.text(f"Processed {i+1} / {len(names)} items...")

        # Merge results back into the original dataframe
        results_df = pd.DataFrame.from_dict(results_map, orient='index')
        final_df = pd.concat([df_up, results_df], axis=1)
        
        st.success("Matching Complete!")
        
        # DISPLAY COMPARISON
        display_cols = [col_name]
        if orig_cat_col: display_cols.append(orig_cat_col)
        display_cols += ["Status", "Matched Category", "Confidence Score"]
        
        st.subheader("Comparison: Assigned vs Matched")
        st.dataframe(final_df[display_cols].head(200), use_container_width=True)
        
        st.download_button("📥 Download Combined File", final_df.to_csv(index=False).encode('utf-8'), "comparison_results.csv")

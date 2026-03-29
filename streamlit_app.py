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
st.set_page_config(page_title="Deep Match Comparison", layout="wide")
st.title("🔎 Category Comparison & Matching Engine")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. NOISE CLEANING
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements and quantity noise
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return " ".join(text.split())

# ==========================================
# 3. DATA LOADING (Hierarchical + Depth)
# ==========================================
@st.cache_data
def load_hierarchical_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found.")
        st.stop()
        
    # Read the taxonomy (Master categories)
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Calculate path depth (Automobile / Car / Wax = 3)
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    
    # Department Level (Level 1)
    df['tlc'] = df['category_path'].apply(lambda x: str(x).split('/')[0].strip())
    
    # Combine Path + Keywords for the AI to "read"
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    tlc_list = df['tlc'].unique().tolist()
    return df, tlc_list

df_main, tlc_list = load_hierarchical_data()

# ==========================================
# 4. MATCHING ENGINE WITH DEPTH BIAS
# ==========================================
def match_single_item(product_name, threshold, learned_dict, row_index):
    original_query = str(product_name).lower().strip()
    clean_query = clean_product_name(product_name)
    
    # 1. Check Learning Database
    if learned_dict and original_query in learned_dict:
        return (row_index, learned_dict[original_query][0], learned_dict[original_query][1], 100.0, "✅ Approved")

    # 2. Step 1: Identify Department (TLC)
    best_tlc = process.extractOne(clean_query, tlc_list, scorer=fuzz.token_set_ratio)
    winning_tlc = best_tlc[0]
    
    # 3. Step 2: Deep Match with Depth Bonus (favoring Level 4-6)
    sub_df = df_main[df_main['tlc'] == winning_tlc].copy()
    candidates = process.extract(clean_query, sub_df['search_text'].tolist(), scorer=fuzz.token_set_ratio, limit=5)
    
    best_final_score = -1
    best_match_data = ("Uncategorized", None, 0.0)

    for match_str, base_score, local_idx in candidates:
        candidate_row = sub_df.iloc[local_idx]
        
        # Depth Bonus: Encourages the engine to go deeper than Level 1 or 2
        depth_bonus = candidate_row['depth'] * 2.0 
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            display_score = round(min(base_score, 100), 2)
            best_match_data = (candidate_row['category_path'], candidate_row['category_code'], display_score)

    status = "✅ Approved" if best_match_data[2] >= threshold else "❌ Rejected"
    return (row_index, *best_match_data, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match Threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
        if len(df_up.columns) == 1:
            uploaded_file.seek(0)
            df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
    except:
        st.error("Error reading file."); st.stop()

    # AUTO-DETECT COLUMNS
    col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    # Find the original category column (e.g. "Category", "Category Path", "Original Cat")
    orig_cat_col = next((c for c in df_up.columns if any(x in c.upper() for x in ["CAT", "PATH", "ORIGINAL"]) and "MATCHED" not in c.upper()), None)

    st.info(f"Analyzing: **{col_name}**")
    if orig_cat_col:
        st.success(f"Original Category detected in: **{orig_cat_col}**")

    if st.button("Start Comparison Analysis 🚀"):
        # Load Manual Overrides
        conn = sqlite3.connect(DB_PATH)
        try:
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
        except:
            learned_dict = {}
        conn.close()

        names = df_up[col_name].tolist()
        results_map = {}
        progress_bar = st.progress(0); status_text = st.empty()

        # Parallel Execution
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(match_single_item, name, threshold, learned_dict, i): i for i, name in enumerate(names)}
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results_map[row_idx] = {
                    "Status": status,
                    "AI Matched Category": path,
                    "Matched Code": code,
                    "Confidence Score": score
                }
                if i % 10 == 0:
                    progress_bar.progress((i + 1) / len(names))
                    status_text.text(f"Crunching: {i+1}/{len(names)}")

        # Clean Up Duplicate Columns
        new_cols = ["Status", "AI Matched Category", "Matched Code", "Confidence Score"]
        df_up_clean = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
        
        # Merge
        results_df = pd.DataFrame.from_dict(results_map, orient='index')
        final_df = pd.concat([df_up_clean, results_df], axis=1)

        # Create Comparison Table
        display_cols = [col_name]
        if orig_cat_col:
            display_cols.append(orig_cat_col)
        display_cols += ["Status", "AI Matched Category", "Confidence Score"]

        st.subheader("Side-by-Side Comparison")
        st.write("Review how your original category compares to the AI's deep-path suggestion:")
        
        # Display the table (full levels will be visible in the cells)
        st.dataframe(final_df[display_cols].head(500), use_container_width=True)
        
        st.download_button("📥 Download Analysis CSV", final_df.to_csv(index=False).encode('utf-8'), "category_comparison.csv")

# ==========================================
# 6. SIDEBAR STATS
# ==========================================
st.sidebar.divider()
st.sidebar.metric("Taxonomy Depth Levels", df_main['depth'].max())
st.sidebar.write("The engine is currently biased to favor Level 4+ matches.")

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
st.title("🧠 Professional Hierarchical Category Engine")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. NOISE CLEANING (Removes measurements/fluff)
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements (ml, g, kg, tablets, capsules, etc.)
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    # Remove special characters and common marketing noise
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return " ".join(text.split())

# ==========================================
# 3. DATA LOADING (Hierarchical + Depth)
# ==========================================
@st.cache_data
def load_hierarchical_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found in the directory.")
        st.stop()
        
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Calculate Depth for the "Depth Bonus" (favors deeper categories)
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    
    # Identify Top Level Department
    df['tlc'] = df['category_path'].apply(lambda x: str(x).split('/')[0].strip())
    
    # Pre-process search text
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    tlc_list = df['tlc'].unique().tolist()
    return df, tlc_list

df_main, tlc_list = load_hierarchical_data()

# ==========================================
# 4. CORE MATCHING ENGINE (Depth-Aware)
# ==========================================
def match_single_item(product_name, threshold, learned_dict, row_index):
    original_query = str(product_name).lower().strip()
    clean_query = clean_product_name(product_name)
    
    # 1. Check Learning Database Priority
    if learned_dict and original_query in learned_dict:
        return (row_index, learned_dict[original_query][0], learned_dict[original_query][1], 100.0, "✅ Approved")

    # 2. Step 1: Find Department (Hierarchical)
    best_tlc = process.extractOne(clean_query, tlc_list, scorer=fuzz.token_set_ratio)
    winning_tlc = best_tlc[0]
    
    # 3. Step 2: Deep Match within Department with Depth Bonus
    sub_df = df_main[df_main['tlc'] == winning_tlc].copy()
    candidates = process.extract(clean_query, sub_df['search_text'].tolist(), scorer=fuzz.token_set_ratio, limit=5)
    
    best_final_score = -1
    best_match_data = ("Uncategorized", None, 0.0)

    for match_str, base_score, local_idx in candidates:
        candidate_row = sub_df.iloc[local_idx]
        
        # Depth Bonus: +1.5 points per level to favor specific paths
        depth_bonus = candidate_row['depth'] * 1.5
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            display_score = round(min(base_score, 100), 2)
            best_match_data = (candidate_row['category_path'], candidate_row['category_code'], display_score)

    status = "✅ Approved" if best_match_data[2] >= threshold else "❌ Rejected"
    return (row_index, *best_match_data, status)

# ==========================================
# 5. UI & TABS
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 35)

tab1, tab2 = st.tabs(["Single Item Match", "Batch Match (CSV)"])

with tab1:
    st.header("Search One Item")
    p_input = st.text_input("Enter Product Name:")
    if st.button("Match Now"):
        if p_input:
            _, path, code, score, status = match_single_item(p_input, threshold, None, 0)
            if status == "✅ Approved":
                st.success(f"**{status}** | Category: {path} | Score: {score}")
            else:
                st.warning(f"**{status}** | Best Guess: {path} | Score: {score}")

with tab2:
    st.header("Batch Process Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:
        # Load CSV (handles various separators)
        try:
            df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
            if len(df_up.columns) == 1:
                uploaded_file.seek(0)
                df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
        except Exception as e:
            st.error(f"Error loading CSV: {e}"); st.stop()

        # Target column: 'NAME'
        col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
        orig_cat_col = next((c for c in df_up.columns if "CAT" in c.upper() and "MATCHED" not in c.upper()), None)
        
        st.info(f"Using column: **{col_name}**")

        if st.button("Start Parallel Matching 🚀"):
            # Load learning database once
            conn = sqlite3.connect(DB_PATH)
            try:
                l_df = pd.read_sql("SELECT * FROM feedback", conn)
                learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
            except:
                learned_dict = {}
            conn.close()

            names = df_up[col_name].tolist()
            results_map = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Parallel Threading for Speed
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
                    if i % 10 == 0 or i == len(names)-1:
                        progress_bar.progress((i + 1) / len(names))
                        status_text.text(f"Processed {i+1} / {len(names)} items...")

            # --- PREVENT DUPLICATE COLUMN CRASH ---
            new_cols = ["Status", "Matched Category", "Matched Code", "Confidence Score"]
            df_up_clean = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
            
            # Combine Results
            results_df = pd.DataFrame.from_dict(results_map, orient='index')
            final_df = pd.concat([df_up_clean, results_df], axis=1)
            
            st.success("✅ Matching Complete!")
            
            # Show Comparison View
            display_cols = [col_name]
            if orig_cat_col and orig_cat_col not in new_cols:
                display_cols.append(orig_cat_col)
            display_cols += new_cols
            
            st.dataframe(final_df[display_cols].head(250), use_container_width=True)
            
            # Download Combined Results
            st.download_button("📥 Download Combined CSV", final_df.to_csv(index=False).encode('utf-8'), "final_results.csv")

# ==========================================
# 6. SIDEBAR TOOLS
# ==========================================
st.sidebar.divider()
if st.sidebar.button("Reset Cache"):
    st.cache_data.clear()
    st.rerun()

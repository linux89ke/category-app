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
st.set_page_config(page_title="Ultra-Fast Category Matcher", layout="wide")
st.title("⚡ Parallel Hierarchical Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. NOISE CLEANING (Feature 6)
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements and common quantity noise
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    # Remove special characters and common promo fluff
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return " ".join(text.split())

# ==========================================
# 3. HIERARCHICAL DATA LOADING (Feature 3)
# ==========================================
@st.cache_data
def load_hierarchical_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found.")
        st.stop()
        
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Identify Top Level Category (Department)
    df['tlc'] = df['category_path'].apply(lambda x: str(x).split('/')[0].strip())
    # Merge path and keywords for deep search
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    tlc_list = df['tlc'].unique().tolist()
    return df, tlc_list

df_main, tlc_list = load_hierarchical_data()

# ==========================================
# 4. THE CORE ENGINE
# ==========================================
def match_single_item(product_name, threshold, learned_dict):
    original_query = str(product_name).lower().strip()
    clean_query = clean_product_name(product_name)
    
    # 1. Database Priority
    if learned_dict and original_query in learned_dict:
        return {"Product": product_name, "Status": "✅ Approved", "Path": learned_dict[original_query][0], "Code": learned_dict[original_query][1], "Score": 100.0}

    # 2. Hierarchical Step 1: Find Department
    best_tlc = process.extractOne(clean_query, tlc_list, scorer=fuzz.token_set_ratio)
    winning_tlc = best_tlc[0]
    
    # 3. Hierarchical Step 2: Deep Match within Department
    sub_df = df_main[df_main['tlc'] == winning_tlc]
    best_match = process.extractOne(clean_query, sub_df['search_text'].tolist(), scorer=fuzz.token_set_ratio)
    
    score = round(best_match[1], 2)
    # Map index back to original dataframe
    local_idx = sub_df['search_text'].tolist().index(best_match[0])
    result_row = sub_df.iloc[local_idx]
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    
    return {
        "Product": product_name,
        "Status": status,
        "Path": result_row['category_path'],
        "Code": result_row['category_code'],
        "Score": score
    }

# ==========================================
# 5. UI: BATCH MATCHING WITH PROGRESS
# ==========================================
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 40)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
        if len(df_up.columns) == 1:
            uploaded_file.seek(0)
            df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
    except:
        st.error("CSV format not recognized."); st.stop()

    # Automatically identify 'NAME' column
    col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    st.info(f"Targeting Column: **{col_name}**")

    if st.button("Start AI Parallel Processing 🚀"):
        # Load Manual Overrides
        conn = sqlite3.connect(DB_PATH)
        l_df = pd.read_sql("SELECT * FROM feedback", conn)
        conn.close()
        learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}

        names_to_process = df_up[col_name].tolist()
        total_items = len(names_to_process)
        
        # --- UI ELEMENTS FOR PROGRESS ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        # --- PARALLEL EXECUTION WITH PROGRESS UPDATE ---
        with ThreadPoolExecutor() as executor:
            # Map the work
            futures = {executor.submit(match_single_item, name, threshold, learned_dict): i for i, name in enumerate(names_to_process)}
            
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                
                # Update UI every 5 items to keep it smooth but fast
                if i % 5 == 0 or i == total_items - 1:
                    percent_complete = (i + 1) / total_items
                    progress_bar.progress(percent_complete)
                    status_text.text(f"Processing: {i+1} / {total_items} items complete...")

        status_text.success(f"Successfully processed {total_items} items! ✅")
        
        out_df = pd.DataFrame(results)
        
        # Stats summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total_items)
        c2.metric("✅ Approved", len(out_df[out_df["Status"] == "✅ Approved"]))
        c3.metric("❌ Rejected", len(out_df[out_df["Status"] == "❌ Rejected"]))

        st.dataframe(out_df, use_container_width=True)
        st.download_button("📥 Download Final Results", out_df.to_csv(index=False).encode('utf-8'), "final_categorized.csv")

# ==========================================
# 6. SIDEBAR TOOLS
# ==========================================
st.sidebar.divider()
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="High-Perf Category Engine", layout="wide")
st.title("⚡ Parallel Hierarchical Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. NOISE CLEANING (Feature 6)
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove measurements (ml, g, kg, units, tablets, etc.)
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct)\b', '', text)
    # Remove special characters and promo noise
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of)\b', '', text)
    return " ".join(text.split())

# ==========================================
# 3. HIERARCHICAL DATA LOADING (Feature 3)
# ==========================================
@st.cache_data
def load_hierarchical_data():
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Extract Top Level Category (TLC)
    # e.g., "Health & Beauty / Fragrance" -> "Health & Beauty"
    df['tlc'] = df['category_path'].apply(lambda x: str(x).split('/')[0].strip())
    
    # Pre-process search text
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    # Create unique list of TLCs for the first-pass match
    tlc_list = df['tlc'].unique().tolist()
    
    return df, tlc_list

df_main, tlc_list = load_hierarchical_data()

# ==========================================
# 4. HIERARCHICAL MATCHING LOGIC
# ==========================================
def match_item(product_name, threshold=45, learned_dict=None):
    original_query = str(product_name).lower()
    clean_query = clean_product_name(product_name)
    
    # 1. Check Learning Database
    if learned_dict and original_query in learned_dict:
        return (original_query, "✅ Approved", *learned_dict[original_query], 100.0)

    # 2. Step 1: Hierarchical Pass (Find the Department)
    # Match against the short list of top-level categories
    best_tlc = process.extractOne(clean_query, tlc_list, scorer=fuzz.token_set_ratio)
    winning_tlc = best_tlc[0]
    
    # 3. Step 2: Deep Pass (Within that Department only)
    # This ignores all categories in other departments
    sub_df = df_main[df_main['tlc'] == winning_tlc]
    
    best_match = process.extractOne(
        clean_query, 
        sub_df['search_text'].tolist(), 
        scorer=fuzz.token_set_ratio
    )
    
    score = round(best_match[1], 2)
    best_idx = sub_df.index[sub_df['search_text'].tolist().index(best_match[0])]
    result_row = df_main.iloc[best_idx]
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    
    return (
        product_name, 
        status, 
        result_row['category_path'], 
        result_row['category_code'], 
        score
    )

# ==========================================
# 5. PARALLEL BATCH PROCESSING
# ==========================================
def parallel_batch_process(names, threshold, learned_dict):
    # Uses all available CPU threads to process multiple items at once
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: match_item(x, threshold, learned_dict), names))
    return results

# ==========================================
# 6. UI: BATCH MATCHING
# ==========================================
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Threshold", 0, 100, 45)

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
    # Auto-find 'NAME' column
    col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    
    if st.button("Run Parallel Hierarchical Process ⚡"):
        # Load learned data
        conn = sqlite3.connect(DB_PATH)
        l_df = pd.read_sql("SELECT * FROM feedback", conn)
        conn.close()
        learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
        
        with st.spinner(f"Processing {len(df_up)} items using Parallel CPU..."):
            results = parallel_batch_process(df_up[col].tolist(), threshold, learned_dict)
        
        out_df = pd.DataFrame(results, columns=["Product Name", "Status", "Category Path", "Code", "Score"])
        
        st.success("Finished!")
        st.dataframe(out_df)
        st.download_button("📥 Download", out_df.to_csv(index=False).encode('utf-8'), "results.csv")

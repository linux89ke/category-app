import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Ultra-Fast Category Matcher", layout="wide")
st.title("⚡ Global Category Matcher (Speed Optimized)")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. NOISE CLEANING
# ==========================================
def clean_product_name(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return " ".join(text.split())

# ==========================================
# 3. DATA LOADING & MAPPING
# ==========================================
@st.cache_data
def load_hierarchical_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found. Please ensure it is in the same folder.")
        st.stop()
        
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Calculate depth
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    code_to_path_map = dict(zip(df['category_code'].astype(str), df['category_path']))
    return df, code_to_path_map

df_main, master_code_map = load_hierarchical_data()
search_text_list = df_main['search_text'].tolist()

# ==========================================
# 4. HIGH-SPEED CACHED ENGINE
# ==========================================
# The lru_cache forces the AI to "memorize" up to 10,000 product names
@lru_cache(maxsize=10000)
def get_best_fuzzy_match(clean_query):
    # Global deep search across all categories
    candidates = process.extract(clean_query, search_text_list, scorer=fuzz.token_set_ratio, limit=10)
    
    best_final_score = -1
    best_match_data = ("Uncategorized", None, 0.0)

    for match_str, base_score, local_idx in candidates:
        candidate_row = df_main.iloc[local_idx]
        depth_bonus = candidate_row['depth'] * 2.0 
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            display_score = round(min(base_score, 100), 2)
            best_match_data = (candidate_row['category_path'], candidate_row['category_code'], display_score)
            
    return best_match_data

def process_single_item(product_name, threshold, learned_dict, row_index):
    original_query = str(product_name).lower().strip()
    
    # 1. Check Learning Database (Instant)
    if learned_dict and original_query in learned_dict:
        return (row_index, learned_dict[original_query][0], learned_dict[original_query][1], 100.0, "✅ Approved")

    # 2. Check Cache or Calculate Math
    clean_query = clean_product_name(product_name)
    path, code, score = get_best_fuzzy_match(clean_query)
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    return (row_index, path, code, score, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match Threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
        if len(df_up.columns) == 1:
            uploaded_file.seek(0)
            df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}"); st.stop()

    col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCHED" not in c.upper()), None)

    st.info(f"Analyzing Names from: **{col_name}**")
    
    if code_col:
        st.success(f"Original Code detected in: **{code_col}**")
        df_up["Assigned Full Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Code Not Found in Taxonomy")

    if st.button("Start High-Speed Analysis 🚀"):
        conn = sqlite3.connect(DB_PATH)
        try:
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
        except:
            learned_dict = {}
        conn.close()

        names = df_up[col_name].tolist()
        total_items = len(names)
        results_map = {}
        
        st.write("### ⚡ Processing in High-Speed Mode...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # ThreadPoolExecutor Unleashed (No max_workers limit = 100% CPU usage)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_item, name, threshold, learned_dict, i): i for i, name in enumerate(names)}
            
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results_map[row_idx] = {
                    "Status": status,
                    "AI Matched Category": path,
                    "Matched Code": code,
                    "Confidence Score": score
                }
                
                # Update UI every 25 rows to reduce UI lag and boost calculation speed
                if i % 25 == 0 or i == total_items - 1:
                    pct = min((i + 1) / total_items, 1.0)
                    progress_bar.progress(pct)
                    status_text.markdown(f"**Crunching:** {i+1} / {total_items} items... (*{int(pct*100)}%*)")

        status_text.success(f"✅ Successfully matched {total_items} items!")
        
        # Display cache performance
        cache_info = get_best_fuzzy_match.cache_info()
        st.caption(f"🤖 **AI Memory Stats:** Skipped Math {cache_info.hits} times | Calculated Math {cache_info.misses} times")

        # Clean Up Duplicate Columns
        new_cols = ["Status", "AI Matched Category", "Matched Code", "Confidence Score"]
        df_up_clean = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
        
        results_df = pd.DataFrame.from_dict(results_map, orient='index')
        final_df = pd.concat([df_up_clean, results_df], axis=1)

        display_cols = [col_name]
        if code_col: display_cols.append("Assigned Full Path")
        display_cols += ["Status", "AI Matched Category", "Confidence Score"]

        st.subheader("Side-by-Side Comparison")
        st.dataframe(final_df[display_cols].head(500), use_container_width=True)
        st.download_button("📥 Download Analysis CSV", final_df.to_csv(index=False).encode('utf-8'), "category_comparison.csv")

# ==========================================
# 6. SIDEBAR STATS
# ==========================================
st.sidebar.divider()
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    get_best_fuzzy_match.cache_clear()
    st.rerun()

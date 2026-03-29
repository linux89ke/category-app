import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Global Deep Match Engine", layout="wide")
st.title("🔎 Category Matcher (Global Deep Search)")

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
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove marketing fluff
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
    
    # Calculate path depth (e.g., Automobile / Car Care / Wax = 3)
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    
    # Create the text block the AI will search against
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    # Create a Master Dictionary: Category Code -> Full Path
    code_to_path_map = dict(zip(df['category_code'].astype(str), df['category_path']))
    
    return df, code_to_path_map

df_main, master_code_map = load_hierarchical_data()
search_text_list = df_main['search_text'].tolist() # Cache as list for maximum search speed

# ==========================================
# 4. GLOBAL MATCHING ENGINE
# ==========================================
def match_single_item(product_name, threshold, learned_dict, row_index):
    original_query = str(product_name).lower().strip()
    clean_query = clean_product_name(product_name)
    
    # 1. Manual Override Check (Instant 100% Match)
    if learned_dict and original_query in learned_dict:
        return (row_index, learned_dict[original_query][0], learned_dict[original_query][1], 100.0, "✅ Approved")

    # 2. GLOBAL SEARCH: Grab the Top 10 best matches across the ENTIRE catalog
    candidates = process.extract(clean_query, search_text_list, scorer=fuzz.token_set_ratio, limit=10)
    
    best_final_score = -1
    best_match_data = ("Uncategorized", None, 0.0)

    # 3. Apply Depth Bonus to find the best specific category
    for match_str, base_score, local_idx in candidates:
        candidate_row = df_main.iloc[local_idx]
        
        # Add 2 points for every level of depth to encourage specific matching
        depth_bonus = candidate_row['depth'] * 2.0 
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            # Cap the display score at 100
            display_score = round(min(base_score, 100), 2)
            best_match_data = (candidate_row['category_path'], candidate_row['category_code'], display_score)

    status = "✅ Approved" if best_match_data[2] >= threshold else "❌ Rejected"
    return (row_index, *best_match_data, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match Threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    try:
        # Load CSV and handle potential semicolon separation
        df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
        if len(df_up.columns) == 1:
            uploaded_file.seek(0)
            df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}"); st.stop()

    # AUTO-DETECT COLUMNS
    col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCHED" not in c.upper()), None)

    st.info(f"Analyzing Names from: **{col_name}**")
    
    # Map uploaded codes to full paths for comparison
    if code_col:
        st.success(f"Original Code detected in: **{code_col}**")
        df_up["Assigned Full Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Code Not Found in Taxonomy")

    if st.button("Start Global Analysis 🚀"):
        # Load the learning database into memory
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
        
        # --- VISUAL PROGRESS TRACKER ---
        st.write("### ⚙️ Processing...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # ThreadPoolExecutor restricted to 4 workers to keep UI responsive
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(match_single_item, name, threshold, learned_dict, i): i for i, name in enumerate(names)}
            
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results_map[row_idx] = {
                    "Status": status,
                    "AI Matched Category": path,
                    "Matched Code": code,
                    "Confidence Score": score
                }
                
                # Frequent UI updates to force Streamlit to render the bar
                if i % 5 == 0 or i == total_items - 1:
                    pct = min((i + 1) / total_items, 1.0)
                    progress_bar.progress(pct)
                    status_text.markdown(f"**Crunching:** {i+1} / {total_items} items... (*{int(pct*100)}%*)")

        status_text.success(f"✅ Successfully matched {total_items} items!")

        # Clean Up Duplicate Columns from previous runs
        new_cols = ["Status", "AI Matched Category", "Matched Code", "Confidence Score"]
        df_up_clean = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
        
        # Merge Original Data with Results
        results_df = pd.DataFrame.from_dict(results_map, orient='index')
        final_df = pd.concat([df_up_clean, results_df], axis=1)

        # Build Display Columns for the UI Table
        display_cols = [col_name]
        if code_col:
            display_cols.append("Assigned Full Path")
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
    st.rerun()

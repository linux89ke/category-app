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
    # Remove measurements/quantities (e.g., 100ml, 50g)
    text = re.sub(r'\d+\s*(ml|g|kg|units|tablets|pcs|capsules|oz|s|count|ct|units)\b', '', text)
    # Fast cleaning for special chars
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove marketing fluff
    text = re.sub(r'\b(free shipping|best price|new|promo|original|authentic|pack of|dozen|strong|instant)\b', '', text)
    return utils.default_process(text)

# ==========================================
# 3. DATA LOADING (Pre-Processed for Speed)
# ==========================================
@st.cache_data
def load_optimized_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ Master file `{CATEGORY_FILE}` missing.")
        st.stop()
        
    # Read the taxonomy (Master categories)
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Pre-calculate constants for depth-aware matching
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)
    df['search_text'] = (df['category_path'].astype(str) + " " + df['enriched_keywords'].fillna("")).str.lower()
    
    # Store as dictionary for O(1) lookup speed (Code -> Full Path)
    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))
    
    return df, code_to_path

df_main, master_code_map = load_optimized_data()
CATEGORY_LIST = df_main['search_text'].tolist()

# ==========================================
# 4. TWO-PASS SEARCH ENGINE (Speed + Accuracy)
# ==========================================
@lru_cache(maxsize=10000)
def get_best_match(clean_query):
    # PASS 1: Quick Filter (Narrow 10,000 down to 50 using fast Simple Ratio)
    quick_candidates = process.extract(
        clean_query, 
        CATEGORY_LIST, 
        scorer=fuzz.ratio, 
        limit=50
    )
    
    # PASS 2: Deep Match (Check only those 50 using complex token scorer)
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
        global_idx = candidate_indices[local_idx]
        row = df_main.iloc[global_idx]
        
        # Depth Bonus: Favors deep categories over broad top-level ones
        depth_bonus = row['depth'] * 2.0 
        final_score = base_score + depth_bonus
        
        if final_score > best_final_score:
            best_final_score = final_score
            best_match_data = (row['category_path'], row['category_code'], round(min(base_score, 100), 2))
            
    return best_match_data

def process_row(name, threshold, learned_dict, idx):
    q = str(name).lower().strip()
    # Check Manual Overrides
    if learned_dict and q in learned_dict:
        return (idx, learned_dict[q][0], learned_dict[q][1], 100.0, "✅ Approved")
    
    # AI Matching
    clean_q = clean_product_name(name)
    path, code, score = get_best_match(clean_q)
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    return (idx, path, code, score, status)

# ==========================================
# 5. UI: BATCH PROCESSING
# ==========================================
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 35)

uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    # --- ROBUST CSV READING ---
    try:
        # We use engine='python' and sep=None to let pandas auto-detect , or ;
        # on_bad_lines='skip' prevents the crash you saw
        df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
        st.stop()

    # Detect Columns
    col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "MATCHED" not in c.upper()), None)
    
    st.info(f"Targeting Column: **{col_name}**")
    if code_col:
        df_up["Assigned Full Path"] = df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Code Not Found")

    if st.button("Run High-Speed Analysis 🚀"):
        # Load Manual Overrides
        conn = sqlite3.connect(DB_PATH)
        try:
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}
        except: learned_dict = {}
        conn.close()

        names = df_up[col_name].tolist()
        total = len(names)
        results = {}
        
        # UI Progress Setup
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # Parallel Execution (Uses all CPU cores)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, name, threshold, learned_dict, i): i for i, name in enumerate(names)}
            
            for i, future in enumerate(as_completed(futures)):
                row_idx, path, code, score, status = future.result()
                results[row_idx] = {
                    "Status": status, 
                    "AI Matched Category": path, 
                    "Matched Code": code, 
                    "Confidence Score": score
                }
                
                # Update UI every 50 items to reduce browser lag
                if i % 50 == 0 or i == total - 1:
                    pct = min((i + 1) / total, 1.0)
                    progress_bar.progress(pct)
                    status_text.text(f"Processed {i+1}/{total} products...")

        # Build Output DataFrame
        res_df = pd.DataFrame.from_dict(results, orient='index')
        
        # Remove old result columns if they exist to avoid duplicate errors
        cols_to_drop = [c for c in res_df.columns if c in df_up.columns]
        df_up_clean = df_up.drop(columns=cols_to_drop)
        
        final_df = pd.concat([df_up_clean, res_df], axis=1)

        # Show Results
        display_cols = [col_name]
        if "Assigned Full Path" in final_df.columns: display_cols.append("Assigned Full Path")
        display_cols += ["Status", "AI Matched Category", "Confidence Score"]

        st.subheader("Analysis Results")
        st.dataframe(final_df[display_cols].head(500), use_container_width=True)
        
        st.download_button("📥 Download Results", final_df.to_csv(index=False).encode('utf-8'), "categorized_results.csv")

# ==========================================
# 6. SIDEBAR STATS
# ==========================================
st.sidebar.divider()
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    get_best_match.cache_clear()
    st.rerun()

import streamlit as st
import pandas as pd
import sqlite3
import os
from rapidfuzz import fuzz, utils

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="E-commerce Category Matcher", layout="wide")
st.title("🛒 High-Speed Category Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. INITIALIZE DATABASE
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            correct_category TEXT,
            category_code TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 3. CATEGORY MATCHER LOGIC (Optimized)
# ==========================================
class EcommerceCategoryMatcher:
    def __init__(self, taxonomy_df):
        self.df = taxonomy_df
        # Pre-process taxonomy for speed
        self.taxonomy = taxonomy_df["category_path"].tolist()
        self.taxonomy_clean = [str(c).lower().replace(">", " ").replace("&", " ") for c in self.taxonomy]

    def match(self, product_title, min_score=0.25, learned_dict=None):
        if not product_title or not isinstance(product_title, str):
            return "Uncategorized", None, 0.0

        query = str(product_title).lower()
        
        # 1. Instant Learning Check
        if learned_dict and query in learned_dict:
            return learned_dict[query][0], learned_dict[query][1], 1.0

        best_match_path = "Uncategorized"
        highest_score = 0.0
        best_idx = -1

        query_set = set(query.split())

        # 2. Optimized Loop
        for idx, cat_normalized in enumerate(self.taxonomy_clean):
            # Fast Fuzzy Similarity (Ratio)
            similarity = fuzz.ratio(query, cat_normalized) / 100.0
            
            # Word Overlap
            cat_set = set(cat_normalized.split())
            intersection = query_set.intersection(cat_set)
            overlap_score = (len(intersection) / len(cat_set)) if cat_set else 0
            
            # Final Weighted Score
            final_score = (similarity * 0.4) + (overlap_score * 0.6)
            
            if final_score > highest_score:
                highest_score = final_score
                best_idx = idx

        if highest_score < min_score:
            return "Uncategorized", None, round(highest_score, 2)

        best_row = self.df.iloc[best_idx]
        return best_row["category_path"], best_row["category_code"], round(min(highest_score, 1.0), 2)

# ==========================================
# 4. DATA LOADING
# ==========================================
@st.cache_data
def load_taxonomy():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found.")
        st.stop()
    tdf = pd.read_excel(CATEGORY_FILE)
    tdf.columns = tdf.columns.str.lower().str.strip().str.replace(' ', '_')
    if "category_code" not in tdf.columns:
        tdf["category_code"] = tdf.index.astype(str)
    return tdf

taxonomy_df = load_taxonomy()
matcher = EcommerceCategoryMatcher(taxonomy_df)

# ==========================================
# 5. UI TABS
# ==========================================
tab1, tab2 = st.tabs(["Single Match", "Batch Match (CSV)"])

with tab1:
    st.header("Single Product Matcher")
    product_input = st.text_input("Enter Product Title:", placeholder="e.g., iPhone 16 Pro Max")
    if st.button("Match Category"):
        if product_input:
            cat, code, score = matcher.match(product_input)
            st.success(f"**Category:** {cat} | **Score:** {score}")
        else:
            st.warning("Please enter a title.")

with tab2:
    st.header("Batch Matcher")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
            if len(df_upload.columns) == 1:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, sep=",", on_bad_lines='skip')
        except Exception as e:
            st.error(f"Read Error: {e}"); st.stop()

        # --- AUTOMATIC COLUMN DETECTION ---
        auto_col = next((c for c in df_upload.columns if c.upper() in ["NAME", "PRODUCT_NAME", "TITLE", "PRODUCT"]), df_upload.columns[0])
        st.info(f"Using column: **{auto_col}**")

        if st.button("Process Batch"):
            # Load learned data into memory for speed
            conn = sqlite3.connect(DB_PATH)
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            conn.close()
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}

            # Setup Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_cat, results_code, results_score = [], [], []
            total = len(df_upload)
            
            for i, val in enumerate(df_upload[auto_col]):
                cat, code, score = matcher.match(str(val), learned_dict=learned_dict)
                results_cat.append(cat)
                results_code.append(code)
                results_score.append(score)
                
                # Update progress every 20 rows to keep UI snappy
                if i % 20 == 0:
                    progress_bar.progress((i + 1) / total)
                    status_text.text(f"Processing row {i+1} of {total}...")

            df_upload['Matched Category'] = results_cat
            df_upload['Category Code'] = results_code
            df_upload['Confidence Score'] = results_score
            
            progress_bar.progress(1.0)
            status_text.text("Processing Complete! ✅")
            
            st.dataframe(df_upload.head(100))
            st.download_button("📥 Download Results", df_upload.to_csv(index=False).encode('utf-8'), "results.csv")

# ==========================================
# 6. SIDEBAR
# ==========================================
st.sidebar.header("Taxonomy")
st.sidebar.metric("Categories Loaded", len(taxonomy_df))

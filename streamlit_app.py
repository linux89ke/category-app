import streamlit as st
import pandas as pd
import sqlite3
import os
from rapidfuzz import fuzz

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="E-commerce Category Matcher", layout="wide")
st.title("🛒 Smart Category Matcher (AI-Free & Fast)")

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
# 3. CATEGORY MATCHER LOGIC
# ==========================================
class EcommerceCategoryMatcher:
    def __init__(self, taxonomy_df):
        self.df = taxonomy_df
        # Pre-process taxonomy for speed
        self.taxonomy = taxonomy_df["category_path"].tolist()
        self.taxonomy_clean = [str(c).lower().replace(">", " ").replace("&", " ") for c in self.taxonomy]

    def match(self, product_title, threshold=45, learned_dict=None):
        if not product_title or not isinstance(product_title, str):
            return "Uncategorized", None, 0.0, "❌ Rejected"

        query = str(product_title).lower()
        
        # 1. Instant Learning Check (Manual Overrides)
        if learned_dict and query in learned_dict:
            return learned_dict[query][0], learned_dict[query][1], 1.0, "✅ Approved"

        best_match_path = "Uncategorized"
        highest_score = 0.0
        best_idx = -1
        query_set = set(query.split())

        # 2. High-Speed Fuzzy Loop
        for idx, cat_normalized in enumerate(self.taxonomy_clean):
            # RapidFuzz Ratio
            similarity = fuzz.ratio(query, cat_normalized) / 100.0
            
            # Word Overlap
            cat_set = set(cat_normalized.split())
            intersection = query_set.intersection(cat_set)
            overlap_score = (len(intersection) / len(cat_set)) if cat_set else 0
            
            # Weighted Final Score (40% Fuzzy, 60% Overlap)
            final_score = (similarity * 0.4) + (overlap_score * 0.6)
            
            if final_score > highest_score:
                highest_score = final_score
                best_idx = idx

        # Convert to 100-based scale
        final_score_pct = round(min(highest_score, 1.0) * 100, 2)
        
        if final_score_pct < threshold:
            status = "❌ Rejected"
        else:
            status = "✅ Approved"

        if best_idx != -1:
            best_row = self.df.iloc[best_idx]
            return best_row["category_path"], best_row["category_code"], final_score_pct, status
        
        return "Uncategorized", None, final_score_pct, "❌ Rejected"

# ==========================================
# 4. DATA LOADING
# ==========================================
@st.cache_data
def load_taxonomy():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found in the directory.")
        st.stop()
    tdf = pd.read_excel(CATEGORY_FILE)
    tdf.columns = tdf.columns.str.lower().str.strip().str.replace(' ', '_')
    if "category_code" not in tdf.columns:
        tdf["category_code"] = tdf.index.astype(str)
    return tdf

taxonomy_df = load_taxonomy()
matcher = EcommerceCategoryMatcher(taxonomy_df)

# ==========================================
# 5. SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider("Auto-Approval Threshold", 0, 100, 45, help="Scores below this will be marked 'Rejected'")
st.sidebar.divider()
st.sidebar.metric("Categories Loaded", len(taxonomy_df))

# ==========================================
# 6. UI TABS
# ==========================================
tab1, tab2 = st.tabs(["Single Match", "Batch Match (CSV)"])

with tab1:
    st.header("Single Matcher")
    product_input = st.text_input("Enter Product Title:")
    if st.button("Analyze Product"):
        if product_input:
            cat, code, score, status = matcher.match(product_input, threshold=threshold)
            if status == "✅ Approved":
                st.success(f"**Status:** {status} | **Category:** {cat} | **Score:** {score}")
            else:
                st.warning(f"**Status:** {status} | **Best Guess:** {cat} | **Score:** {score}")
        else:
            st.warning("Please enter a title.")

with tab2:
    st.header("Batch Matcher")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file is not None:
        try:
            # Handle different separators
            df_upload = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
            if len(df_upload.columns) == 1:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, sep=",", on_bad_lines='skip')
        except Exception as e:
            st.error(f"File Load Error: {e}"); st.stop()

        # AUTO-DETECT COLUMN
        valid_cols = [c for c in df_upload.columns if any(x in c.upper() for x in ["NAME", "PRODUCT", "TITLE"])]
        auto_col = valid_cols[0] if valid_cols else df_upload.columns[0]
        st.info(f"Automatically using column: **{auto_col}**")

        if st.button("Run Batch Process"):
            # Load learning DB into memory
            conn = sqlite3.connect(DB_PATH)
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            conn.close()
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}

            # Progress Tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_rows = len(df_upload)
            
            for i, val in enumerate(df_upload[auto_col]):
                res = matcher.match(str(val), threshold=threshold, learned_dict=learned_dict)
                results.append({
                    "Product Name": val,
                    "Status": res[3],
                    "Matched Category": res[0],
                    "Category Code": res[1],
                    "Confidence Score": res[2]
                })
                
                # Update progress bar
                if i % 20 == 0:
                    progress_bar.progress((i + 1) / total_rows)
                    status_text.text(f"Matching row {i+1} of {total_rows}...")

            out_df = pd.DataFrame(results)
            progress_bar.progress(1.0)
            status_text.text("Processing Complete! ✅")
            
            # Display stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Items", len(out_df))
            c2.metric("✅ Approved", len(out_df[out_df["Status"] == "✅ Approved"]))
            c3.metric("❌ Rejected", len(out_df[out_df["Status"] == "❌ Rejected"]))

            st.dataframe(out_df, use_container_width=True)
            
            # Download
            st.download_button(
                "📥 Download Categorized CSV", 
                out_df.to_csv(index=False).encode('utf-8'), 
                "categorized_results.csv",
                "text/csv"
            )

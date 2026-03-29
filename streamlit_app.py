import streamlit as st
import pandas as pd
import sqlite3
import os
from rapidfuzz import fuzz

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="E-commerce Category Matcher", layout="wide")
st.title("🛒 Professional Category Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. INITIALIZE DATABASE (For Learning Memory)
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
# 3. HIGH-SPEED MATCHER LOGIC
# ==========================================
class EcommerceCategoryMatcher:
    def __init__(self, taxonomy_df):
        self.df = taxonomy_df
        # Pre-process taxonomy once for maximum speed
        self.taxonomy = taxonomy_df["category_path"].tolist()
        self.taxonomy_clean = [str(c).lower().replace(">", " ").replace("&", " ") for c in self.taxonomy]

    def match(self, product_title, threshold=45, learned_dict=None):
        if not product_title or not isinstance(product_title, str):
            return "Uncategorized", None, 0.0, "❌ Rejected"

        query = str(product_title).lower()
        
        # 1. Instant Learning Check
        if learned_dict and query in learned_dict:
            return learned_dict[query][0], learned_dict[query][1], 100.0, "✅ Approved"

        best_match_path = "Uncategorized"
        highest_score = 0.0
        best_idx = -1
        query_set = set(query.split())

        # 2. Fast Fuzzy + Semantic Overlap Loop
        for idx, cat_normalized in enumerate(self.taxonomy_clean):
            # RapidFuzz similarity ratio (Fast C++ implementation)
            similarity = fuzz.ratio(query, cat_normalized) / 100.0
            
            # Word Overlap logic
            cat_set = set(cat_normalized.split())
            intersection = query_set.intersection(cat_set)
            overlap_score = (len(intersection) / len(cat_set)) if cat_set else 0
            
            # Weighted Final Score (40% Fuzzy, 60% Overlap)
            final_score = (similarity * 0.4) + (overlap_score * 0.6)
            
            if final_score > highest_score:
                highest_score = final_score
                best_idx = idx

        # Convert to 100-scale
        score_pct = round(min(highest_score, 1.0) * 100, 2)
        status = "✅ Approved" if score_pct >= threshold else "❌ Rejected"

        if best_idx != -1:
            best_row = self.df.iloc[best_idx]
            return best_row["category_path"], best_row["category_code"], score_pct, status
        
        return "Uncategorized", None, score_pct, "❌ Rejected"

# ==========================================
# 4. DATA LOADING
# ==========================================
@st.cache_data
def load_taxonomy():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found. Please ensure it is in your app folder.")
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
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Approval Threshold", 0, 100, 45, help="Scores below this are 'Rejected'")
st.sidebar.metric("Categories Loaded", len(taxonomy_df))

# ==========================================
# 6. APP TABS
# ==========================================
tab1, tab2 = st.tabs(["Single Item", "Batch Process (CSV)"])

with tab1:
    st.header("Single Product Match")
    p_input = st.text_input("Product Title:")
    if st.button("Match"):
        if p_input:
            cat, code, score, status = matcher.match(p_input, threshold=threshold)
            if status == "✅ Approved":
                st.success(f"**{status}** | Category: {cat} | Score: {score}")
            else:
                st.warning(f"**{status}** | Best Guess: {cat} | Score: {score}")
            
            with st.expander("Teach the Engine"):
                c_cat = st.text_input("Correct Path")
                c_cod = st.text_input("Correct Code")
                if st.button("Save Override"):
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)", 
                                 (p_input.lower(), c_cat, c_cod))
                    conn.commit()
                    conn.close()
                    st.success("Learned! This will be 100% accurate next time.")

with tab2:
    st.header("Batch Process")
    u_file = st.file_uploader("Upload CSV", type="csv")

    if u_file:
        try:
            df_up = pd.read_csv(u_file, sep=";", on_bad_lines='skip')
            if len(df_up.columns) == 1:
                u_file.seek(0)
                df_up = pd.read_csv(u_file, sep=",", on_bad_lines='skip')
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

        # FIXED COLUMN SELECTION: Prioritize 'NAME'
        col_name = "NAME" if "NAME" in df_up.columns else next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
        st.info(f"Analyzing column: **{col_name}**")

        if st.button("Start Processing ⚡"):
            # Load learning DB once into RAM for speed
            conn = sqlite3.connect(DB_PATH)
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            conn.close()
            learned_dict = {row['query']: (row['correct_category'], row['category_code']) for _, row in l_df.iterrows()}

            # Progress Bar Setup
            progress_bar = st.progress(0)
            status_txt = st.empty()
            
            results = []
            total = len(df_up)
            
            for i, val in enumerate(df_up[col_name]):
                # Call matcher
                res = matcher.match(str(val), threshold=threshold, learned_dict=learned_dict)
                results.append({
                    "Product Name": val,
                    "Status": res[3],
                    "Category Path": res[0],
                    "Category Code": res[1],
                    "Score": res[2]
                })
                
                # Update UI every 25 rows (keeps it fast)
                if i % 25 == 0 or i == total - 1:
                    progress_bar.progress((i + 1) / total)
                    status_txt.text(f"Processing {i+1} of {total} products...")

            out_df = pd.DataFrame(results)
            status_txt.text("Batch Processing Complete! ✅")
            
            # Stats Display
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Items", len(out_df))
            c2.metric("✅ Approved", len(out_df[out_df["Status"] == "✅ Approved"]))
            c3.metric("❌ Rejected", len(out_df[out_df["Status"] == "❌ Rejected"]))

            st.dataframe(out_df, use_container_width=True)
            
            # Download Results
            st.download_button("📥 Download Results", out_df.to_csv(index=False).encode('utf-8'), "categorized_results.csv")

import streamlit as st
import pandas as pd
import difflib
import sqlite3
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="E-commerce Category Matcher", layout="wide")
st.title("🛒 E-commerce Category Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ==========================================
# 2. INITIALIZE DATABASE (For Learning)
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
        # Use your Excel file as the taxonomy source
        self.df = taxonomy_df
        # Create a list of paths for the fuzzy matcher
        self.taxonomy = taxonomy_df["category_path"].tolist()

    def match(self, product_title, min_score=0.25):
        if not product_title or not isinstance(product_title, str):
            return "Uncategorized", None, 0.0

        query = str(product_title).lower()
        
        # --- 1. CHECK LEARNING DATABASE FIRST ---
        conn = sqlite3.connect(DB_PATH)
        learned = pd.read_sql(f"SELECT * FROM feedback WHERE query = ?", conn, params=(query,))
        conn.close()
        
        if not learned.empty:
            row = learned.iloc[-1]
            return row["correct_category"], row["category_code"], 1.0

        # --- 2. FUZZY MATCHING LOGIC ---
        best_match_path = "Uncategorized"
        highest_score = 0.0
        best_idx = -1

        for idx, category in enumerate(self.taxonomy):
            # Clean category string for comparison
            cat_normalized = str(category).lower().replace(">", " ").replace("&", " ")
            
            # Sequence Similarity (Fuzzy)
            similarity = difflib.SequenceMatcher(None, query, cat_normalized).ratio()
            
            # Word Overlap (Semantic)
            query_set = set(query.split())
            cat_set = set(cat_normalized.split())
            intersection = query_set.intersection(cat_set)
            
            overlap_score = (len(intersection) / len(cat_set)) if cat_set else 0
            
            # Weighted Final Score (40% Fuzzy, 60% Overlap)
            final_score = (similarity * 0.4) + (overlap_score * 0.6)
            
            if final_score > highest_score:
                highest_score = final_score
                best_match_path = category
                best_idx = idx

        if highest_score < min_score:
            return "Uncategorized", None, round(highest_score, 2)

        # Get the corresponding code from your original dataframe
        cat_code = self.df.iloc[best_idx]["category_code"]
        
        return best_match_path, cat_code, round(min(highest_score, 1.0), 2)

# ==========================================
# 4. LOAD TAXONOMY DATA
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
    product_input = st.text_input("Enter Product Title:", placeholder="e.g., Samsung Galaxy S24")

    if st.button("Match Category"):
        if product_input:
            category, code, score = matcher.match(product_input)
            if category == "Uncategorized":
                st.warning(f"**Result:** {category} (Score: {score})")
            else:
                st.success(f"**Matched Category:** {category}")
                st.info(f"**Category Code:** {code} | **Confidence:** {score}")
            
            # Feedback Loop
            with st.expander("Is this wrong? Teach the engine"):
                correct_cat = st.text_input("Correct Category Path")
                correct_code = st.text_input("Correct Code")
                if st.button("Save Correction"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)", 
                              (product_input.lower(), correct_cat, correct_code))
                    conn.commit()
                    conn.close()
                    st.success("Corrected! This match will be 100% accurate next time.")
        else:
            st.warning("Please enter a product title.")

with tab2:
    st.header("Batch Matcher")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Load CSV (handling different separators)
        try:
            df_upload = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
            if len(df_upload.columns) == 1:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, sep=",", on_bad_lines='skip')
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        # Identify column
        col_name = st.selectbox("Select the Product Title column:", df_upload.columns)
        
        if st.button("Process Batch"):
            with st.spinner('Categorizing...'):
                # Apply the matcher logic
                results = df_upload[col_name].astype(str).apply(matcher.match)
                
                df_upload['Matched Category'] = [r[0] for r in results]
                df_upload['Category Code'] = [r[1] for r in results]
                df_upload['Confidence Score'] = [r[2] for r in results]
                
                st.write("Results Summary:")
                st.dataframe(df_upload.head(100))
                
                # Download button
                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="categorized_products.csv",
                    mime="text/csv",
                )

# ==========================================
# 6. SIDEBAR
# ==========================================
st.sidebar.header("Taxonomy Stats")
st.sidebar.info(f"Matching against {len(taxonomy_df)} categories from your Excel file.")
with st.sidebar.expander("View Full Taxonomy"):
    st.write(taxonomy_df[["category_path", "category_code"]])

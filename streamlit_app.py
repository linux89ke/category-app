import streamlit as st
import pandas as pd
import sqlite3
import os
from rapidfuzz import fuzz, process

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Category Matching Engine", layout="wide")
st.title("⚡ Category Engine (Fuzzy Logic + Learning)")

DB_PATH = "learning.db"
# Updated to exactly match your Excel file name
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
# 3. LOAD DATA (No AI Embeddings)
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ File not found: `{CATEGORY_FILE}`. Please ensure it is in the same folder as this script.")
        st.write("📂 Files Python can see in this folder:", os.listdir())
        st.stop()
        
    # Switched to read_excel since it is an .xlsx file
    df = pd.read_excel(CATEGORY_FILE)
    
    # Standardize headers to lowercase with underscores
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Ensure required columns exist
    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    keywords = df["enriched_keywords"].fillna("").astype(str) if "enriched_keywords" in df.columns else ""
    
    # Create the text to match against
    df["search_text"] = df["category_path"].astype(str) + " " + keywords
    
    return df

df = load_data()
search_choices = df["search_text"].tolist()
st.success("System Ready ⚡")

# ==========================================
# 4. APP SETTINGS & LEARNING DATA
# ==========================================
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Auto-reject threshold", 0, 100, 65)

def get_learning():
    conn = sqlite3.connect(DB_PATH)
    learning_df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    return learning_df

learn_df = get_learning()

# ==========================================
# 5. CORE MATCHING LOGIC (Fuzzy Only)
# ==========================================
def match_query(query):
    q = str(query).lower()
    
    # 1. Check if we already learned this exact product
    learned = learn_df[learn_df["query"] == q]
    if not learned.empty:
        row = learned.iloc[-1]
        return row["correct_category"], row["category_code"], 100.0

    # 2. RapidFuzz Exact/Fuzzy Search
    result = process.extractOne(q, search_choices, scorer=fuzz.token_set_ratio)
    
    if result:
        best_match_str, score, best_idx = result
        best_row = df.iloc[best_idx]
        
        if score < threshold:
            return "REJECTED", None, score
        else:
            return best_row["category_path"], best_row["category_code"], score
            
    return "REJECTED", None, 0.0

# ==========================================
# 6. HIGH-SPEED BATCH LOGIC
# ==========================================
def batch_match_queries(queries):
    results = []
    
    # Pre-load learned dictionary for instant lookups
    learned_dict = {row["query"]: (row["correct_category"], row["category_code"]) for _, row in learn_df.iterrows()}
    
    for i, q in enumerate(queries):
        if pd.isna(q) or str(q).strip() == "":
            continue
            
        q_lower = str(q).lower()
        
        # Check dictionary first
        if q_lower in learned_dict:
            cat, code = learned_dict[q_lower]
            results.append({"product": q, "category": cat, "category_code": code, "score": 100.0})
        else:
            # Fallback to RapidFuzz
            result = process.extractOne(q_lower, search_choices, scorer=fuzz.token_set_ratio)
            if result:
                _, score, best_idx = result
                best_row = df.iloc[best_idx]
                
                if score < threshold:
                    results.append({"product": q, "category": "REJECTED", "category_code": None, "score": score})
                else:
                    results.append({"product": q, "category": best_row["category_path"], "category_code": best_row["category_code"], "score": score})
            else:
                results.append({"product": q, "category": "REJECTED", "category_code": None, "score": 0.0})
                
    return results

# ==========================================
# 7. UI: SINGLE MATCH
# ==========================================
st.subheader("🔍 Single Match")
query = st.text_input("Enter product name")

if query:
    match, code, score = match_query(query)
    if match == "REJECTED":
        st.warning(f"❌ Rejected (Score: {score:.2f})")
    else:
        st.success(f"✅ {match} (Code: {code}) | Score: {score:.2f}")

    with st.expander("Teach the Engine"):
        correct = st.text_input("Correct category path")
        correct_code = st.text_input("Correct code")
        if st.button("Save Learning") and correct:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)", (query.lower(), correct, correct_code))
            conn.commit()
            conn.close()
            st.success("Learned! ✅")

# ==========================================
# 8. UI: BULK MATCHING
# ==========================================
st.divider()
st.subheader("📂 Bulk Batch Processing")

bulk_file = st.file_uploader("Upload Product CSV", type=["csv"])

if bulk_file:
    try:
        bulk_df = pd.read_csv(bulk_file, sep=";")
        if len(bulk_df.columns) == 1:
            bulk_file.seek(0)
            bulk_df = pd.read_csv(bulk_file, sep=",")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # SMART COLUMN DETECTION - STRICTLY LOOKS FOR 'NAME'
    if "NAME" in bulk_df.columns:
        col = "NAME"
    elif "name" in bulk_df.columns:
        col = "name"
    else:
        # Fallback ignoring IDs and Seller Names
        col = next((c for c in bulk_df.columns if ("name" in c.lower() or "product" in c.lower()) and "id" not in c.lower() and "seller" not in c.lower()), None)

    if col:
        st.info(f"Detected product column: `{col}` ({len(bulk_df)} items)")
        if st.button("Start Processing"):
            
            with st.spinner(f"Fuzzy matching {len(bulk_df)} items... ⚡"):
                queries_list = bulk_df[col].tolist()
                results = batch_match_queries(queries_list)

            st.success("Processing Complete! ✅")

            out_df = pd.DataFrame(results)
            st.dataframe(out_df.head(100))

            # Analytics
            st.subheader("📊 Analytics")
            total = len(out_df)
            rejected = len(out_df[out_df["category"] == "REJECTED"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Processed", total)
            col2.metric("Rejected (Needs Review)", rejected)
            col3.metric("Average Score", round(out_df["score"].mean(), 2))

            csv_data = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Matched Results", csv_data, "categorized_products.csv", "text/csv")
    else:
        st.error("No product name column found.")

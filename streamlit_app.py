import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Category Matching Engine", layout="wide")
st.title("🧠 Ultimate Category Engine (FAISS + Learning + Analytics)")

DB_PATH = "learning.db"
# Pointing directly to your Excel file
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 
PROCESSED_DATA_FILE = "category_map_with_embeddings.pkl"

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
# 3. LOAD AI MODEL (Cached so it only loads once)
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================================
# 4. LOAD DATA & EMBEDDINGS (Optimized for Speed)
# ==========================================
# Check if we already processed and saved the data
if os.path.exists(PROCESSED_DATA_FILE):
    with open(PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
else:
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ File not found: `{CATEGORY_FILE}`. Please make sure the exact file name is in the same folder as this script.")
        # Debugging: Show files in the directory so you know what Python sees
        st.write("Files in current folder:", os.listdir())
        st.stop()
        
    st.info("First run detected! Crunching AI embeddings... this will take a minute but will be instant next time.")
    
    # Load the raw Excel file (requires 'openpyxl' in requirements.txt)
    df = pd.read_excel(CATEGORY_FILE)
    
    # Standardize headers (e.g., "Category Path" -> "category_path")
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Ensure required columns exist
    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    if "enriched_keywords" in df.columns:
        keywords = df["enriched_keywords"].fillna("").astype(str)
    else:
        keywords = ""

    # Create the text for the vector embeddings
    df["category_text"] = df["category_path"].astype(str) + " " + keywords
    
    # Calculate embeddings (This is the slow part we want to save)
    embeddings = model.encode(df["category_text"].tolist(), convert_to_numpy=True)
    df["embedding"] = list(embeddings)
    
    # Save to disk using Pickle for instant loading next time
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(df, f)
        
    st.success("✅ Embeddings saved for future use!")

# ==========================================
# 5. BUILD FAISS INDEX (Takes milliseconds)
# ==========================================
embeddings_matrix = np.vstack(df["embedding"].values).astype('float32')
faiss.normalize_L2(embeddings_matrix)

index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

st.success("System Ready ⚡")

# ==========================================
# 6. APP SETTINGS & LEARNING DATA
# ==========================================
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Suggestions to fetch", 1, 10, 3)
threshold = st.sidebar.slider("Auto-reject threshold", 0, 100, 65)

def get_learning():
    conn = sqlite3.connect(DB_PATH)
    learning_df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    return learning_df

learn_df = get_learning()

# ==========================================
# 7. CORE MATCHING LOGIC
# ==========================================
def match_query(query):
    q = str(query).lower()

    # 1. Check if we already learned this exact product
    learned = learn_df[learn_df["query"] == q]
    if not learned.empty:
        row = learned.iloc[-1]
        return row["correct_category"], row["category_code"], 100.0, pd.DataFrame() # Empty df for candidates

    # 2. Semantic Search (FAISS)
    q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)
    candidates = df.iloc[indices[0]].copy()
    candidates["semantic_score"] = scores[0]

    # 3. Fuzzy Matching (RapidFuzz)
    def fuzzy_score(x):
        return fuzz.token_set_ratio(q, str(x)) if pd.notna(x) else 0

    keyword_col = "enriched_keywords" if "enriched_keywords" in candidates.columns else "category_path"
    candidates["keyword_score"] = candidates[keyword_col].apply(fuzzy_score)

    # 4. Final Score Calculation
    candidates["final_score"] = (candidates["keyword_score"] * 0.5) + (candidates["semantic_score"] * 100 * 0.5)
    best = candidates.iloc[0]

    if best["final_score"] < threshold:
        return "REJECTED", None, best["final_score"], candidates
    else:
        return best["category_path"], best["category_code"], best["final_score"], candidates

# ==========================================
# 8. UI: SINGLE MATCH
# ==========================================
st.subheader("🔍 Single Match")
query = st.text_input("Enter product name (e.g., 'Galaxy Buds 2 Pro')")

if query:
    match, code, score, candidates = match_query(query)

    if match == "REJECTED":
        st.warning(f"❌ Rejected (Score: {score:.2f})")
        if not candidates.empty:
            st.dataframe(candidates[["category_path", "category_code", "final_score"]])
    else:
        st.success(f"✅ {match} (Code: {code}) | Score: {score:.2f}")

    # Manual override / Learning
    with st.expander("Teach the Engine (Override)"):
        correct = st.text_input("Correct category path")
        correct_code = st.text_input("Correct category code")

        if st.button("Save Learning") and correct:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)",
                      (query.lower(), correct, correct_code))
            conn.commit()
            conn.close()
            st.success("Learned! ✅ Next time you search this, it will be 100% accurate.")

# ==========================================
# 9. UI: BULK MATCHING
# ==========================================
st.divider()
st.subheader("📂 Bulk Matching")

bulk_file = st.file_uploader("Upload Product CSV", type=["csv"])

if bulk_file:
    # Try reading with semicolon first, fallback to comma
    try:
        bulk_df = pd.read_csv(bulk_file, sep=";")
        if len(bulk_df.columns) == 1:
            bulk_file.seek(0)
            bulk_df = pd.read_csv(bulk_file, sep=",")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Find the column containing the product names
    col = next((c for c in bulk_df.columns if "name" in c.lower() or "product" in c.lower()), None)

    if col:
        st.info(f"Detected product column: `{col}`")
        if st.button("Start Bulk Process"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_items = len(bulk_df)
            
            for i, q in enumerate(bulk_df[col].astype(str)):
                # Skip empty rows
                if pd.isna(q) or q.strip() == "":
                    continue
                    
                match, code, score, candidates = match_query(q)

                results.append({
                    "product": q,
                    "category": match,
                    "category_code": code,
                    "score": score
                })
                
                # Update progress bar every 10 items or on the last item to save UI rendering time
                if i % 10 == 0 or i == total_items - 1:
                    progress_bar.progress((i + 1) / total_items)
                    status_text.text(f"Processing {i + 1} of {total_items}...")

            status_text.text("Processing Complete! ✅")

            # Show results
            out_df = pd.DataFrame(results)
            st.dataframe(out_df.head(100)) # Show preview

            # Analytics
            st.subheader("📊 Analytics")
            total = len(out_df)
            rejected = len(out_df[out_df["category"] == "REJECTED"])
            avg_score = out_df["score"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Processed", total)
            col2.metric("Rejected (Needs Review)", rejected)
            col3.metric("Average Score", round(avg_score, 2))

            # Download button
            csv_data = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Matched Results", csv_data, "categorized_products.csv", "text/csv")
    else:
        st.error("No product name column found. Found columns: " + ", ".join(bulk_df.columns))

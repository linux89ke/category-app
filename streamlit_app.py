import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import faiss
import os

st.set_page_config(page_title="Category Matching Engine", layout="wide")
st.title("🧠 Ultimate Category Engine (FAISS + Learning + Analytics)")

DB_PATH = "learning.db"
# NOTE: If you are using the CSV version you just uploaded, change this to .csv and use pd.read_csv below
CATEGORY_FILE = "category_map_fully_enriched.xlsx" 

# ----------------------
# INIT DB
# ----------------------
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

# ----------------------
# LOAD MODEL
# ----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------
# LOAD CATEGORY DATA (100% FIXED FOR YOUR EXACT FILE)
# ----------------------
@st.cache_data
def load_data():
    # If using the CSV you just uploaded, change this to pd.read_csv(CATEGORY_FILE)
    df = pd.read_excel(CATEGORY_FILE)
    
    # MAGIC FIX: Make all columns lowercase AND replace spaces with underscores.
    # "Category Path" becomes "category_path"
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Ensure we have the code column
    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    # Safely get keywords
    if "enriched_keywords" in df.columns:
        keywords = df["enriched_keywords"].fillna("").astype(str)
    else:
        keywords = ""

    # Create the text for the vector embeddings
    df["category_text"] = df["category_path"].astype(str) + " " + keywords
    
    return df

if os.path.exists(CATEGORY_FILE):
    df = load_data()

    # Embeddings + FAISS
    if "embedding" not in df.columns:
        embeddings = model.encode(df["category_text"].tolist(), convert_to_numpy=True)
        df["embedding"] = list(embeddings)

    embeddings_matrix = np.vstack(df["embedding"].values).astype('float32')
    faiss.normalize_L2(embeddings_matrix)

    index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    st.success("System Ready ⚡")

    # Settings
    st.sidebar.header("⚙️ Settings")
    top_k = st.sidebar.slider("Suggestions", 1, 10, 3)
    threshold = st.sidebar.slider("Auto-reject threshold", 0, 100, 65)

    # Load learning
    def get_learning():
        conn = sqlite3.connect(DB_PATH)
        learning_df = pd.read_sql("SELECT * FROM feedback", conn)
        conn.close()
        return learning_df

    learn_df = get_learning()

    # ----------------------
    # MATCH FUNCTION
    # ----------------------
    def match_query(query):
        q = str(query).lower()

        # Check if we already learned this exact product
        learned = learn_df[learn_df["query"] == q]
        if not learned.empty:
            row = learned.iloc[-1]
            return row["correct_category"], row["category_code"], 100.0, []

        q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)

        scores, indices = index.search(q_emb, top_k)
        candidates = df.iloc[indices[0]].copy()
        candidates["semantic_score"] = scores[0]

        # Handle missing keywords safely for RapidFuzz
        def fuzzy_score(x):
            return fuzz.token_set_ratio(q, str(x)) if pd.notna(x) else 0

        keyword_col = "enriched_keywords" if "enriched_keywords" in candidates.columns else "category_path"
        candidates["keyword_score"] = candidates[keyword_col].apply(fuzzy_score)

        candidates["final_score"] = (candidates["keyword_score"] * 0.5) + (candidates["semantic_score"] * 100 * 0.5)

        best = candidates.iloc[0]

        if best["final_score"] < threshold:
            return "REJECTED", None, best["final_score"], candidates
        else:
            return best["category_path"], best["category_code"], best["final_score"], candidates

    # ----------------------
    # SINGLE MATCH
    # ----------------------
    st.subheader("🔍 Single Match")
    query = st.text_input("Enter product name")

    if query:
        match, code, score, candidates = match_query(query)

        if match == "REJECTED":
            st.warning(f"❌ Rejected ({score:.2f})")
            st.dataframe(candidates[["category_path", "category_code", "final_score"]])
        else:
            st.success(f"✅ {match} (Code: {code}) | Score: {score:.2f}")

        # Learning
        correct = st.text_input("Correct category")
        correct_code = st.text_input("Correct category code")

        if st.button("Save Learning") and correct:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)",
                      (query.lower(), correct, correct_code))
            conn.commit()
            conn.close()
            st.success("Learned ✅")

    # ----------------------
    # BULK MATCH
    # ----------------------
    st.divider()
    st.subheader("📂 Bulk Matching")

    bulk_file = st.file_uploader("Upload CSV", type=["csv"])

    if bulk_file:
        # Tries semicolon separator first (for the CSV you uploaded earlier), then comma
        try:
            bulk_df = pd.read_csv(bulk_file, sep=";")
            if len(bulk_df.columns) == 1:
                bulk_file.seek(0)
                bulk_df = pd.read_csv(bulk_file, sep=",")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        col = next((c for c in bulk_df.columns if "name" in c.lower() or "product" in c.lower()), None)

        results = []

        if col:
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_items = len(bulk_df)
            
            for i, q in enumerate(bulk_df[col].astype(str)):
                match, code, score, candidates = match_query(q)

                results.append({
                    "product": q,
                    "category": match,
                    "category_code": code,
                    "score": score
                })
                
                # Update progress bar
                progress_bar.progress((i + 1) / total_items)
                status_text.text(f"Processing {i + 1} of {total_items}...")

            status_text.text("Processing Complete! ✅")

            out_df = pd.DataFrame(results)
            st.dataframe(out_df.head(50)) 

            # ----------------------
            # ANALYTICS
            # ----------------------
            st.subheader("📊 Analytics")

            total = len(out_df)
            rejected = len(out_df[out_df["category"] == "REJECTED"])
            avg_score = out_df["score"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products", total)
            col2.metric("Rejected", rejected)
            col3.metric("Avg Score", round(avg_score, 2))

            st.bar_chart(out_df["score"])

            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "final_results.csv")
        else:
            st.error("No product name column found. Found columns: " + ", ".join(bulk_df.columns))

else:
    st.error(f"`{CATEGORY_FILE}` not found in the current folder. Please ensure the file is named exactly like this and is in the same folder as the script.")

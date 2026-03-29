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
# LOAD CATEGORY DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_excel(CATEGORY_FILE)

    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    df["category_text"] = df["category_path"].astype(str) + " " + df.get("enriched_keywords", "")
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
        df = pd.read_sql("SELECT * FROM feedback", conn)
        conn.close()
        return df

    learn_df = get_learning()

    # ----------------------
    # MATCH FUNCTION
    # ----------------------
    def match_query(query):
        q = query.lower()

        learned = learn_df[learn_df["query"] == q]
        if not learned.empty:
            row = learned.iloc[-1]
            return row["correct_category"], row["category_code"], 100, []

        q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)

        scores, indices = index.search(q_emb, top_k)
        candidates = df.iloc[indices[0]].copy()
        candidates["semantic_score"] = scores[0]

        candidates["keyword_score"] = candidates["enriched_keywords"].apply(
            lambda x: fuzz.token_set_ratio(q, str(x))
        )

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
        bulk_df = pd.read_csv(bulk_file)

        col = next((c for c in bulk_df.columns if "name" in c.lower() or "product" in c.lower()), None)

        results = []

        if col:
            for q in bulk_df[col].astype(str):
                match, code, score, candidates = match_query(q)

                results.append({
                    "product": q,
                    "category": match,
                    "category_code": code,
                    "score": score
                })

            out_df = pd.DataFrame(results)
            st.dataframe(out_df)

            # ----------------------
            # ANALYTICS
            # ----------------------
            st.subheader("📊 Analytics")

            total = len(out_df)
            rejected = len(out_df[out_df["category"] == "REJECTED"])
            avg_score = out_df["score"].mean()

            st.metric("Total Products", total)
            st.metric("Rejected", rejected)
            st.metric("Avg Score", round(avg_score, 2))

            st.bar_chart(out_df["score"])

            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "final_results.csv")
        else:
            st.error("No product name column found")

else:
    st.error("category_map_fully_enriched.xlsx not found")

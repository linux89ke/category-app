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
st.title("🧠 Ultimate Category Engine (Batched + FAISS)")

DB_PATH = "learning.db"
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
# 3. LOAD AI MODEL
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================================
# 4. LOAD DATA & EMBEDDINGS 
# ==========================================
if os.path.exists(PROCESSED_DATA_FILE):
    with open(PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
else:
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ File not found: `{CATEGORY_FILE}`.")
        st.stop()
        
    st.info("First run detected! Crunching AI embeddings...")
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    keywords = df["enriched_keywords"].fillna("").astype(str) if "enriched_keywords" in df.columns else ""
    df["category_text"] = df["category_path"].astype(str) + " " + keywords
    
    embeddings = model.encode(df["category_text"].tolist(), convert_to_numpy=True)
    df["embedding"] = list(embeddings)
    
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(df, f)
        
    st.success("✅ Embeddings saved!")

# ==========================================
# 5. BUILD FAISS INDEX
# ==========================================
embeddings_matrix = np.vstack(df["embedding"].values).astype('float32')
faiss.normalize_L2(embeddings_matrix)
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

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
# 7. CORE MATCHING LOGIC (Single)
# ==========================================
def match_query(query):
    q = str(query).lower()
    learned = learn_df[learn_df["query"] == q]
    if not learned.empty:
        row = learned.iloc[-1]
        return row["correct_category"], row["category_code"], 100.0, pd.DataFrame() 

    q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)
    candidates = df.iloc[indices[0]].copy()
    candidates["semantic_score"] = scores[0]

    keyword_col = "enriched_keywords" if "enriched_keywords" in candidates.columns else "category_path"
    candidates["keyword_score"] = candidates[keyword_col].apply(
        lambda x: fuzz.token_set_ratio(q, str(x)) if pd.notna(x) else 0
    )

    candidates["final_score"] = (candidates["keyword_score"] * 0.5) + (candidates["semantic_score"] * 100 * 0.5)
    
    # Sort by the combined final score to ensure the absolute best match is at the top
    candidates = candidates.sort_values(by="final_score", ascending=False)
    best = candidates.iloc[0]

    if best["final_score"] < threshold:
        return "REJECTED", None, best["final_score"], candidates
    else:
        return best["category_path"], best["category_code"], best["final_score"], candidates

# ==========================================
# 8. HIGH-SPEED BATCH LOGIC (For Bulk Uploads)
# ==========================================
def batch_match_queries(queries):
    results = []
    
    # Pre-load learned dictionary for instant O(1) lookups
    learned_dict = {row["query"]: (row["correct_category"], row["category_code"]) for _, row in learn_df.iterrows()}
    
    needs_ai_indices = []
    needs_ai_queries = []
    
    # Filter out empty queries and check if we already learned the answer
    for i, q in enumerate(queries):
        if pd.isna(q) or str(q).strip() == "":
            continue
            
        q_lower = str(q).lower()
        if q_lower in learned_dict:
            cat, code = learned_dict[q_lower]
            results.append({"index": i, "product": q, "category": cat, "category_code": code, "score": 100.0})
        else:
            needs_ai_indices.append(i)
            needs_ai_queries.append(q_lower)
            
    # If we have queries that need the AI, process them ALL AT ONCE
    if needs_ai_queries:
        # 1. Batch Encode (32 items at a time internally)
        q_embs = model.encode(needs_ai_queries, batch_size=32, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_embs)
        
        # 2. Batch Search FAISS
        all_scores, all_indices = index.search(q_embs, top_k)
        
        # 3. Score combinations
        keyword_col = "enriched_keywords" if "enriched_keywords" in df.columns else "category_path"
        
        for idx, q_text, scores, indices_row in zip(needs_ai_indices, needs_ai_queries, all_scores, all_indices):
            candidates = df.iloc[indices_row].copy()
            candidates["semantic_score"] = scores
            candidates["keyword_score"] = candidates[keyword_col].apply(
                lambda x: fuzz.token_set_ratio(q_text, str(x)) if pd.notna(x) else 0
            )
            candidates["final_score"] = (candidates["keyword_score"] * 0.5) + (candidates["semantic_score"] * 100 * 0.5)
            candidates = candidates.sort_values(by="final_score", ascending=False)
            best = candidates.iloc[0]
            
            if best["final_score"] < threshold:
                results.append({"index": idx, "product": queries[idx], "category": "REJECTED", "category_code": None, "score": best["final_score"]})
            else:
                results.append({"index": idx, "product": queries[idx], "category": best["category_path"], "category_code": best["category_code"], "score": best["final_score"]})
                
    # Re-sort results back to original CSV order
    results = sorted(results, key=lambda x: x["index"])
    
    # Clean up the "index" key before returning
    for r in results:
        del r["index"]
        
    return results

# ==========================================
# 9. UI: SINGLE MATCH
# ==========================================
st.subheader("🔍 Single Match")
query = st.text_input("Enter product name")

if query:
    match, code, score, candidates = match_query(query)
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
# 10. UI: BULK MATCHING
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

    col = next((c for c in bulk_df.columns if "name" in c.lower() or "product" in c.lower()), None)

    if col:
        st.info(f"Detected product column: `{col}` ({len(bulk_df)} items)")
        if st.button("Start High-Speed Process"):
            
            with st.spinner(f"Engaging Batch AI Processing for {len(bulk_df)} items... ⚡"):
                # Pass all items directly to the new Batch Matcher
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

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Category Matching Engine", layout="wide")
st.title("🧠 Category Engine (Semantic AI + FAISS)")

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
    # This AI model understands context and meaning, not just spelling!
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================================
# 4. LOAD DATA & BUILD AI INDEX
# ==========================================
# We use Pickle to save the AI's math so it loads instantly after the first run
if os.path.exists(PROCESSED_DATA_FILE):
    with open(PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
else:
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ File not found: `{CATEGORY_FILE}`.")
        st.stop()
        
    st.info("First run detected! Crunching AI embeddings... this will take a minute but will be instant next time.")
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    keywords = df["enriched_keywords"].fillna("").astype(str) if "enriched_keywords" in df.columns else ""
    df["search_text"] = df["category_path"].astype(str) + " " + keywords
    
    # Calculate embeddings
    embeddings = model.encode(df["search_text"].tolist(), convert_to_numpy=True)
    df["embedding"] = list(embeddings)
    
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(df, f)
    st.success("✅ Embeddings saved!")

# Build FAISS Vector Database for instant searching
embeddings_matrix = np.vstack(df["embedding"].values).astype('float32')
faiss.normalize_L2(embeddings_matrix)

index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

st.success("AI Vector Database Ready ⚡")

# ==========================================
# 5. APP SETTINGS & LEARNING DATA
# ==========================================
st.sidebar.header("⚙️ Settings")
# AI scores differently. 40-50 is usually a great threshold for semantic search.
threshold = st.sidebar.slider("Auto-reject threshold", 0, 100, 45)

def get_learning():
    conn = sqlite3.connect(DB_PATH)
    learning_df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    return learning_df

learn_df = get_learning()

# ==========================================
# 6. CORE MATCHING LOGIC (Single)
# ==========================================
def match_query(query):
    q = str(query).lower()
    
    learned = learn_df[learn_df["query"] == q]
    if not learned.empty:
        row = learned.iloc[-1]
        return "✅ Approved", row["correct_category"], row["category_code"], 100.0

    # Semantic Vector Search
    q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, 1) # Get top 1 match
    best_idx = indices[0][0]
    score = scores[0][0] * 100 # Convert cosine similarity to percentage
    
    best_row = df.iloc[best_idx]
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    
    return status, best_row["category_path"], best_row["category_code"], score

# ==========================================
# 7. HIGH-SPEED BATCH LOGIC
# ==========================================
def batch_match_queries(queries):
    results = [{"Product Name": q, "Status": "❌ Rejected", "Full Category Path": None, "Category Code": None, "Score": 0.0} for q in queries]
    
    learned_dict = {row["query"]: (row["correct_category"], row["category_code"]) for _, row in learn_df.iterrows()}
    
    unseen_queries = []
    unseen_indices = []
    
    for i, q in enumerate(queries):
        if pd.isna(q) or str(q).strip() == "":
            results[i]["Status"] = "⚠️ Empty"
            continue
            
        q_lower = str(q).lower()
        if q_lower in learned_dict:
            cat, code = learned_dict[q_lower]
            results[i]["Status"] = "✅ Approved"
            results[i]["Full Category Path"] = cat
            results[i]["Category Code"] = code
            results[i]["Score"] = 100.0
        else:
            unseen_queries.append(q_lower)
            unseen_indices.append(i)
            
    # BATCH AI PROCESSING (Incredibly fast)
    if unseen_queries:
        # Batch encode 64 items at a time
        q_embs = model.encode(unseen_queries, batch_size=64, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_embs)
        
        # Batch search FAISS database
        all_scores, all_indices = index.search(q_embs, 1)
        
        for idx, best_idx_array, score_array in zip(unseen_indices, all_indices, all_scores):
            best_idx = best_idx_array[0]
            score = score_array[0] * 100
            
            best_row = df.iloc[best_idx]
            
            results[idx]["Score"] = round(score, 2)
            results[idx]["Full Category Path"] = best_row["category_path"]
            results[idx]["Category Code"] = best_row["category_code"]
            
            if score < threshold:
                results[idx]["Status"] = "❌ Rejected"
            else:
                results[idx]["Status"] = "✅ Approved"
                
    return results

# ==========================================
# 8. UI: SINGLE MATCH
# ==========================================
st.subheader("🔍 Single Match")
query = st.text_input("Enter product name")

if query:
    status, match, code, score = match_query(query)
    
    if "Rejected" in status:
        st.warning(f"{status} (Score: {score:.2f}) | Best Guess: **{match}** (Code: {code})")
    else:
        st.success(f"{status} | **{match}** (Code: {code}) | Score: {score:.2f}")

    with st.expander("Teach the Engine (Override)"):
        correct = st.text_input("Correct category path")
        correct_code = st.text_input("Correct category code")
        if st.button("Save Learning") and correct:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)", (query.lower(), correct, correct_code))
            conn.commit()
            conn.close()
            st.success("Learned! ✅ Next time you search this, it will be 100% accurate.")

# ==========================================
# 9. UI: BULK MATCHING
# ==========================================
st.divider()
st.subheader("📂 Bulk Batch Processing")

bulk_file = st.file_uploader("Upload Product CSV", type=["csv"])

if bulk_file:
    # Handle files with bad lines/mixed separators
    try:
        bulk_df = pd.read_csv(bulk_file, sep=";", on_bad_lines='skip')
        if len(bulk_df.columns) == 1:
            bulk_file.seek(0)
            bulk_df = pd.read_csv(bulk_file, sep=",", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # SMART COLUMN DETECTION - STRICTLY LOOKS FOR 'NAME'
    if "NAME" in bulk_df.columns:
        col = "NAME"
    elif "name" in bulk_df.columns:
        col = "name"
    else:
        col = next((c for c in bulk_df.columns if ("name" in c.lower() or "product" in c.lower()) and "id" not in c.lower() and "seller" not in c.lower()), None)

    if col:
        st.info(f"Detected product column: `{col}` ({len(bulk_df)} items)")
        if st.button("Start AI Batch Process"):
            
            with st.spinner(f"Running AI Context Engine on {len(bulk_df)} items... 🧠⚡"):
                queries_list = bulk_df[col].tolist()
                results = batch_match_queries(queries_list)

            st.success("Processing Complete! ✅")

            out_df = pd.DataFrame(results)
            
            st.dataframe(
                out_df.head(100),
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        help="AI Confidence Score",
                        format="%f",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )

            st.subheader("📊 Analytics")
            total = len(out_df)
            approved = len(out_df[out_df["Status"] == "✅ Approved"])
            rejected = len(out_df[out_df["Status"] == "❌ Rejected"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Processed", total)
            col2.metric("✅ Approved", approved)
            col3.metric("❌ Rejected", rejected)

            csv_data = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Matched Results", csv_data, "ai_categorized_products.csv", "text/csv")
    else:
        st.error("No product name column found.")

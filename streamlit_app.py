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
st.title("🧠 Semantic AI Category Engine")

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
    # Downloads the 'brain' of the engine
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================================
# 4. LOAD DATA & AI INITIALIZATION
# ==========================================
if "initialized" not in st.session_state:
    st.session_state.initialized = False

df = None

if os.path.exists(PROCESSED_DATA_FILE):
    with open(PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
    st.session_state.initialized = True
else:
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found. Please upload it to your repository.")
        st.stop()
    
    st.warning("⚠️ The AI Engine needs to be initialized (First-time setup).")
    if st.button("🚀 Start AI Initialization (Takes ~2 mins)"):
        with st.spinner("Crunching AI embeddings... please wait."):
            df_raw = pd.read_excel(CATEGORY_FILE)
            df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(' ', '_')

            if "category_code" not in df_raw.columns:
                df_raw["category_code"] = df_raw.index.astype(str)

            keywords = df_raw["enriched_keywords"].fillna("").astype(str) if "enriched_keywords" in df_raw.columns else ""
            df_raw["search_text"] = df_raw["category_path"].astype(str) + " " + keywords
            
            # Batch process the whole library
            embeddings = model.encode(df_raw["search_text"].tolist(), batch_size=32, show_progress_bar=True)
            df_raw["embedding"] = list(embeddings)
            
            with open(PROCESSED_DATA_FILE, "wb") as f:
                pickle.dump(df_raw, f)
            
            df = df_raw
            st.session_state.initialized = True
            st.rerun()
    else:
        st.info("Waiting for initialization... Click the button above to start.")
        st.stop()

# ==========================================
# 5. BUILD FAISS INDEX (Instant after init)
# ==========================================
embeddings_matrix = np.vstack(df["embedding"].values).astype('float32')
faiss.normalize_L2(embeddings_matrix)
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

st.success("AI Vector Database Ready ⚡")

# ==========================================
# 6. APP SETTINGS & LEARNING DATA
# ==========================================
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Auto-reject threshold", 0, 100, 45)

def get_learning():
    conn = sqlite3.connect(DB_PATH)
    learning_df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    return learning_df

learn_df = get_learning()

# ==========================================
# 7. MATCHING LOGIC
# ==========================================
def batch_match_queries(queries):
    results = [{"Product Name": q, "Status": "❌ Rejected", "Full Category Path": None, "Category Code": None, "Score": 0.0} for q in queries]
    learned_dict = {row["query"]: (row["correct_category"], row["category_code"]) for _, row in learn_df.iterrows()}
    
    unseen_queries, unseen_indices = [], []
    for i, q in enumerate(queries):
        if pd.isna(q) or str(q).strip() == "":
            results[i]["Status"] = "⚠️ Empty"
            continue
        q_lower = str(q).lower()
        if q_lower in learned_dict:
            cat, code = learned_dict[q_lower]
            results[i].update({"Status": "✅ Approved", "Full Category Path": cat, "Category Code": code, "Score": 100.0})
        else:
            unseen_queries.append(q_lower); unseen_indices.append(i)
            
    if unseen_queries:
        q_embs = model.encode(unseen_queries, batch_size=64, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_embs)
        all_scores, all_indices = index.search(q_embs, 1)
        
        for idx, best_idx_array, score_array in zip(unseen_indices, all_indices, all_scores):
            best_row = df.iloc[best_idx_array[0]]
            score = round(score_array[0] * 100, 2)
            results[idx].update({
                "Score": score, 
                "Full Category Path": best_row["category_path"], 
                "Category Code": best_row["category_code"],
                "Status": "✅ Approved" if score >= threshold else "❌ Rejected"
            })
    return results

# ==========================================
# 8. UI: SINGLE MATCH
# ==========================================
st.subheader("🔍 Single Match")
query = st.text_input("Enter product name")
if query:
    # Use the batch logic for a single query to keep code clean
    res = batch_match_queries([query])[0]
    if "Rejected" in res["Status"]:
        st.warning(f"❌ Rejected (Score: {res['Score']}) | Best Guess: **{res['Full Category Path']}**")
    else:
        st.success(f"✅ Approved | **{res['Full Category Path']}** | Score: {res['Score']}")

# ==========================================
# 9. UI: BULK MATCHING
# ==========================================
st.divider()
st.subheader("📂 Bulk Batch Processing")
bulk_file = st.file_uploader("Upload Product CSV", type=["csv"])

if bulk_file:
    try:
        bulk_df = pd.read_csv(bulk_file, sep=";", on_bad_lines='skip')
        if len(bulk_df.columns) == 1:
            bulk_file.seek(0)
            bulk_df = pd.read_csv(bulk_file, sep=",", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading CSV: {e}"); st.stop()

    col = next((c for c in bulk_df.columns if "NAME" == c.upper() or ("NAME" in c.upper() and "ID" not in c.upper())), None)

    if col:
        st.info(f"Detected column: `{col}`")
        if st.button("Start AI Processing"):
            with st.spinner("AI is thinking..."):
                results = batch_match_queries(bulk_df[col].tolist())
            
            out_df = pd.DataFrame(results)
            st.dataframe(out_df.head(100), column_config={"Score": st.column_config.ProgressColumn("Score", format="%f", min_value=0, max_value=100)})
            
            st.download_button("📥 Download Results", out_df.to_csv(index=False).encode('utf-8'), "ai_results.csv")
    else:
        st.error("No product name column found.")

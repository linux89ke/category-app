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
st.title("🧠 Semantic AI Category Engine (Cloud Optimized)")

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
    # Downloads the 'brain' of the engine - all-MiniLM-L6-v2
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================================
# 4. LOAD DATA & CHUNKED AI INITIALIZATION
# ==========================================
if "initialized" not in st.session_state:
    st.session_state.initialized = False

df = None

# Check if we already have the pre-calculated math file
if os.path.exists(PROCESSED_DATA_FILE):
    with open(PROCESSED_DATA_FILE, "rb") as f:
        df = pickle.load(f)
    st.session_state.initialized = True
else:
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found. Please upload the Excel file to your GitHub repository.")
        st.stop()
    
    st.warning("⚠️ The AI Engine needs a one-time initialization to understand your categories.")
    if st.button("🚀 Start AI Initialization"):
        # 1. Load the Excel
        df_raw = pd.read_excel(CATEGORY_FILE)
        df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(' ', '_')
        if "category_code" not in df_raw.columns:
            df_raw["category_code"] = df_raw.index.astype(str)
        
        keywords = df_raw["enriched_keywords"].fillna("").astype(str) if "enriched_keywords" in df_raw.columns else ""
        df_raw["search_text"] = df_raw["category_path"].astype(str) + " " + keywords
        
        # 2. Process in Chunks of 500 rows to prevent Streamlit Cloud Timeouts
        all_embeddings = []
        texts = df_raw["search_text"].tolist()
        chunk_size = 500 
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i:i + chunk_size]
            status_text.text(f"AI is crunching math for rows {i} to {i+len(batch_texts)} of {len(texts)}...")
            
            # Encode this small chunk
            batch_encodings = model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_encodings)
            
            # Update progress UI so the server knows the app is alive
            progress_bar.progress(min((i + chunk_size) / len(texts), 1.0))

        # 3. Combine and Save to Pickle
        df_raw["embedding"] = list(np.vstack(all_embeddings))
        with open(PROCESSED_DATA_FILE, "wb") as f:
            pickle.dump(df_raw, f)
        
        st.success("✅ AI Math Complete! App is restarting...")
        st.session_state.initialized = True
        st.rerun()
    else:
        st.info("Waiting for initialization. Click the button above to begin.")
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
threshold = st.sidebar.slider("Auto-reject threshold (Confidence)", 0, 100, 45)

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
        # Batch encode user queries
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
query = st.text_input("Enter product name (e.g. 'Mens Perfume')")
if query:
    res = batch_match_queries([query])[0]
    if "Rejected" in res["Status"]:
        st.warning(f"❌ Rejected (Score: {res['Score']}) | AI Suggestion: **{res['Full Category Path']}**")
    else:
        st.success(f"✅ Approved | **{res['Full Category Path']}** | Score: {res['Score']}")

    with st.expander("Teach the Engine (Override)"):
        correct = st.text_input("Correct category path")
        correct_code = st.text_input("Correct category code")
        if st.button("Save Learning") and correct:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (query, correct_category, category_code) VALUES (?, ?, ?)", (query.lower(), correct, correct_code))
            conn.commit()
            conn.close()
            st.success("Learned! ✅ Next time this product will be 100% accurate.")

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

    # Detect the correct Name column
    col = next((c for c in bulk_df.columns if "NAME" == c.upper() or ("NAME" in c.upper() and "ID" not in c.upper())), None)

    if col:
        st.info(f"Detected product column: `{col}`")
        if st.button("Start AI Batch Processing"):
            with st.spinner("AI is analyzing meanings and context..."):
                results = batch_match_queries(bulk_df[col].tolist())
            
            out_df = pd.DataFrame(results)
            st.dataframe(
                out_df.head(100), 
                column_config={"Score": st.column_config.ProgressColumn("Confidence Score", format="%f", min_value=0, max_value=100)}
            )
            
            st.download_button("📥 Download Categorized Results", out_df.to_csv(index=False).encode('utf-8'), "ai_categorized_results.csv")
    else:
        st.error("Could not find a 'NAME' column in your CSV.")

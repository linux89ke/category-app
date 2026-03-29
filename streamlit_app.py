import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Category Matching Engine", layout="wide")
st.title("⚡ Category Engine (Matrix Math + Learning)")

DB_PATH = "learning.db"
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
# 3. LOAD DATA & BUILD MATRIX ENGINE
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ File not found: `{CATEGORY_FILE}`. Please ensure it is in the same folder as this script.")
        st.write("📂 Files Python can see in this folder:", os.listdir())
        st.stop()
        
    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    if "category_code" not in df.columns:
        df["category_code"] = df.index.astype(str)

    keywords = df["enriched_keywords"].fillna("").astype(str) if "enriched_keywords" in df.columns else ""
    df["search_text"] = df["category_path"].astype(str) + " " + keywords
    
    return df

@st.cache_resource
def build_search_engine(dataframe):
    # This acts like fuzzy matching by breaking words into 3-4 letter chunks
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
    tfidf_matrix = vectorizer.fit_transform(dataframe["search_text"])
    return vectorizer, tfidf_matrix

df = load_data()
vectorizer, tfidf_matrix = build_search_engine(df)
st.success("High-Speed Matrix Engine Ready ⚡")

# ==========================================
# 4. APP SETTINGS & LEARNING DATA
# ==========================================
st.sidebar.header("⚙️ Settings")
# Note: TF-IDF scores slightly differently than RapidFuzz. 
# You might need to adjust this threshold. Usually, 30-50 is a good strict cutoff.
threshold = st.sidebar.slider("Auto-reject threshold", 0, 100, 35)

def get_learning():
    conn = sqlite3.connect(DB_PATH)
    learning_df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    return learning_df

learn_df = get_learning()

# ==========================================
# 5. CORE MATCHING LOGIC (Single)
# ==========================================
def match_query(query):
    q = str(query).lower()
    
    learned = learn_df[learn_df["query"] == q]
    if not learned.empty:
        row = learned.iloc[-1]
        return "✅ Approved", row["correct_category"], row["category_code"], 100.0

    # Fast Matrix Math for single item
    q_vec = vectorizer.transform([q])
    cosine_similarities = linear_kernel(q_vec, tfidf_matrix).flatten()
    
    best_idx = cosine_similarities.argmax()
    score = cosine_similarities[best_idx] * 100 # Convert to percentage
    
    best_row = df.iloc[best_idx]
    
    status = "✅ Approved" if score >= threshold else "❌ Rejected"
    
    return status, best_row["category_path"], best_row["category_code"], score

# ==========================================
# 6. INSTANT BATCH LOGIC
# ==========================================
def batch_match_queries(queries):
    # Prepare an empty list to keep original CSV order
    results = [{"Product Name": q, "Status": "❌ Rejected", "Full Category Path": None, "Category Code": None, "Score": 0.0} for q in queries]
    
    learned_dict = {row["query"]: (row["correct_category"], row["category_code"]) for _, row in learn_df.iterrows()}
    
    unseen_queries = []
    unseen_indices = []
    
    # Check what we already learned
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
            
    # BLAZING FAST MATH: Calculate everything else instantly
    if unseen_queries:
        q_vecs = vectorizer.transform(unseen_queries)
        cosine_similarities = linear_kernel(q_vecs, tfidf_matrix)
        
        best_indices = cosine_similarities.argmax(axis=1)
        best_scores = cosine_similarities.max(axis=1) * 100
        
        for idx, best_idx, score in zip(unseen_indices, best_indices, best_scores):
            best_row = df.iloc[best_idx]
            
            # Set the values
            results[idx]["Score"] = round(score, 2)
            results[idx]["Full Category Path"] = best_row["category_path"]
            results[idx]["Category Code"] = best_row["category_code"]
            
            # Assign Status
            if score < threshold:
                results[idx]["Status"] = "❌ Rejected"
            else:
                results[idx]["Status"] = "✅ Approved"
                
    return results

# ==========================================
# 7. UI: SINGLE MATCH
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

    if "NAME" in bulk_df.columns:
        col = "NAME"
    elif "name" in bulk_df.columns:
        col = "name"
    else:
        col = next((c for c in bulk_df.columns if ("name" in c.lower() or "product" in c.lower()) and "id" not in c.lower() and "seller" not in c.lower()), None)

    if col:
        st.info(f"Detected product column: `{col}` ({len(bulk_df)} items)")
        if st.button("Start High-Speed Process"):
            
            with st.spinner(f"Running Matrix Math on {len(bulk_df)} items... ⚡"):
                queries_list = bulk_df[col].tolist()
                results = batch_match_queries(queries_list)

            st.success("Processing Complete! ✅")

            out_df = pd.DataFrame(results)
            
            # Make the table look nice in Streamlit
            st.dataframe(
                out_df.head(100),
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        help="Confidence Score",
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
            col3.metric("❌ Rejected (Needs Review)", rejected)

            csv_data = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Matched Results", csv_data, "categorized_products.csv", "text/csv")
    else:
        st.error("No product name column found.")

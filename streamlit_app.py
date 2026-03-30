import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Category Matcher", layout="wide")
st.title("🔎 Category Matcher")

DB_PATH = "learning.db"
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

# ==========================================
# 2. QUERY CLEANING & EXPANSION
# ==========================================

# Strip these from queries only — not from category text
_BRANDS = {
    'samsung', 'apple', 'sony', 'lg', 'huawei', 'xiaomi', 'nokia', 'oppo',
    'tecno', 'itel', 'infinix', 'toyota', 'honda', 'ford', 'nissan', 'mazda',
    'pampers', 'huggies', 'molfix', 'nike', 'adidas', 'puma', 'reebok', 'fila',
    'hp', 'dell', 'lenovo', 'acer', 'asus', 'hisense', 'whirlpool', 'nescafe',
    'unilever', 'nestle', 'loreal', 'colgate', 'dettol', 'gillette', 'philips',
}
_TECH_NOISE = {
    'i3', 'i5', 'i7', 'i9', 'qled', 'oled', 'uhd', 'fhd', 'hd', '4k', '8k',
    'gen', 'monocrystalline', 'polycrystalline', 'pro', 'max', 'ultra', 'plus',
    'mini', 'lite', 'turbo', 'series', 'edition', 'model', 'version',
}
_FILLER = {
    'free', 'shipping', 'best', 'price', 'new', 'promo', 'original', 'authentic',
    'pack', 'dozen', 'strong', 'instant', 'buy', 'sale', 'the', 'with', 'and',
    'for', 'set', 'kit',
}
_ALL_QUERY_NOISE = _BRANDS | _TECH_NOISE | _FILLER

# Map unknown/brand terms to taxonomy-friendly words
_EXPANSIONS = {
    'iphone':       'phone smartphone mobile',
    'ipad':         'tablet',
    'android':      'phone smartphone',
    'tv':           'television',
    'telly':        'television',
    'fridge':       'refrigerator',
    'moringa':      'herbal supplement vitamins',
    'ashwagandha':  'herbal supplement',
    'turmeric':     'herbal supplement spice',
    'spirulina':    'herbal supplement',
    'neem':         'herbal supplement',
    'blender':      'blender mixer kitchen',
    'earbuds':      'earphones headphones',
    'airpods':      'earphones headphones wireless',
    'sneakers':     'shoes athletic',
    'trainers':     'shoes athletic',
    'tyre':         'tire',
    'tyres':        'tires',
    'jerrycan':     'container storage',
    'sufuria':      'cookware pot',
    'jiko':         'stove charcoal',
    'mkeka':        'mat floor',
}

def clean_query(text: str) -> str:
    """Strip noise, expand known terms, normalise for TF-IDF lookup."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove quantities/units
    text = re.sub(
        r'\d+\s*(ml|cl|l|g|kg|mg|units|tablets|pcs|capsules|oz|count|ct|'
        r'inch|cm|mm|w|kw|mah|gb|tb|mb|m\b)',
        '', text
    )
    text = re.sub(r'\b\d+\b', '', text)          # bare numbers
    text = re.sub(r'[^a-z\s]', ' ', text)        # punctuation
    tokens = [t for t in text.split() if t not in _ALL_QUERY_NOISE and len(t) > 1]

    # Expand known terms
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in _EXPANSIONS:
            expanded.extend(_EXPANSIONS[t].split())

    # Pluralise / singularise the last meaningful word to improve bigram recall
    if expanded:
        last = expanded[-1]
        if not last.endswith('s'):
            expanded.append(last + 's')
        elif len(last) > 4:
            expanded.append(last[:-1])

    result = " ".join(expanded).strip()
    # Fallback: if expansion stripped everything, use raw clean text
    if not result:
        result = re.sub(r'[^a-z\s]', ' ', text.lower()).strip()
    return result


def clean_category_text(text: str) -> str:
    """Light clean for category paths and keywords (no noise stripping)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z\s]', ' ', text.lower()).strip()


# ==========================================
# 3. DATA LOADING & INDEX BUILDING
# ==========================================
@st.cache_resource(show_spinner="Building search index…")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"❌ `{CATEGORY_FILE}` not found. Place it in the same folder as this script.")
        st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df['depth'] = df['category_path'].apply(lambda x: str(x).count('/') + 1)

    # Build search text: repeat path 3× (high weight) + plain keywords
    # Do NOT use enriched_keywords — it only adds "buy X / X for sale" noise.
    df['path_clean'] = df['category_path'].apply(clean_category_text)
    df['kw_clean']   = df['keywords'].fillna('').apply(clean_category_text)
    df['search_text'] = (df['path_clean'] + ' ') * 3 + df['kw_clean']

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=80_000,
        strip_accents='unicode',
    )
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    # Master code → full path lookup
    code_to_path = dict(zip(df['category_code'].astype(str), df['category_path']))

    return df, vectorizer, tfidf_matrix, code_to_path


df_main, vectorizer, tfidf_matrix, master_code_map = build_index()


# ==========================================
# 4. MATCHING ENGINE
# ==========================================
def match_single_item(
    product_name: str,
    threshold: int,
    learned_dict: dict,
) -> tuple[str, str | None, float, str]:
    """
    Returns (category_path, category_code, confidence_score, status_label).

    Pipeline:
      1. Manual-override check (instant, from learning DB).
      2. TF-IDF cosine similarity across all 30 k categories (vectorised).
      3. Fuzzy re-rank of top-25 candidates against the clean path string.
      4. Small depth bonus to prefer specific categories over generic parents.
    """
    original_query = str(product_name).lower().strip()

    # ── 1. Manual override ──────────────────────────────────────────────────
    if learned_dict and original_query in learned_dict:
        cat, code = learned_dict[original_query]
        return cat, code, 100.0, "✅ Approved"

    # ── 2. TF-IDF cosine similarity ─────────────────────────────────────────
    q = clean_query(product_name)
    if not q:
        return "Uncategorized", None, 0.0, "❌ Rejected"

    q_vec = vectorizer.transform([q])
    sims  = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # Pull top-25 candidates (argpartition is O(n), faster than full argsort)
    top_n   = 25
    top_idx = np.argpartition(sims, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    # ── 3. Fuzzy re-rank ────────────────────────────────────────────────────
    best_combined = -1.0
    best_row      = None

    for idx in top_idx:
        row        = df_main.iloc[idx]
        cos_score  = float(sims[idx])
        path_fuzzy = fuzz.token_set_ratio(q, row['path_clean']) / 100.0
        depth      = int(row['depth'])

        # Cosine dominates (0.65), fuzzy adds precision (0.30),
        # tiny depth nudge (0.008 per level) breaks ties in favour of specificity.
        combined = cos_score * 0.65 + path_fuzzy * 0.30 + depth * 0.008

        if combined > best_combined:
            best_combined = combined
            best_row      = row

    if best_row is None:
        return "Uncategorized", None, 0.0, "❌ Rejected"

    # Map combined score → 0-100 confidence and cap at 100
    confidence = round(min(best_combined * 125.0, 100.0), 2)
    status     = "✅ Approved" if confidence >= threshold else "❌ Rejected"
    return best_row['category_path'], str(best_row['category_code']), confidence, status


# ==========================================
# 5. SIDEBAR
# ==========================================
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Match threshold", 0, 100, 35)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Taxonomy loaded**  \n"
    f"{len(df_main):,} categories · "
    f"depth 1–{df_main['depth'].max()}"
)
if st.sidebar.button("Clear cache & reload"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()


# ==========================================
# 6. QUICK SINGLE-ITEM TEST
# ==========================================
with st.expander("🔬 Quick single-item test", expanded=False):
    test_input = st.text_input("Product name", placeholder="e.g. Samsung 65 inch QLED TV 4K")
    if st.button("Match") and test_input.strip():
        path, code, conf, status = match_single_item(test_input, threshold, {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", path)
        col2.metric("Code", code or "—")
        col3.metric("Confidence", f"{conf}%")
        st.write(f"**Status:** {status}")
        st.caption(f"Cleaned query sent to index: `{clean_query(test_input)}`")


# ==========================================
# 7. BATCH CSV UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Upload Product CSV", type="csv")

if uploaded_file:
    # Read CSV — handle comma and semicolon separators
    try:
        df_up = pd.read_csv(uploaded_file, on_bad_lines='skip')
        if len(df_up.columns) == 1:
            uploaded_file.seek(0)
            df_up = pd.read_csv(uploaded_file, sep=";", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Auto-detect columns
    name_col = next(
        (c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0]
    )
    code_col = next(
        (c for c in df_up.columns if "CODE" in c.upper() and "MATCH" not in c.upper()),
        None,
    )

    st.info(f"Matching names from column: **{name_col}**")
    if code_col:
        st.success(f"Original codes found in: **{code_col}**")
        df_up["Assigned Full Path"] = (
            df_up[code_col].astype(str).map(master_code_map).fillna("⚠️ Not in taxonomy")
        )

    if st.button("Start Analysis 🚀"):
        # Load learning DB
        conn = sqlite3.connect(DB_PATH)
        try:
            l_df = pd.read_sql("SELECT * FROM feedback", conn)
            learned_dict = {
                row['query']: (row['correct_category'], row['category_code'])
                for _, row in l_df.iterrows()
            }
        except Exception:
            learned_dict = {}
        conn.close()

        names      = df_up[name_col].tolist()
        total      = len(names)
        results    = [None] * total

        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        # Process sequentially — TF-IDF is already vectorised, threading adds
        # overhead without benefit for CPU-bound numpy/sklearn operations.
        for i, name in enumerate(names):
            results[i] = match_single_item(name, threshold, learned_dict)

            if i % 20 == 0 or i == total - 1:
                pct = (i + 1) / total
                progress_bar.progress(pct)
                status_text.markdown(
                    f"**Processing:** {i + 1} / {total} items — {int(pct * 100)}%"
                )

        status_text.success(f"✅ Matched {total} items!")

        # Attach results to dataframe
        new_cols = ["AI Category", "Matched Code", "Confidence", "Status"]
        df_out = df_up.drop(columns=[c for c in new_cols if c in df_up.columns])
        df_out["AI Category"]  = [r[0] for r in results]
        df_out["Matched Code"] = [r[1] for r in results]
        df_out["Confidence"]   = [r[2] for r in results]
        df_out["Status"]       = [r[3] for r in results]

        # Summary stats
        approved = sum(1 for r in results if r[3] == "✅ Approved")
        avg_conf = round(sum(r[2] for r in results) / total, 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total matched",  total)
        c2.metric("Approved",       f"{approved} ({round(approved/total*100)}%)")
        c3.metric("Avg confidence", f"{avg_conf}%")

        # Build display columns
        display_cols = [name_col]
        if code_col:
            display_cols.append("Assigned Full Path")
        display_cols += ["Status", "AI Category", "Confidence"]

        st.subheader("Side-by-side comparison")
        st.dataframe(df_out[display_cols].head(500), use_container_width=True)

        st.download_button(
            "📥 Download full results CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            "category_matches.csv",
        )

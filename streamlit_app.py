import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ==============================================================================
# 1. SETTINGS & PROFESSIONAL STYLE
# ==============================================================================
st.set_page_config(page_title="Category Test", layout="wide")
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #f8f9fa; color: #1c1e21; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #000000 !important; border-bottom: 2px solid #000000; padding-bottom: 10px; }
    .stButton > button { background: #000000 !important; color: white !important; border-radius: 4px; width: 100%; border: none; font-weight: 600; height: 45px; }
</style>
""", unsafe_allow_html=True)

st.title("Category Test")
st.caption("Taxonomy Audit | Un-Truncated Keywords | Additive Scoring | Version 2026.16")

# ==============================================================================
# 2. LOGIC HELPERS & CLEANERS
# ==============================================================================
_INVALID_NAMES = re.compile(r'^\s*(deleted|invalid|n\/a|na|null|none|test|sample|placeholder|tbd|xxx|remove|dummy)\s*$', re.I)

DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"}, "tablet": {"Phones & Tablets"},
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "dress": {"Fashion"},
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},
    "tv": {"Electronics"}, "television": {"Electronics"}, "headphone": {"Electronics"},
    "diaper": {"Grocery", "Baby Products"}, "perfume": {"Health & Beauty"}, "foundation": {"Health & Beauty"},
    "bike": {"Sporting Goods", "Automobile"}, "pot": {"Home & Office"}
}

def is_invalid_name(text):
    if not isinstance(text, str) or len(text.strip()) < 3: return True
    return bool(_INVALID_NAMES.match(text.strip()))

_STRIP_UNITS = re.compile(r'\b\d+\.?\d*\s*(?:ml|cl|l|pcs|inch|cm|mm|kw|kg|oz|lbs?)\b', re.IGNORECASE)
_PUNCT = re.compile(r"[^a-z0-9\s]")

def clean_product(text: str) -> str:
    """Cleans product names, stripping measurement noise."""
    if not isinstance(text, str): return ""
    text = text[:150].lower() 
    text = _STRIP_UNITS.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).strip()

def clean_keywords(text: str) -> str:
    """NEVER truncate the keywords! Strip punctuation only."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).strip()

def normalize_path(path_str: str) -> str:
    if not isinstance(path_str, str): return ""
    return re.sub(r"[^a-z0-9]", "", path_str.lower())

def resolve_path_idx(norm_assigned: str, path_map: dict) -> int | None:
    if norm_assigned in path_map: return path_map[norm_assigned]
    if not norm_assigned: return None
    best_score, best_idx = 0, None
    for key, idx in path_map.items():
        score = fuzz.ratio(norm_assigned, key)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score >= 90 else None

# ==============================================================================
# 3. INDEX BUILDER
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Taxonomy Logic...")
def build_index():
    try:
        df = pd.read_excel(CATEGORY_FILE, engine="openpyxl")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        path_col  = next((c for c in df.columns if "PATH" in c.upper()), df.columns[2])
        kw_col    = next((c for c in df.columns if "KEY"  in c.upper()), None)
        code_col  = next((c for c in df.columns if "CODE" in c.upper()), None)

        df["path_str"]  = df[path_col].astype(str).str.strip()
        df["norm_path"] = df["path_str"].apply(normalize_path)
        df["leaf_name"] = df["path_str"].apply(lambda x: x.split("/")[-1].strip().lower())

        # Process Keywords WITHOUT Truncation
        df["kw_list"] = (
            df[kw_col].fillna("").astype(str).apply(lambda x: set(clean_keywords(x).split()))
            if kw_col else pd.Series([set()] * len(df))
        )

        p_clean = df["path_str"].str.replace("/", " ").str.lower()
        k_clean = df[kw_col].fillna("").astype(str).str.lower() if kw_col else pd.Series([""] * len(df))
        df["search_text"] = (p_clean + " ") * 3 + (k_clean + " ") * 5

        vectorizer   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(df["search_text"])

        norm_path_to_idx = {row["norm_path"]: i for i, row in df.iterrows()}
        code_to_path = (
            dict(zip(df[code_col].astype(str).str.replace(r"\.0$", "", regex=True), df[path_col]))
            if code_col else {}
        )

        return df, vectorizer, tfidf_matrix, code_to_path, path_col, norm_path_to_idx

    except Exception as e:
        st.error(f"Load Error: {e}")
        return None, None, None, None, None, None

index_data = build_index()
if index_data[0] is not None:
    df_cat, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME, path_map = index_data
else:
    st.stop()

# ==============================================================================
# 4. ADDITIVE SCORING ENGINE (Max 100 Points)
# ==============================================================================
def score_assignment(clean_q: str, assigned_path: str, sims_row) -> float:
    norm_assigned = normalize_path(assigned_path)
    idx = resolve_path_idx(norm_assigned, path_map)
    if idx is None: return 0.0

    row       = df_cat.iloc[idx]
    cos_score = float(sims_row[idx])
    leaf      = row["leaf_name"]
    full_path = str(row[PATH_COL_NAME]).lower()

    query_tokens = set(clean_q.split())
    query_stems  = {t.rstrip("s") for t in query_tokens}

    # 1. KEYWORD MATCH (Max 45 Points)
    # +15 points for every matched keyword, up to a max of 45.
    kw_overlap = len(query_tokens & row["kw_list"])
    comp_kw = min(kw_overlap * 15.0, 45.0)

    # 2. LITERAL LEAF MATCH (Max 20 Points)
    # If any significant word from the category name is in the product name
    leaf_words = [w.rstrip("s") for w in leaf.replace("&", " ").split() if len(w) >= 3]
    literal_hit = bool(leaf_words and any(lw in query_stems for lw in leaf_words))
    comp_literal = 20.0 if literal_hit else 0.0

    # 3. TF-IDF COSINE MATCH (Max 25 Points)
    comp_cos = cos_score * 25.0

    # 4. FUZZY PATH MATCH (Max 10 Points)
    fuzz_ratio = fuzz.token_set_ratio(clean_q, full_path) / 100.0
    comp_fuzz = fuzz_ratio * 10.0

    # 5. DOMAIN PENALTY (-40 Points)
    top_level = str(row[PATH_COL_NAME]).split("/")[0].strip()
    required_verticals = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]
    penalty = 40.0 if (required_verticals and top_level not in required_verticals) else 0.0

    # FINAL MATH
    score = comp_kw + comp_literal + comp_cos + comp_fuzz - penalty
    return round(min(max(score, 0.0), 100.0), 1)

# ==============================================================================
# 5. UI
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    threshold = st.slider("Audit Threshold", 0, 100, 20)
    st.caption("Standard: ≥ 20.0 is Approved.")

uploaded_file = st.file_uploader("Upload CSV for Audit", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    if st.button("Run Audit Analysis", width="stretch"):
        if code_col:
            clean_codes = df_up[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df_up["Assigned category in full"] = clean_codes.map(master_code_map).fillna("Unknown")
        else:
            df_up["Assigned category in full"] = df_up.get("CATEGORY", "N/A")

        names   = df_up[name_col].fillna("").astype(str).tolist()
        cleaned = [clean_product(n) if not is_invalid_name(n) else "" for n in names]

        q_vecs   = vectorizer.transform(cleaned)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)

        results = []
        for i in range(len(names)):
            if not cleaned[i]:
                results.append((0.0, "Rejected"))
                continue
            conf = score_assignment(cleaned[i], df_up["Assigned category in full"].iloc[i], all_sims[i])
            status = "Approved" if conf >= threshold else "Rejected"
            results.append((conf, status))

        df_up["confidence"] = [r[0] for r in results]
        df_up["status"]     = [r[1] for r in results]

        final_cols = ["NAME", "Assigned category in full", "confidence", "status"]

        total    = len(df_up)
        approved = (df_up["status"] == "Approved").sum()
        rejected = total - approved

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Products", total)
        col2.metric("Approved", approved, delta=f"{approved/total*100:.1f}%")
        col3.metric("Rejected", rejected, delta=f"-{rejected/total*100:.1f}%", delta_color="inverse")

        st.subheader("Audit Results")
        st.dataframe(
            df_up[final_cols].head(2500),
            column_config={
                "confidence": st.column_config.ProgressColumn("Confidence Score", min_value=0, max_value=100)
            },
            width="stretch",
            hide_index=True,
        )
        st.download_button("Export Results", df_up[final_cols].to_csv(index=False).encode("utf-8"), "category_audit_v16.csv")

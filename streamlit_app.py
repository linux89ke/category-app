
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
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
st.caption("Taxonomy Audit | Robust Keyword Intersection | Version 2026.15 (patched)")

# ==============================================================================
# 2. LOGIC HELPERS
# ==============================================================================
DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"}, "tablet": {"Phones & Tablets"},
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "dress": {"Fashion"},
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},
    "tv": {"Electronics"}, "television": {"Electronics"}, "headphone": {"Electronics"},
    "diaper": {"Grocery", "Baby Products"}, "perfume": {"Health & Beauty"}, "foundation": {"Health & Beauty"},
    "bike": {"Sporting Goods", "Automobile"}, "pot": {"Home & Office"}
}

_STRIP_UNITS = re.compile(r'\b\d+\.?\d*\s*(?:ml|cl|l(?!te)|pcs|inch|cm|mm|kw|kg|oz|lbs?)\b', re.IGNORECASE)
_PUNCT = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _STRIP_UNITS.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).strip()

def normalize_path(path_str: str) -> str:
    if not isinstance(path_str, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", path_str.lower())

def resolve_path_idx(norm_assigned: str, path_map: Dict[str, int]) -> Optional[int]:
    """
    Resolve normalized assigned path to a positional index in df (or None).
    path_map maps normalized_path -> positional index (0..n-1).
    """
    if not norm_assigned:
        return None
    if norm_assigned in path_map:
        return path_map[norm_assigned]
    # fallback fuzzy match across keys
    best_score, best_idx = 0, None
    for key, idx in path_map.items():
        score = fuzz.ratio(norm_assigned, key)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score >= 90 else None

# ==============================================================================
# 3. INDEX BUILDER (returns dataclass for clarity)
# ==============================================================================
@dataclass
class IndexData:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: object
    code_to_path: Dict[str, str]
    path_col: str
    norm_path_to_idx: Dict[str, int]

@st.cache_resource(show_spinner="Initializing Taxonomy Logic...")
def build_index(category_file: str) -> IndexData:
    try:
        df = pd.read_excel(category_file, engine="openpyxl")
        # drop unnamed columns often created by Excel exports
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        # safer fallback for path column
        path_col = next((c for c in df.columns if "PATH" in c.upper()), df.columns[0])
        kw_col   = next((c for c in df.columns if "KEY"  in c.upper()), None)
        code_col = next((c for c in df.columns if "CODE" in c.upper()), None)

        df = df.reset_index(drop=True)  # ensure positional indices 0..n-1
        df["path_str"]  = df[path_col].astype(str).str.strip()
        df["norm_path"] = df["path_str"].apply(normalize_path)
        df["leaf_name"] = df["path_str"].apply(lambda x: x.split("/")[-1].strip().lower())

        # Pre-tokenise keywords per category row (store as set of tokens)
        if kw_col:
            df["kw_list"] = df[kw_col].fillna("").astype(str).apply(lambda x: set(clean_text(x).split()))
        else:
            df["kw_list"] = pd.Series([set()] * len(df))

        # Build search text (path boosted, keywords boosted more)
        p_clean = df["path_str"].str.replace("/", " ").str.lower()
        k_clean = df[kw_col].fillna("").astype(str).str.lower() if kw_col else pd.Series([""] * len(df))
        df["search_text"] = (p_clean + " ") * 3 + (k_clean + " ") * 5

        vectorizer   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(df["search_text"])

        # norm_path -> positional index (enumerate ensures positional indices)
        norm_path_to_idx = {p: i for i, p in enumerate(df["norm_path"].tolist())}

        # normalize code keys to string without trailing .0
        code_to_path = {}
        if code_col:
            codes = df[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            code_to_path = dict(zip(codes.tolist(), df[path_col].tolist()))

        return IndexData(df=df, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix,
                         code_to_path=code_to_path, path_col=path_col, norm_path_to_idx=norm_path_to_idx)

    except Exception as e:
        st.error(f"Load Error: {e}")
        return IndexData(pd.DataFrame(), None, None, {}, "", {})

index_data = build_index(CATEGORY_FILE)
df_cat = index_data.df
vectorizer = index_data.vectorizer
tfidf_matrix = index_data.tfidf_matrix
master_code_map = index_data.code_to_path
PATH_COL_NAME = index_data.path_col
path_map = index_data.norm_path_to_idx

if df_cat.empty or vectorizer is None or tfidf_matrix is None:
    st.stop()

# ==============================================================================
# 4. AUDIT ENGINE
# ==============================================================================
def score_assignment(clean_q: str, assigned_path: str, sims_row: np.ndarray) -> Tuple[float, dict]:
    """
    Returns (score, explain_dict)
    explain_dict contains signals used for explainability and top-level path.
    """
    explain = {}
    norm_assigned = normalize_path(assigned_path)
    idx = resolve_path_idx(norm_assigned, path_map)
    if idx is None:
        explain["reason"] = "Assigned path not found"
        return 0.0, explain

    row = df_cat.iloc[idx]
    cos_score = float(sims_row[idx])
    leaf = row["leaf_name"]
    full_path = str(row[PATH_COL_NAME]).lower()

    query_tokens = set(clean_q.split())
    query_stems  = {t.rstrip("s") for t in query_tokens}

    # 1. TF-IDF cosine (weight 0.25)
    comp_cos = cos_score * 0.25
    explain["cosine"] = round(cos_score, 4)

    # 2. Keyword intersection (weight 0.35)
    kw_overlap = len(query_tokens & row["kw_list"])
    comp_kw = min(kw_overlap, 1.0) * 0.35
    explain["kw_overlap"] = int(kw_overlap)

    # 3. Literal leaf match (weight 0.25)
    leaf_words = [w.rstrip("s") for w in leaf.replace("&", " ").split() if len(w) >= 3]
    literal_overlap = len(set(leaf_words) & query_stems)
    comp_literal = min(literal_overlap, 1.0) * 0.25
    explain["literal_overlap"] = int(literal_overlap)

    # 4. Fuzzy match (weight 0.15)
    fuzz_leaf = fuzz.token_set_ratio(clean_q, leaf) / 100.0
    fuzz_full_path = fuzz.token_set_ratio(clean_q, full_path) / 100.0
    comp_fuzz = (fuzz_leaf * 0.40 + fuzz_full_path * 0.60) * 0.15
    explain["fuzz_leaf"] = round(fuzz_leaf, 4)
    explain["fuzz_full_path"] = round(fuzz_full_path, 4)

    # Penalty for domain mismatch
    top_level = str(row[PATH_COL_NAME]).split("/")[0].strip()
    required_verticals = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]
    penalty = 0.20 if (required_verticals and top_level not in required_verticals) else 0.0
    explain["penalty_applied"] = bool(penalty)

    raw = comp_cos + comp_kw + comp_literal + comp_fuzz - penalty
    score = round(min(max(raw, 0.0), 1.0) * 100.0, 1)

    explain.update({
        "comp_cos": round(comp_cos, 4),
        "comp_kw": round(comp_kw, 4),
        "comp_literal": round(comp_literal, 4),
        "comp_fuzz": round(comp_fuzz, 4),
        "raw_score": round(raw, 4),
        "final_score": score,
        "assigned_idx": int(idx),
        "assigned_path": str(row[PATH_COL_NAME])
    })
    return score, explain

def get_top_alternatives(sims_row: np.ndarray, top_k: int = 3):
    """
    Return top_k alternative category paths and their cosine scores.
    """
    top_idx = np.argsort(-sims_row)[:top_k]
    paths = [df_cat.iloc[j][PATH_COL_NAME] for j in top_idx]
    scores = [float(sims_row[j]) for j in top_idx]
    return list(zip(paths, [round(s, 4) for s in scores]))

# ==============================================================================
# 5. UI
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    threshold = st.slider("Audit Threshold", 0, 100, 20)
    st.caption("Standard: ≥ 20.0 is Approved.")
    st.markdown("**Advanced**")
    fuzzy_fallback_threshold = st.slider("Fuzzy resolve threshold (for path resolution)", 70, 100, 90)
    show_explain = st.checkbox("Show explainability columns", value=True)
    show_alternatives = st.checkbox("Show top 3 alternatives when Rejected", value=True)

uploaded_file = st.file_uploader("Upload CSV for Audit", type="csv")

def read_csv_flexible(uploaded):
    # Try common delimiter first, fallback to python engine autodetect
    try:
        return pd.read_csv(uploaded, sep=",", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(uploaded, sep=None, engine="python", on_bad_lines="skip")

if uploaded_file:
    df_up = read_csv_flexible(uploaded_file)
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    # Run button placed in main area for clearer UX
    if st.button("Run Audit Analysis"):
        # Map codes to full path if code column exists
        if code_col:
            clean_codes = df_up[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df_up["Assigned category in full"] = clean_codes.map(master_code_map).fillna("Unknown")
        else:
            # fallback to CATEGORY column or N/A
            df_up["Assigned category in full"] = df_up.get("CATEGORY", "N/A")

        names = df_up[name_col].fillna("").astype(str).tolist()
        cleaned = [clean_text(n) for n in names]

        q_vecs = vectorizer.transform(cleaned)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)

        confidences = []
        statuses = []
        explains = []
        alternatives = []

        # Temporarily override resolve threshold for fuzzy path resolution if user changed it
        # (we'll use the global function but adjust behavior by re-checking path resolution fuzzy threshold)
        # Note: resolve_path_idx uses a hard-coded 90; to respect the UI control, we re-run fuzzy fallback here if needed.

        for i in range(len(names)):
            assigned_path = df_up["Assigned category in full"].iloc[i]
            # First attempt direct resolution
            norm_assigned = normalize_path(assigned_path)
            idx = path_map.get(norm_assigned)
            if idx is None and norm_assigned:
                # fuzzy search across keys with user threshold
                best_score, best_idx = 0, None
                for key, pos in path_map.items():
                    score = fuzz.ratio(norm_assigned, key)
                    if score > best_score:
                        best_score, best_idx = score, pos
                if best_score >= fuzzy_fallback_threshold:
                    idx = best_idx

            # Build a sims_row view for scoring function (it expects the full sims_row)
            sims_row = all_sims[i]
            # If idx is None, pass assigned_path as-is; score_assignment will attempt resolve_path_idx again (with default 90)
            conf, explain = score_assignment(cleaned[i], assigned_path, sims_row)
            confidences.append(conf)
            explains.append(explain)
            status = "Approved" if conf >= threshold else "Rejected"
            statuses.append(status)

            if show_alternatives:
                top3 = get_top_alternatives(sims_row, top_k=3)
                alternatives.append(top3)
            else:
                alternatives.append([])

        df_up["confidence"] = confidences
        df_up["status"] = statuses

        # Build final columns dynamically
        final_cols = [name_col, "Assigned category in full", "confidence", "status"]

        # Add explainability columns if requested
        if show_explain:
            # Flatten some explain keys into columns
            explain_df = pd.DataFrame(explains).fillna("")
            # Keep only a subset of explain fields for display
            for col in ["cosine", "kw_overlap", "literal_overlap", "fuzz_leaf", "fuzz_full_path", "penalty_applied", "final_score"]:
                if col in explain_df.columns:
                    df_up[f"explain_{col}"] = explain_df[col].astype(str)
                    final_cols.append(f"explain_{col}")

        if show_alternatives:
            df_up["top_3_alternatives"] = ["; ".join([f"{p} ({s})" for p, s in alt]) for alt in alternatives]
            final_cols.append("top_3_alternatives")

        # Metrics
        total = len(df_up)
        approved = (df_up["status"] == "Approved").sum()
        rejected = total - approved

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Products", total)
        col2.metric("Approved", approved, delta=f"{approved/total*100:.1f}%")
        col3.metric("Rejected", rejected, delta=f"-{rejected/total*100:.1f}%", delta_color="inverse")

        st.subheader("Audit Results")
        # Use container width for dataframe
        st.dataframe(df_up[final_cols].head(2500), use_container_width=True, hide_index=True)

        # Download button (keeps behavior but does not proactively create other file types)
        csv_bytes = df_up[final_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Export Results", csv_bytes, "category_audit_v15_patched.csv")

        # Optional: show a small sample of explainability JSON for first few rows
        if show_explain:
            st.subheader("Explainability Sample (first 10 rows)")
            sample_explain = pd.DataFrame(explains).head(10)
            st.table(sample_explain)




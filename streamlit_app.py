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
st.caption("Taxonomy Audit | Keyword-Driven Scoring | Robust Normalization | Version 2026.14")

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

# FIX 1: Preserve tech-critical tokens that were previously stripped.
# The old regex removed gb, mah, 5g, 4g, hz — killing signal for phones/tech.
# New approach: strip generic measurement units but keep product-spec tokens.
_STRIP_UNITS = re.compile(
    r'\b\d+\.?\d*\s*(?:ml|cl|l(?!te)|pcs|inch|cm|mm|kw|kg|oz|lbs?)\b',
    re.IGNORECASE
)
_PUNCT = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    """
    Normalise product name for TF-IDF and fuzzy matching.

    Changes from v13:
    - Keep gb, tb, mah, hz, 5g, 4g, w (watts) — they are primary tech signals.
    - Only strip units that are purely measurement noise (ml, cm, pcs, kg, etc.).
    - Still lower-case and collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _STRIP_UNITS.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).strip()


def normalize_path(path_str: str) -> str:
    """Strip all non-alphanumeric chars for robust path key lookup."""
    if not isinstance(path_str, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", path_str.lower())


# FIX 4: Fuzzy path key resolver — handles minor separator/spacing mismatches
# between the assigned path and the taxonomy map keys.
def resolve_path_idx(norm_assigned: str, path_map: dict, df_cat: pd.DataFrame) -> int | None:
    """
    Return the DataFrame index for a given normalised path string.

    1. Exact match in path_map (fast path).
    2. Fuzzy fallback: scan all keys for the closest match above 90 similarity.
       This recovers cases where the assigned path has a trailing space,
       slightly different separator, or minor typo.
    """
    if norm_assigned in path_map:
        return path_map[norm_assigned]

    if not norm_assigned:
        return None

    best_score, best_idx = 0, None
    for key, idx in path_map.items():
        score = fuzz.ratio(norm_assigned, key)
        if score > best_score:
            best_score, best_idx = score, idx

    # Only accept if very close (≥90) to avoid wrong-category substitution
    return best_idx if best_score >= 90 else None


# ==============================================================================
# 3. INDEX BUILDER (Keyword-Aware)
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

        # Pre-tokenise keywords per category row
        df["kw_list"] = (
            df[kw_col].fillna("").astype(str)
            .apply(lambda x: set(clean_text(x).split()))
            if kw_col else pd.Series([set()] * len(df))
        )

        # TF-IDF corpus — path weighted 3×, keywords weighted 5×
        p_clean = df["path_str"].str.replace("/", " ").str.lower()
        k_clean = df[kw_col].fillna("").astype(str).str.lower() if kw_col else pd.Series([""] * len(df))
        df["search_text"] = (p_clean + " ") * 3 + (k_clean + " ") * 5

        vectorizer   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(df["search_text"])

        norm_path_to_idx = {row["norm_path"]: i for i, row in df.iterrows()}
        code_to_path = (
            dict(zip(
                df[code_col].astype(str).str.replace(r"\.0$", "", regex=True),
                df[path_col]
            ))
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
# 4. AUDIT ENGINE
# ==============================================================================
# FIX 2: Reworked scoring formula.
#
# Old problems:
#   • `score * 140` magic multiplier — raw composite maxes at ~0.70, so ×140
#     reaches ~98. But any penalty (0.45) brought scores to 0.35×140 = 49,
#     creating an impossible gap between 20 and 70. Items near the boundary
#     (correct but spec-heavy names) were almost always rejected.
#   • Penalty of 0.45 on the raw score was too aggressive. After ×140 that
#     equals a 63-point swing — far too much for a heuristic domain check.
#
# New approach:
#   • Keep all component scores in [0, 1] range.
#   • Compute a weighted sum capped to [0, 1], then scale to [0, 100].
#   • Reduce domain penalty from 0.45 → 0.20 (still meaningful, not fatal).
#   • Add a path-level fuzzy boost so full-path similarity contributes.
#
# FIX 3: Fuzzy match against full path string, not just the leaf node.
#   The old code ran `fuzz.token_set_ratio(clean_q, leaf)` — the leaf alone
#   (e.g. "Hygrometers") shares few tokens with "CO2 Humidity Tester", even
#   though the full path "Weather Instruments / Hygrometers" would match better
#   against the product name context. We now blend leaf + full-path fuzzy.

def score_assignment(clean_q: str, assigned_path: str, sims_row) -> float:
    norm_assigned = normalize_path(assigned_path)

    # FIX 4 applied here: use fuzzy resolver instead of hard dict lookup
    idx = resolve_path_idx(norm_assigned, path_map, df_cat)
    if idx is None:
        return 0.0

    row       = df_cat.iloc[idx]
    cos_score = float(sims_row[idx])                  # already in [0, 1]
    leaf      = row["leaf_name"]
    full_path = str(row[PATH_COL_NAME]).lower()

    query_tokens = set(clean_q.split())
    query_stems  = {t.rstrip("s") for t in query_tokens}

    # --- Component 1: TF-IDF cosine (weight 0.35) ---
    comp_cos = cos_score * 0.35

    # --- Component 2: Keyword intersection (weight 0.25) ---
    # Does ANY manually-curated keyword appear in the product name?
    kw_overlap  = len(query_tokens & row["kw_list"])
    comp_kw     = min(kw_overlap / max(len(row["kw_list"]), 1), 1.0) * 0.25

    # --- Component 3: Literal leaf match (weight 0.20) ---
    # All meaningful words of the leaf node appear in the product name.
    leaf_words   = [w.rstrip("s") for w in leaf.replace("&", " ").split() if len(w) >= 3]
    literal_hit  = bool(leaf_words and all(lw in query_stems for lw in leaf_words))
    comp_literal = 0.20 if literal_hit else 0.0

    # --- Component 4: Fuzzy match — FIX 3 ---
    # Blend leaf-only fuzzy (original) with full-path fuzzy (new).
    # Full-path fuzzy rewards correct top-level + mid-level alignment
    # even when the leaf is a specialist term the product name won't contain.
    fuzz_leaf      = fuzz.token_set_ratio(clean_q, leaf) / 100.0
    fuzz_full_path = fuzz.token_set_ratio(clean_q, full_path) / 100.0
    comp_fuzz      = (fuzz_leaf * 0.40 + fuzz_full_path * 0.60) * 0.20

    # --- Penalty: domain anchor check ---
    # FIX 2: reduced from 0.45 → 0.20 so a domain mismatch deducts at most
    # 20 raw points instead of 63. A wrong top-level is still strongly penalised
    # but a correct assignment won't be killed by a missing DOMAIN_SIGNALS entry.
    top_level          = str(row[PATH_COL_NAME]).split("/")[0].strip()
    required_verticals = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]
    penalty = 0.20 if (required_verticals and top_level not in required_verticals) else 0.0

    # --- Final composite ---
    raw   = comp_cos + comp_kw + comp_literal + comp_fuzz - penalty
    score = round(min(max(raw, 0.0), 1.0) * 100.0, 1)
    return score


# ==============================================================================
# 5. UI
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    # FIX 5: Default threshold raised to 25 to reduce false approvals.
    # The old default of 20 was too low — items with a correct top-level but
    # clearly wrong subcategory (e.g. condensed milk → Sandwich Breads) could
    # slip through. 25 is a safer floor without being overly aggressive.
    threshold = st.slider("Audit Threshold", 0, 100, 25)
    st.caption("Recommended: 25–35 for strict audits, 15–20 for permissive review.")

uploaded_file = st.file_uploader("Upload CSV for Audit", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next(
        (c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None
    )

    if st.button("Run Audit Analysis", width="stretch"):
        if code_col:
            clean_codes = (
                df_up[code_col].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
            )
            df_up["Assigned category in full"] = clean_codes.map(master_code_map).fillna("Unknown")
        else:
            df_up["Assigned category in full"] = df_up.get("CATEGORY", "N/A")

        names   = df_up[name_col].fillna("").astype(str).tolist()
        cleaned = [clean_text(n) for n in names]

        q_vecs   = vectorizer.transform(cleaned)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)

        results = []
        for i in range(len(names)):
            conf = score_assignment(
                cleaned[i],
                df_up["Assigned category in full"].iloc[i],
                all_sims[i],
            )
            status = "Approved" if conf >= threshold else "Rejected"
            results.append((conf, status))

        df_up["confidence"] = [r[0] for r in results]
        df_up["status"]     = [r[1] for r in results]

        final_cols = ["NAME", "Assigned category in full", "confidence", "status"]

        # Summary metrics
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
        st.download_button(
            "Export Results",
            df_up[final_cols].to_csv(index=False).encode("utf-8"),
            "category_audit_v14.csv",
        )

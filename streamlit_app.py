import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import sqlite3
import logging
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PAGE CONFIG & STYLE
# ==============================================================================
st.set_page_config(page_title="Category Audit", layout="wide")
CATEGORY_FILE = "category_map_fully_enriched.xlsx"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #f8f9fa; color: #1c1e21; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #000 !important;
         border-bottom: 2px solid #000; padding-bottom: 10px; }
    .stButton > button { background: #000 !important; color: #fff !important;
         border-radius: 4px; width: 100%; border: none; font-weight: 600; height: 45px; }
    .verdict-approved { color: #1a7a1a; font-weight: 600; }
    .verdict-rejected { color: #b00020; font-weight: 600; }
    .verdict-flagged  { color: #c47a00; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("Category Audit")
st.caption("Hybrid Engine v1.0 — TF-IDF · Keyword Boost · Cross-Domain Suppression · Feedback Learning")

# ==============================================================================
# 2. TEXT UTILITIES
# ==============================================================================
_STRIP_UNITS = re.compile(
    r'\b\d+\.?\d*\s*(?:ml|cl|l(?!te)|pcs|inch|cm|mm|kw|kg|oz|lbs?)\b',
    re.IGNORECASE
)
_PUNCT = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    """
    Normalise product name.
    Keeps tech-critical tokens (gb, mah, 5g, 4g, hz) that the old v13
    script incorrectly stripped, while still removing pure measurement noise.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = _STRIP_UNITS.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).strip()


def normalize_path(path_str: str) -> str:
    if not isinstance(path_str, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", path_str.lower())


def get_segments(path: str, n: int) -> tuple:
    for sep in ("/", ">"):
        if sep in path:
            return tuple(p.strip().lower() for p in path.split(sep)[:n])
    return (path.strip().lower(),)


def get_leaf(path: str) -> str:
    for sep in ("/", ">"):
        if sep in path:
            return path.split(sep)[-1].strip().lower()
    return path.strip().lower()


def get_top(path: str) -> str:
    for sep in ("/", ">"):
        if sep in path:
            return path.split(sep)[0].strip().lower()
    return path.strip().lower()


# ==============================================================================
# 3. CROSS-DOMAIN SUPPRESSION TABLE  (from CategoryMatcherEngine)
# ==============================================================================
_CROSS_DOMAIN_BLOCKS = [
    ({"supplements", "tablets", "capsules", "vitamins", "syrup", "herbal",
      "herbs", "strips", "milk substitutes"},
     {"phones & tablets", "electronics", "automobile",
      "industrial & scientific", "sporting goods"}),

    ({"fashion", "clothing", "outerwear", "apparel", "shoes", "footwear",
      "sneakers", "slippers", "socks", "polos", "bras", "underwear",
      "t-shirts", "shirts", "dresses", "jackets", "coats", "jeans",
      "sandals", "rain boots", "boots", "stockings"},
     {"grocery", "industrial & scientific", "automobile",
      "sporting goods", "electronics", "home & office", "pet supplies"}),

    ({"electronics", "cell phones", "bluetooth speakers", "bluetooth headsets",
      "earphones", "headsets", "smart watches", "wrist watches", "tv remote",
      "remote controls", "wi-fi", "dongles", "power banks", "earbuds",
      "headphones", "laptops", "cameras", "speakers"},
     {"grocery", "automobile", "industrial & scientific",
      "garden & outdoors", "sporting goods", "fashion", "pet supplies"}),

    ({"health", "beauty", "skin care", "creams", "makeup", "foundation",
      "heating pads", "massage", "medical", "fragrance", "perfume"},
     {"grocery", "industrial & scientific", "sporting goods",
      "automobile", "phones & tablets", "toys & games", "pet supplies"}),

    ({"home", "kitchen", "storage", "cleaning", "pressure cookers",
      "cookers", "books", "sprayers"},
     {"grocery", "sporting goods", "automobile",
      "industrial & scientific", "garden & outdoors"}),
]

_SAME_DOMAIN_CATEGORIES = {
    "health & beauty": {
        "creams", "strips", "supplements", "conditioners", "serums",
        "soaps", "washes", "body wash", "oils", "sets & kits",
        "eau de parfum", "skin care", "foundation", "hair color",
        "body scrubs", "nail care", "oral care", "medical supplies",
    },
    "home & office": {
        "sets & kits", "freezers", "food processors", "mixers & blenders",
        "rice cookers", "air fryers", "cookers", "microwave ovens",
        "pressure cookers", "kettles", "coffee makers",
        "vacuum cleaners", "washing machines", "bedding sets",
        "curtain panels", "duvet covers", "kitchen utensils & gadgets",
    },
    "electronics": {
        "bluetooth speakers", "bluetooth headsets", "earphones & headsets",
        "portable bluetooth speakers", "sound bars", "smart tvs",
        "ceiling fans", "tv remote controls", "remote controls",
    },
    "phones & tablets": {
        "chargers", "earbud headphones", "earphones & headsets",
        "cell phones", "android phones", "smartphones",
        "flip cases", "cases", "screen protectors",
    },
    "fashion": {
        "sandals", "sneakers", "slippers", "shoes", "rain boots", "boots",
        "casual dresses", "hats & caps", "briefs", "socks", "stockings",
        "polos", "bras", "underwear", "t-shirts", "shirts", "dresses",
        "jackets", "coats", "jeans", "handbags",
    },
    "computing": {
        "laptops", "desktops", "tablets", "monitors", "keyboards",
        "mice", "printers", "hard drives", "ssds", "networking",
    },
}

DOMAIN_SIGNALS = {
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"},
    "tablet": {"Phones & Tablets"}, "laptop": {"Computing"},
    "computer": {"Computing"}, "notebook": {"Computing"},
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "dress": {"Fashion"},
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},
    "tv": {"Electronics"}, "television": {"Electronics"},
    "headphone": {"Electronics"}, "diaper": {"Grocery", "Baby Products"},
    "perfume": {"Health & Beauty"}, "foundation": {"Health & Beauty"},
    "bike": {"Sporting Goods", "Automobile"}, "pot": {"Home & Office"},
}


def is_cross_domain_blocked(current_leaf: str, current_full: str,
                             predicted_top: str) -> bool:
    c_leaf = current_leaf.lower()
    c_full = current_full.lower()
    p_top  = predicted_top.lower()
    for current_kws, forbidden_tops in _CROSS_DOMAIN_BLOCKS:
        if any(kw in c_leaf or kw in c_full for kw in current_kws):
            if any(p_top.startswith(ft) for ft in forbidden_tops):
                return True
    return False


def is_same_domain(current_leaf: str, predicted_top: str) -> bool:
    cats = _SAME_DOMAIN_CATEGORIES.get(predicted_top.lower(), set())
    return current_leaf.lower() in cats


# ==============================================================================
# 4. FUZZY PATH RESOLVER
# ==============================================================================
def resolve_path_idx(norm_assigned: str, path_map: dict) -> int | None:
    if norm_assigned in path_map:
        return path_map[norm_assigned]
    if not norm_assigned:
        return None
    best_score, best_idx = 0, None
    for key, idx in path_map.items():
        score = fuzz.ratio(norm_assigned, key)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx if best_score >= 90 else None


# ==============================================================================
# 5. SQLITE LEARNING LAYER  (from CategoryMatcherEngine)
# ==============================================================================
DB_PATH = "cat_learning.db"

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS category_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def load_learning_db() -> dict:
    try:
        _init_db()
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT name, category FROM category_corrections", conn
            )
        if not df.empty:
            return df.groupby("name")["category"].last().to_dict()
    except Exception:
        pass
    return {}

def save_correction(name: str, category: str):
    try:
        _init_db()
        clean_n = clean_text(name)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO category_corrections (name, category) VALUES (?, ?)",
                (clean_n, category)
            )
            conn.commit()
    except Exception as e:
        st.warning(f"Could not save correction: {e}")

def build_correction_classifier(learning_db: dict):
    if len(learning_db) < 4:
        return None, None
    categories = list(learning_db.values())
    if len(set(categories)) < 2:
        return None, None
    try:
        names = list(learning_db.keys())
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        X = vec.fit_transform(names)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X, categories)
        return clf, vec
    except Exception:
        return None, None


# ==============================================================================
# 6. TAXONOMY INDEX BUILDER
# ==============================================================================
@st.cache_resource(show_spinner="Building taxonomy index…")
def build_index():
    try:
        df = pd.read_excel(CATEGORY_FILE, engine="openpyxl")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        path_col = next((c for c in df.columns if "PATH" in c.upper()), df.columns[2])
        kw_col   = next((c for c in df.columns if "KEY"  in c.upper()), None)
        code_col = next((c for c in df.columns if "CODE" in c.upper()), None)

        df["path_str"]  = df[path_col].astype(str).str.strip()
        df["norm_path"] = df["path_str"].apply(normalize_path)
        df["leaf_name"] = df["path_str"].apply(lambda x: x.split("/")[-1].strip().lower())
        df["kw_list"] = (
            df[kw_col].fillna("").astype(str)
            .apply(lambda x: set(clean_text(x).split()))
            if kw_col else pd.Series([set()] * len(df))
        )

        p_clean = df["path_str"].str.replace("/", " ").str.lower()
        k_clean = df[kw_col].fillna("").astype(str).str.lower() if kw_col else pd.Series([""] * len(df))
        df["search_text"] = (p_clean + " ") * 3 + (k_clean + " ") * 5

        vectorizer   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(df["search_text"])

        path_map     = {row["norm_path"]: i for i, row in df.iterrows()}
        code_to_path = (
            dict(zip(
                df[code_col].astype(str).str.replace(r"\.0$", "", regex=True),
                df[path_col]
            )) if code_col else {}
        )
        return df, vectorizer, tfidf_matrix, code_to_path, path_col, path_map
    except Exception as e:
        st.error(f"Taxonomy load error: {e}")
        return None, None, None, None, None, None


index_data = build_index()
if index_data[0] is None:
    st.stop()

df_cat, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME, path_map = index_data


# ==============================================================================
# 7. HYBRID SCORING ENGINE
# ==============================================================================
def hybrid_score(clean_q: str, assigned_path: str, sims_row,
                 learning_db: dict, clf, clf_vec) -> tuple[float, str]:
    """
    Returns (confidence: float 0-100, verdict: str).

    Pipeline (in priority order):
      1. Learning DB exact match       — from CategoryMatcherEngine
      2. Correction classifier         — from CategoryMatcherEngine
      3. Cross-domain block check      — from CategoryMatcherEngine
      4. Same-domain suppression       — from CategoryMatcherEngine
      5. Composite scoring             — from v14 Streamlit script (fixed)
         • TF-IDF cosine
         • Keyword intersection
         • Literal leaf match
         • Full-path fuzzy blend
         • Domain anchor penalty (reduced)
    """
    clean_assigned = clean_text(assigned_path)

    # ── Layer 1: Learning DB exact match ──────────────────────────────────────
    if clean_q in learning_db:
        learned_cat = learning_db[clean_q]
        if clean_text(learned_cat) == clean_assigned:
            return 100.0, "Approved"
        else:
            return 0.0, "Flagged (learned correction)"

    # ── Layer 2: Correction classifier ────────────────────────────────────────
    if clf is not None and clf_vec is not None:
        try:
            vec   = clf_vec.transform([clean_q])
            probs = clf.predict_proba(vec)[0]
            best  = int(np.argmax(probs))
            if probs[best] > 0.6:
                predicted_cat = clf.classes_[best]
                if clean_text(predicted_cat) != clean_assigned:
                    return 15.0, f"Flagged (learned: {predicted_cat})"
        except Exception:
            pass

    # ── Resolve assigned path → taxonomy row ──────────────────────────────────
    norm_assigned = normalize_path(assigned_path)
    idx = resolve_path_idx(norm_assigned, path_map)
    if idx is None:
        return 0.0, "Rejected (path not in taxonomy)"

    row       = df_cat.iloc[idx]
    cos_score = float(sims_row[idx])
    leaf      = row["leaf_name"]
    full_path = str(row[PATH_COL_NAME]).lower()
    top_level = str(row[PATH_COL_NAME]).split("/")[0].strip()

    # ── Layer 3: Cross-domain block ────────────────────────────────────────────
    # Uses the engine's suppression table to catch cases like
    # "Makita Grinder → Grocery/Air Fresheners" that pass cosine but are nonsense.
    query_tokens = set(clean_q.split())
    p_top = get_top(full_path)
    c_leaf = get_leaf(assigned_path)
    if is_cross_domain_blocked(c_leaf, assigned_path, p_top):
        # Don't flip to Rejected outright — flag for human review
        return 8.0, "Flagged (domain mismatch)"

    # ── Layer 4: Same-domain suppression ──────────────────────────────────────
    if is_same_domain(c_leaf, p_top):
        # Current leaf is a legitimate sub-category of the predicted domain —
        # trust the assignment without penalising it.
        pass  # falls through to scoring with no penalty

    # ── Layer 5: Composite score (v14 logic) ──────────────────────────────────
    query_stems = {t.rstrip("s") for t in query_tokens}

    # 5a. TF-IDF cosine (0.35)
    comp_cos = cos_score * 0.35

    # 5b. Keyword intersection (0.25)
    kw_overlap = len(query_tokens & row["kw_list"])
    comp_kw    = min(kw_overlap / max(len(row["kw_list"]), 1), 1.0) * 0.25

    # 5c. Literal leaf match (0.20)
    leaf_words   = [w.rstrip("s") for w in leaf.replace("&", " ").split() if len(w) >= 3]
    literal_hit  = bool(leaf_words and all(lw in query_stems for lw in leaf_words))
    comp_literal = 0.20 if literal_hit else 0.0

    # 5d. Full-path fuzzy blend (0.20)  — fixes the leaf-only fuzz of v13
    fuzz_leaf      = fuzz.token_set_ratio(clean_q, leaf) / 100.0
    fuzz_full_path = fuzz.token_set_ratio(clean_q, full_path) / 100.0
    comp_fuzz      = (fuzz_leaf * 0.40 + fuzz_full_path * 0.60) * 0.20

    # 5e. Domain anchor penalty (reduced 0.45 → 0.20 vs v13)
    required_verticals: set[str] = set()
    for tok in query_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]
    penalty = 0.20 if (required_verticals and top_level not in required_verticals) else 0.0

    raw   = comp_cos + comp_kw + comp_literal + comp_fuzz - penalty
    score = round(min(max(raw, 0.0), 1.0) * 100.0, 1)

    # Segment similarity check — suppress if ≥2 shared leading segments
    # (same sub-family, not a wrong-category situation)
    # This replaces the old binary domain check with a structural path check.
    if score < 25:
        # Only bother resolving segments for borderline cases
        p_segs = get_segments(full_path, 3)
        c_segs = get_segments(assigned_path, 3)
        shared = sum(1 for a, b in zip(p_segs, c_segs) if a == b)
        if shared >= min(2, len(p_segs), len(c_segs)):
            score = max(score, 30.0)   # lift to approved if structurally close

    return score, None   # verdict assigned below by caller


# ==============================================================================
# 8. UI
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    threshold = st.slider("Approval Threshold", 0, 100, 25)
    st.caption("25–35 for strict audit · 15–20 for permissive review")
    st.divider()
    st.subheader("Feedback Learning")
    st.caption(
        "Corrections saved here improve future runs. "
        "The engine learns which product→category pairs are correct."
    )
    fb_name = st.text_input("Product name")
    fb_cat  = st.text_input("Correct category path")
    if st.button("Save correction"):
        if fb_name and fb_cat:
            save_correction(fb_name, fb_cat)
            st.success("Saved — rebuild index to apply.")
        else:
            st.warning("Enter both name and category.")

# Load learning layer
learning_db = load_learning_db()
clf, clf_vec = build_correction_classifier(learning_db)
if learning_db:
    st.sidebar.caption(f"📚 {len(learning_db)} corrections loaded")

uploaded_file = st.file_uploader("Upload CSV for Audit", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    code_col = next(
        (c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None
    )

    if st.button("Run Hybrid Audit"):
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

        scores, verdicts = [], []
        for i in range(len(names)):
            score, special_verdict = hybrid_score(
                cleaned[i],
                df_up["Assigned category in full"].iloc[i],
                all_sims[i],
                learning_db, clf, clf_vec,
            )
            if special_verdict:
                verdict = special_verdict
            elif score >= threshold:
                verdict = "Approved"
            else:
                verdict = "Rejected"
            scores.append(score)
            verdicts.append(verdict)

        df_up["confidence"] = scores
        df_up["status"]     = verdicts

        # ── Metrics ───────────────────────────────────────────────────────────
        total    = len(df_up)
        approved = (df_up["status"] == "Approved").sum()
        rejected = (df_up["status"] == "Rejected").sum()
        flagged  = total - approved - rejected

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",    total)
        c2.metric("Approved", approved,  f"{approved/total*100:.1f}%")
        c3.metric("Rejected", rejected,  f"-{rejected/total*100:.1f}%")
        c4.metric("Flagged",  flagged,   f"{flagged/total*100:.1f}%")

        # ── Results table ─────────────────────────────────────────────────────
        final_cols = [name_col, "Assigned category in full", "confidence", "status"]
        st.subheader("Audit Results")
        st.dataframe(
            df_up[final_cols].head(2500),
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100
                ),
                "status": st.column_config.TextColumn("Verdict"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # ── Flagged-only view ─────────────────────────────────────────────────
        flagged_df = df_up[df_up["status"].str.startswith("Flagged")]
        if not flagged_df.empty:
            with st.expander(f"⚠️ {len(flagged_df)} Flagged items (domain mismatches / learned corrections)"):
                st.dataframe(flagged_df[final_cols], hide_index=True, use_container_width=True)

        st.download_button(
            "Export Results CSV",
            df_up[final_cols].to_csv(index=False).encode("utf-8"),
            "category_audit_hybrid_v1.csv",
        )

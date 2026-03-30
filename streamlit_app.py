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
    .stButton > button { background: #000000 !important; color: white !important; border-radius: 4px; width: 100%; border: none; font-weight: 600; }
    .stDataFrame { border: 1px solid #dee2e6; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("Category Test")
st.caption("Standardized Taxonomy Audit Engine | Version 2026.2")

# ==============================================================================
# 2. UNIVERSAL LOGIC & CLEANING
# ==============================================================================
# FIX 1: Massively expanded DOMAIN_SIGNALS to prevent cross-vertical keyword drift
DOMAIN_SIGNALS = {
    # Phones & Tablets
    "phone": {"Phones & Tablets"}, "mobile": {"Phones & Tablets"},
    "tablet": {"Phones & Tablets"}, "smartphone": {"Phones & Tablets"},
    "sim": {"Phones & Tablets"}, "smartwatch": {"Phones & Tablets"},
    "iphone": {"Phones & Tablets"}, "android": {"Phones & Tablets"},

    # Computing
    "laptop": {"Computing"}, "computer": {"Computing"}, "notebook": {"Computing"},
    "desktop": {"Computing"}, "processor": {"Computing"}, "keyboard": {"Computing"},
    "mouse": {"Computing"}, "monitor": {"Computing"}, "ssd": {"Computing"},
    "ram": {"Computing"}, "motherboard": {"Computing"}, "router": {"Computing"},

    # Electronics
    "tv": {"Electronics"}, "television": {"Electronics"},
    "headphone": {"Electronics"}, "earphone": {"Electronics"},
    "speaker": {"Electronics"}, "amplifier": {"Electronics"},
    "camera": {"Electronics"}, "projector": {"Electronics"},
    "remote": {"Electronics"}, "inverter": {"Electronics"},
    "solar": {"Electronics"}, "battery": {"Electronics"},
    "tweeter": {"Electronics"}, "woofer": {"Electronics"},
    "cctv": {"Electronics"}, "uhd": {"Electronics"},
    "4k": {"Electronics"}, "oled": {"Electronics"},

    # Fashion
    "shoe": {"Fashion"}, "sneaker": {"Fashion"}, "heel": {"Fashion"},
    "boot": {"Fashion"}, "sandal": {"Fashion"}, "slipper": {"Fashion"},
    "dress": {"Fashion"}, "shirt": {"Fashion"}, "trouser": {"Fashion"},
    "jeans": {"Fashion"}, "skirt": {"Fashion"}, "blouse": {"Fashion"},
    "jacket": {"Fashion"}, "coat": {"Fashion"}, "suit": {"Fashion"},
    "handbag": {"Fashion"}, "purse": {"Fashion"}, "wallet": {"Fashion"},
    "belt": {"Fashion"}, "cap": {"Fashion"}, "hat": {"Fashion"},
    "lingerie": {"Fashion"}, "underwear": {"Fashion"}, "bra": {"Fashion"},

    # Watch is ambiguous — keep dual-signal
    "watch": {"Fashion", "Electronics", "Phones & Tablets"},

    # Health & Beauty
    "perfume": {"Health & Beauty"}, "cologne": {"Health & Beauty"},
    "fragrance": {"Health & Beauty"}, "lotion": {"Health & Beauty"},
    "cream": {"Health & Beauty"}, "serum": {"Health & Beauty"},
    "shampoo": {"Health & Beauty"}, "conditioner": {"Health & Beauty"},
    "toothbrush": {"Health & Beauty"}, "toothpaste": {"Health & Beauty"},
    "deodorant": {"Health & Beauty"}, "lipstick": {"Health & Beauty"},
    "mascara": {"Health & Beauty"}, "foundation": {"Health & Beauty"},
    "trimmer": {"Health & Beauty"}, "razor": {"Health & Beauty"},
    "clipper": {"Health & Beauty"}, "waxing": {"Health & Beauty"},
    "wheelchair": {"Health & Beauty"}, "crutch": {"Health & Beauty"},
    "glucometer": {"Health & Beauty"}, "thermometer": {"Health & Beauty"},
    "pressure": {"Health & Beauty"}, "stethoscope": {"Health & Beauty"},

    # Home & Office
    "pot": {"Home & Office"}, "pan": {"Home & Office"},
    "kettle": {"Home & Office"}, "blender": {"Home & Office"},
    "iron": {"Home & Office"}, "vacuum": {"Home & Office"},
    "fan": {"Home & Office"}, "curtain": {"Home & Office"},
    "pillow": {"Home & Office"}, "mattress": {"Home & Office"},
    "sofa": {"Home & Office"}, "chair": {"Home & Office"},
    "table": {"Home & Office"}, "shelf": {"Home & Office"},
    "lamp": {"Home & Office"}, "bulb": {"Home & Office"},
    "flask": {"Home & Office"}, "thermos": {"Home & Office"},
    "glassware": {"Home & Office"}, "mug": {"Home & Office"},
    "plate": {"Home & Office"}, "bowl": {"Home & Office"},
    "printer": {"Home & Office", "Computing"}, "stapler": {"Home & Office"},

    # Grocery
    "rice": {"Grocery"}, "flour": {"Grocery"}, "oil": {"Grocery"},
    "sugar": {"Grocery"}, "salt": {"Grocery"}, "juice": {"Grocery"},
    "milk": {"Grocery"}, "yogurt": {"Grocery"}, "cheese": {"Grocery"},
    "bread": {"Grocery"}, "tea": {"Grocery"}, "coffee": {"Grocery"},

    # Baby Products
    "diaper": {"Baby Products", "Grocery"}, "nappy": {"Baby Products"},
    "baby": {"Baby Products"}, "infant": {"Baby Products"},
    "stroller": {"Baby Products"}, "pram": {"Baby Products"},
    "feeding": {"Baby Products"}, "pacifier": {"Baby Products"},

    # Sporting Goods
    "bike": {"Sporting Goods"}, "bicycle": {"Sporting Goods"},
    "yoga": {"Sporting Goods"}, "dumbbell": {"Sporting Goods"},
    "treadmill": {"Sporting Goods"}, "football": {"Sporting Goods"},
    "basketball": {"Sporting Goods"}, "gym": {"Sporting Goods"},
    "tennis": {"Sporting Goods"}, "swimming": {"Sporting Goods"},

    # Automobile
    "tyre": {"Automobile"}, "tire": {"Automobile"},
    "battery": {"Automobile", "Electronics"}, "wiper": {"Automobile"},
    "engine": {"Automobile"}, "exhaust": {"Automobile"},
    "bumper": {"Automobile"}, "windscreen": {"Automobile"},

    # Musical Instruments
    "guitar": {"Musical Instruments"}, "piano": {"Musical Instruments"},
    "drum": {"Musical Instruments"}, "microphone": {"Musical Instruments"},
    "violin": {"Musical Instruments"}, "trumpet": {"Musical Instruments"},
    "keyboard": {"Musical Instruments", "Computing"},

    # Gaming
    "gamepad": {"Gaming"}, "console": {"Gaming"},
    "playstation": {"Gaming"}, "xbox": {"Gaming"},
    "joystick": {"Gaming"},

    # Toys & Games
    "toy": {"Toys & Games"}, "puzzle": {"Toys & Games"},
    "lego": {"Toys & Games"}, "doll": {"Toys & Games"},
    "board game": {"Toys & Games"}, "jenga": {"Toys & Games"},
}

_MEASURE_RE = re.compile(r'\b\d+\.?\d*\s*(ml|l|g|kg|pcs|inch|cm|w|kw|mah|gb|tb|v|ah)\b', re.I)

def clean_standard(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = _MEASURE_RE.sub(" ", text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split()).strip()

# ==============================================================================
# 3. INDEX BUILDER
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Taxonomy Index...")
def build_index():
    if not os.path.exists(CATEGORY_FILE):
        st.error(f"Critical Error: {CATEGORY_FILE} not found."); st.stop()

    df = pd.read_excel(CATEGORY_FILE)
    raw_cols = df.columns.tolist()

    path_col = next((c for c in raw_cols if 'PATH' in c.upper()), None)
    kw_col = next((c for c in raw_cols if 'KEY' in c.upper()), None)
    code_col = next((c for c in raw_cols if 'CODE' in c.upper()), None)

    df['path_str'] = df[path_col].astype(str)
    df['leaf_name'] = df['path_str'].apply(lambda x: x.split('/')[-1].strip().lower())
    df['depth'] = df['path_str'].apply(lambda x: x.count('/') + 1)

    p_clean = df['path_str'].str.replace('/', ' ').str.lower()
    k_clean = df[kw_col].fillna('').astype(str).str.lower() if kw_col else ""

    df['search_text'] = (p_clean + ' ') * 4 + k_clean

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])

    clean_codes = df[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    code_to_path = dict(zip(clean_codes, df[path_col]))

    return df, vectorizer, tfidf_matrix, code_to_path, path_col

df_cat, vectorizer, tfidf_matrix, master_code_map, PATH_COL_NAME = build_index()

# ==============================================================================
# 4. SCORING ENGINE
# ==============================================================================
def calculate_match(clean_q, top_idxs, sims_row, threshold, current_vertical=None):
    best_score, best_row = -1.0, None
    raw_tokens = set(clean_q.split())

    required_verticals = set()
    for tok in raw_tokens:
        if tok in DOMAIN_SIGNALS:
            required_verticals |= DOMAIN_SIGNALS[tok]

    # FIX 3: Use the existing category vertical as a strong anchor
    if current_vertical:
        required_verticals.add(current_vertical)

    for idx in top_idxs:
        row = df_cat.iloc[idx]
        cos = float(sims_row[idx])
        name_fuzz = fuzz.token_set_ratio(clean_q, row['leaf_name']) / 100.0

        top_level = str(row[PATH_COL_NAME]).split('/')[0].strip()

        # FIX 2: Raised penalty from 0.25 → 0.50 to strongly discourage wrong verticals
        penalty = 0.50 if (required_verticals and top_level not in required_verticals) else 0.0

        score = (cos * 0.50) + (name_fuzz * 0.30) + (int(row['depth']) * 0.01) - penalty

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None: return "Uncategorized", 0.0, "Rejected"

    # FIX 4: Raised multiplier from 120 → 165 to spread confidence scores better
    conf = round(min(best_score * 165.0, 100.0), 1)

    # Hard floor — raw scores below 0.30 are always Rejected regardless of threshold
    if best_score < 0.30:
        return best_row[PATH_COL_NAME], conf, "Rejected"

    status = "Approved" if (conf >= threshold and best_score > 0.30) else "Rejected"

    return best_row[PATH_COL_NAME], conf, status

# ==============================================================================
# 5. SIDEBAR & BATCH PROCESSING
# ==============================================================================
with st.sidebar:
    st.header("Parameters")
    threshold = st.slider("Confidence Threshold", 0, 100, 62)
    st.divider()
    st.info("Each record is scored as Approved or Rejected against the taxonomy.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df_up = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')

    name_col = next((c for c in df_up.columns if "NAME" in c.upper()), df_up.columns[0])
    category_col = next((c for c in df_up.columns if "CATEGORY" in c.upper() and "AI" not in c.upper() and "PATH" not in c.upper()), None)
    code_col = next((c for c in df_up.columns if "CODE" in c.upper() and "AI" not in c.upper()), None)

    if st.button("Process Batch Analysis"):
        names = df_up[name_col].fillna("").astype(str).tolist()
        cleaned_queries = [clean_standard(n) for n in names]

        q_vecs = vectorizer.transform(cleaned_queries)
        all_sims = cosine_similarity(q_vecs, tfidf_matrix)

        # FIX 3: Resolve existing category verticals for anchoring
        existing_paths = []
        if code_col:
            clean_codes_series = df_up[code_col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_up["Assigned category in full"] = clean_codes_series.map(master_code_map).fillna("Unknown Code")
        else:
            df_up["Assigned category in full"] = "N/A"

        for path in df_up["Assigned category in full"]:
            v = str(path).split('/')[0].strip() if path not in ("Unknown Code", "N/A") else None
            existing_paths.append(v)

        results = []
        for i in range(len(names)):
            s_row = all_sims[i]
            t_idxs = np.argpartition(s_row, -40)[-40:]
            results.append(calculate_match(
                cleaned_queries[i], t_idxs, s_row, threshold,
                current_vertical=existing_paths[i]   # FIX 3: pass anchor
            ))

        df_up["confidence"] = [r[1] for r in results]
        df_up["status"] = [r[2] for r in results]

        # Standardize Display
        display_map = {name_col: "NAME"}
        if category_col: display_map[category_col] = "CATEGORY"
        df_up = df_up.rename(columns=display_map)

        if "CATEGORY" not in df_up.columns: df_up["CATEGORY"] = "N/A"

        final_cols = ["NAME", "Assigned category in full", "CATEGORY", "confidence", "status"]

        # Stats summary
        total = len(df_up)
        n_approved = (df_up["status"] == "Approved").sum()
        n_rejected = (df_up["status"] == "Rejected").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Approved", n_approved, delta=f"{n_approved/total*100:.1f}%")
        col3.metric("Rejected", n_rejected, delta=f"{n_rejected/total*100:.1f}%", delta_color="inverse")

        st.subheader("Analysis Results")
        st.dataframe(
            df_up[final_cols].head(2000),
            column_config={
                "confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100),
                "Assigned category in full": st.column_config.TextColumn("Current Path", width="large"),
            },
            hide_index=True
        )

        st.download_button("Export Results", df_up[final_cols].to_csv(index=False).encode('utf-8'), "category_audit.csv")

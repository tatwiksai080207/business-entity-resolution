"""
Text preprocessing and multilingual normalization module.
Handles entity suffix standardization across jurisdictions (US, India, France)
and address token cleaning.
"""

import re
import pandas as pd
from unidecode import unidecode

LEGAL_SUFFIX_PATTERNS = [
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\blimited\b", "ltd"),
    (r"\bprivate\b", "pvt"),
    (r"\bcompany\b", "co"),
    (r"\bllc\b", "llc"),
    (r"\bl\.l\.c\.?\b", "llc"),
    (r"\bllp\b", "llp"),
    (r"\bl\.l\.p\.?\b", "llp"),
    (r"\binc\.?\b", "inc"),
    (r"\bltd\.?\b", "ltd"),
    (r"\bpvt\.?\b", "pvt"),
    (r"\bcorp\.?\b", "corp"),
    (r"\bco\.?\b", "co"),
    # French jurisdictions
    (r"\bs\.?a\.?s\.?u\.?\b", "sasu"),
    (r"\bs\.?a\.?s\.?\b", "sas"),
    (r"\bs\.?a\.?r\.?l\.?\b", "sarl"),
    (r"\be\.?u\.?r\.?l\.?\b", "eurl"),
    (r"\bs\.?a\.?\b", "sa"),
    (r"\bs\.?c\.?i\.?\b", "sci"),
]

ADDRESS_ABBREVIATIONS = [
    (r"\bst\b", "street"),
    (r"\brd\b", "road"),
    (r"\bave\b", "avenue"),
    (r"\bblvd\b", "boulevard"),
    (r"\bdr\b", "drive"),
    (r"\bct\b", "court"),
    (r"\bln\b", "lane"),
    (r"\bpkwy\b", "parkway"),
    (r"\bhwy\b", "highway"),
    (r"\bapt\b", "apartment"),
    (r"\bste\b", "suite"),
]


def normalize_name(name: str) -> str:
    """
    Standardize business names by removing noise prefixes,
    normalizing legal entity suffixes, and collapsing whitespace.
    """
    if pd.isna(name) or str(name).strip() == "":
        return ""
    text = unidecode(str(name)).lower().strip()
    text = re.sub(r"^[-\s\.\*#]+", "", text)

    for pattern, replacement in LEGAL_SUFFIX_PATTERNS:
        text = re.sub(pattern, replacement, text)

    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(address: str) -> str:
    """
    Standardize address strings by expanding roadway types,
    stripping missing data placeholders, and unifying delimiters.
    """
    if pd.isna(address) or str(address).strip().lower() in ("", "nan", "null", "none", "n/a"):
        return ""
    text = unidecode(str(address)).lower().strip()

    for pattern, replacement in ADDRESS_ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\bnull\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce dual representations for all entities:
    1. Normalized ASCII text for sparse indexing and character n grams.
    2. Raw multilingual text for dense transformer embeddings.
    """
    df = df.copy()

    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
    else:
        df["country"] = "Unknown"

    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    df["name_addr"] = (df["name_norm"] + " " + df["addr_norm"]).str.strip()

    raw_name = df["business_name"].fillna("").astype(str).str.strip()
    raw_addr = df["business_address"].fillna("").astype(str).str.strip()
    df["embed_text"] = (raw_name + " " + raw_addr).str.strip()

    return df

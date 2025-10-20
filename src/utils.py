import re
import pandas as pd

def basic_clean_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"txn|transaction|debited|credited|rs\.?|inr", " ", t)
    t = re.sub(r"\d{4,}", " ", t)           # remove long numbers/order ids
    t = re.sub(r"[^a-z\s]", " ", t)         # keep letters/spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

def ensure_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected {required}.")
    return df.copy()

def coerce_types(df: pd.DataFrame):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["description"] = out["description"].astype(str).fillna("")
    return out

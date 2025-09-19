import re
import datetime as dt
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from constants import MASTER_PLAYERS

def parse_date_from_filename(name: str) -> dt.date | None:
    m = re.search(r'(\d{1,2})[._-](\d{1,2})', name)
    if not m:
        return None
    year = dt.date.today().year
    month, day = int(m.group(1)), int(m.group(2))
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None

def coalesce_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = 0

def clean_and_filter_players(df: pd.DataFrame, name_col="Unnamed: 0") -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()
    if name_col not in d.columns:
        name_col = d.columns[0]
    d[name_col] = d[name_col].astype(str).str.strip()

    def map_name(raw: str):
        if not raw or raw.lower() in ("nan", "none"):
            return None
        s = raw.strip()
        s = re.sub(r'^\s*#?\d+\s+', '', s)
        # map defensive summary rows to friendly team names:
        # D-Team 1 -> "Green", D-Team 2 -> "White"
        if re.match(r'(?i)^\s*d[- ]?team\s*1\b', s):
            return "Green"
        if re.match(r'(?i)^\s*d[- ]?team\s*2\b', s):
            return "White"
        s_low = s.lower()
        for master in MASTER_PLAYERS:
            if s_low == master.lower():
                return master
        for master in MASTER_PLAYERS:
            if master.lower() in s_low:
                return master
        first_token = s_low.split()[0] if s_low.split() else ""
        for master in MASTER_PLAYERS:
            if master.lower().startswith(first_token) or first_token.startswith(master.lower()):
                return master
        return None

    d["Player"] = d[name_col].apply(map_name)
    d["Player"] = d["Player"].astype(object).apply(lambda x: x.strip() if isinstance(x, str) else x)
    d = d[d["Player"].notna()].copy()
    if name_col in d.columns and name_col != "Player":
        d = d.drop(columns=[name_col])
    return d

def read_folder(folder_name: str, processor) -> pd.DataFrame:
    base = Path(__file__).parent
    folder = base / folder_name
    if not folder.exists():
        st.info(f"Folder not found: {folder}")
        return pd.DataFrame()

    dfs = []
    for p in sorted(folder.glob("*.csv")):
        try:
            df = processor(p, p.name)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception as e:
            st.warning(f"Skipping {p.name}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def list_offense_files() -> list[Path]:
    base = Path(__file__).parent
    folder = base / "Offense"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"))

def aggregate_by_player(df_off, df_def, start=None, end=None) -> pd.DataFrame:
    off = df_off.copy() if isinstance(df_off, pd.DataFrame) else pd.DataFrame()
    de  = df_def.copy()  if isinstance(df_def, pd.DataFrame) else pd.DataFrame()

    expected_off = ["Date", "Player", "OffPoints", "OffPoss"]
    expected_def = ["Date", "Player", "DefPoints", "DefPoss"]
    for c in expected_off:
        if c not in off.columns:
            if c == "Player":
                off[c] = pd.Series(dtype="object")
            elif c == "Date":
                off[c] = pd.Series(dtype="object")
            else:
                off[c] = pd.Series(dtype="float")
    for c in expected_def:
        if c not in de.columns:
            if c == "Player":
                de[c] = pd.Series(dtype="object")
            elif c == "Date":
                de[c] = pd.Series(dtype="object")
            else:
                de[c] = pd.Series(dtype="float")

    if start:
        off = off[(off["Date"].isna()) | (off["Date"] >= start)]
        de  = de[(de["Date"].isna())  | (de["Date"] >= start)]
    if end:
        off = off[(off["Date"].isna()) | (off["Date"] <= end)]
        de  = de[(de["Date"].isna())  | (de["Date"] <= end)]

    off_agg = off.groupby("Player", as_index=False)[["OffPoints", "OffPoss"]].sum()
    def_agg = de.groupby("Player", as_index=False)[["DefPoints", "DefPoss"]].sum()

    all_players = pd.DataFrame({"Player": MASTER_PLAYERS})
    out = all_players.merge(off_agg, on="Player", how="left").merge(def_agg, on="Player", how="left")
    for c in ["OffPoints", "OffPoss", "DefPoints", "DefPoss"]:
        out[c] = out[c].fillna(0).astype(int)

    # round ratings to 1 decimal
    out["OffRating"] = np.where(out["OffPoss"] > 0, (out["OffPoints"] / out["OffPoss"] * 100).round(1), np.nan)
    out["DefRating"] = np.where(out["DefPoss"] > 0, (out["DefPoints"] / out["DefPoss"] * 100).round(1), np.nan)
    out["NetRating"] = out["OffRating"] - out["DefRating"]
    out["AvgPoss"] = ((out["OffPoss"] + out["DefPoss"]) / 2).round().astype(int)

    return out
import re
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from utils import clean_and_filter_players, coalesce_cols

def process_offense_file(file, filename: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = clean_and_filter_players(df)
    shot_cols = [
        "shot chart:RZ +",
        "shot chart:TT +",
        "shot chart:IO3 +",
        "shot chart:NIO3 +",
        "shot chart:FT 2",
        "shot chart:FT 3",
    ]
    coalesce_cols(df, ["offense_lab"] + shot_cols)
    for c in shot_cols + ["offense_lab"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    ft2 = df.get("shot chart:FT 2", 0.0)
    ft3 = df.get("shot chart:FT 3", 0.0)
    df["OffPoints"] = (
        2.0 * df["shot chart:RZ +"] +
        2.0 * df["shot chart:TT +"] +
        3.0 * df["shot chart:IO3 +"] +
        3.0 * df["shot chart:NIO3 +"] +
        1.5 * ft2 +
        2.25 * ft3
    )
    if "offense_lab" in df.columns:
        df = df.rename(columns={"offense_lab": "OffPoss"})
    else:
        df["OffPoss"] = 0
    df["Date"] = parse_date_from_filename(filename) if 'parse_date_from_filename' in globals() else None
    grp_cols = ["Date", "Player"]
    numeric_cols = ["OffPoints", "OffPoss"]
    df = df.groupby(grp_cols, as_index=False)[numeric_cols].sum()
    return df[["Date", "Player", "OffPoints", "OffPoss"]]

def compute_offense_success_rates_for_paths(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    stats = []
    plus_totals = {}
    minus_totals = {}
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        mask = df[name_col].astype(str).str.strip().str.endswith("Stat")
        players_df = df[mask].copy()
        if players_df.empty:
            continue
        file_stats = []
        for col in df.columns[1:]:
            base = col.split(":")[-1].strip()
            base = re.sub(r'\s*[+-]$', '', base).strip()
            if base:
                file_stats.append(base)
        for s in file_stats:
            if s not in stats:
                stats.append(s)
        for _, row in players_df.iterrows():
            raw = str(row[name_col]).strip()
            player = raw.strip()
            plus_totals.setdefault(player, {})
            minus_totals.setdefault(player, {})
            for col in df.columns[1:]:
                base = col.split(":")[-1].strip()
                sign = None
                m = re.search(r'([+-])\s*$', base)
                if m:
                    sign = m.group(1)
                    base = re.sub(r'\s*[+-]$', '', base).strip()
                else:
                    if col.strip().endswith("+"):
                        sign = "+"
                        base = col.split(":")[-1].strip()
                        base = re.sub(r'\s*\+\s*$', '', base).strip()
                    elif col.strip().endswith("-"):
                        sign = "-"
                        base = col.split(":")[-1].strip()
                        base = re.sub(r'\s*\-\s*$', '', base).strip()
                if not base:
                    continue
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+" or "+" in col:
                    plus_totals[player][base] = plus_totals[player].get(base, 0) + val
                elif sign == "-" or "-" in col:
                    minus_totals[player][base] = minus_totals[player].get(base, 0) + val
                else:
                    continue
    if not stats:
        return pd.DataFrame()
    rows = []
    all_players = sorted(set(list(plus_totals.keys()) + list(minus_totals.keys())))
    for player in all_players:
        row = {"Player": player}
        for s in stats:
            plus = plus_totals.get(player, {}).get(s, 0)
            minus = minus_totals.get(player, {}).get(s, 0)
            denom = plus + minus
            if denom > 0:
                rate = (plus / denom) * 100.0
            else:
                rate = np.nan
            # round success rates to 1 decimal
            row[s] = round(rate, 1) if not np.isnan(rate) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("Player").reset_index(drop=True)
    cols = ["Player"] + sorted(stats)
    out = out.reindex(columns=cols)
    return out

def compute_offense_team_summary(paths: list[Path], display_stats: list[str]) -> pd.DataFrame:
    if not paths or not display_stats:
        return pd.DataFrame()
    total_plus = {}
    total_minus = {}
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        mask = df[name_col].astype(str).str.strip().str.endswith("Stat")
        players_df = df[mask].copy()
        if players_df.empty:
            continue
        header_info = []
        for col in df.columns[1:]:
            base = col.split(":")[-1].strip()
            m = re.search(r'([+-])\s*$', base)
            if m:
                sign = m.group(1)
                base = re.sub(r'\s*[+-]$', '', base).strip()
            else:
                if col.strip().endswith("+"):
                    sign = "+"
                    base = re.sub(r'\s*\+\s*$', '', base).strip()
                elif col.strip().endswith("-"):
                    sign = "-"
                    base = re.sub(r'\s*\-\s*$', '', base).strip()
                else:
                    continue
            header_info.append((col, base, sign))
        for _, row in players_df.iterrows():
            for col, base, sign in header_info:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+":
                    total_plus[base] = total_plus.get(base, 0) + val
                elif sign == "-":
                    total_minus[base] = total_minus.get(base, 0) + val
    row = {"Player": "Team"}
    for s in display_stats:
        if s == "Pull Behind":
            plus = total_plus.get("Pull Behind", 0)
            denom = total_minus.get("Pull Behind", 0) + total_minus.get("Pull Ahead", 0)
            rate = (plus / denom) * 100.0 if denom > 0 else np.nan
        else:
            plus = total_plus.get(s, 0)
            minus = total_minus.get(s, 0)
            denom = plus + minus
            rate = (plus / denom) * 100.0 if denom > 0 else np.nan
        # round team summary percents to 1 decimal
        row[s] = round(rate, 1) if not np.isnan(rate) else np.nan
    team_df = pd.DataFrame([row])
    cols = ["Player"] + [c for c in display_stats if c in team_df.columns]
    team_df = team_df.reindex(columns=cols)
    return team_df

def compute_offense_player_counts(paths: list[Path], display_stats: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    plus_totals = {}
    minus_totals = {}
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        mask = df[name_col].astype(str).str.strip().str.endswith("Stat")
        players_df = df[mask].copy()
        if players_df.empty:
            continue
        header_info = []
        for col in df.columns[1:]:
            base = col.split(":")[-1].strip()
            m = re.search(r'([+-])\s*$', base)
            if m:
                sign = m.group(1)
                base = re.sub(r'\s*[+-]$', '', base).strip()
            else:
                if col.strip().endswith("+"):
                    sign = "+"
                    base = re.sub(r'\s*\+\s*$', '', base).strip()
                elif col.strip().endswith("-"):
                    sign = "-"
                    base = re.sub(r'\s*\-\s*$', '', base).strip()
                else:
                    continue
            header_info.append((col, base, sign))
        for _, row in players_df.iterrows():
            raw = str(row[name_col]).strip()
            player = raw
            plus_totals.setdefault(player, {})
            minus_totals.setdefault(player, {})
            for col, base, sign in header_info:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+":
                    plus_totals[player][base] = plus_totals[player].get(base, 0) + val
                elif sign == "-":
                    minus_totals[player][base] = minus_totals[player].get(base, 0) + val
    players = sorted(set(list(plus_totals.keys()) + list(minus_totals.keys())))
    rows = []
    for player in players:
        row = {"Player": player}
        for s in display_stats:
            if s == "Pull Behind":
                plus = plus_totals.get(player, {}).get("Pull Behind", 0)
                minus_pb = minus_totals.get(player, {}).get("Pull Behind", 0)
                minus_pa = minus_totals.get(player, {}).get("Pull Ahead", 0)
                total = int(plus + minus_pb + minus_pa)
            else:
                plus = plus_totals.get(player, {}).get(s, 0)
                minus = minus_totals.get(player, {}).get(s, 0)
                total = int(plus + minus)
            row[f"{s} Opportunities"] = total
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = ["Player"] + [f"{s} Opportunities" for s in display_stats]
    out = out.reindex(columns=cols, fill_value=0)
    return out

def compute_offense_team_counts(paths: list[Path], display_stats: list[str]) -> pd.DataFrame:
    if not paths or not display_stats:
        return pd.DataFrame()
    total_plus = {}
    total_minus = {}
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        mask = df[name_col].astype(str).str.strip().str.endswith("Stat")
        players_df = df[mask].copy()
        if players_df.empty:
            continue
        header_info = []
        for col in df.columns[1:]:
            base = col.split(":")[-1].strip()
            m = re.search(r'([+-])\s*$', base)
            if m:
                sign = m.group(1)
                base = re.sub(r'\s*[+-]$', '', base).strip()
            else:
                if col.strip().endswith("+"):
                    sign = "+"
                    base = re.sub(r'\s*\+\s*$', '', base).strip()
                elif col.strip().endswith("-"):
                    sign = "-"
                    base = re.sub(r'\s*\-\s*$', '', base).strip()
                else:
                    continue
            header_info.append((col, base, sign))
        for _, row in players_df.iterrows():
            for col, base, sign in header_info:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+":
                    total_plus[base] = total_plus.get(base, 0) + val
                elif sign == "-":
                    total_minus[base] = total_minus.get(base, 0) + val
    row = {"Player": "Opportunities"}
    for s in display_stats:
        if s == "Pull Behind":
            plus = total_plus.get("Pull Behind", 0)
            minus_pb = total_minus.get("Pull Behind", 0)
            minus_pa = total_minus.get("Pull Ahead", 0)
            total = int(plus + minus_pb + minus_pa)
        else:
            plus = total_plus.get(s, 0)
            minus = total_minus.get(s, 0)
            total = int(plus + minus)
        row[f"{s} Opportunities"] = total
    counts_df = pd.DataFrame([row])
    cols = ["Player"] + [f"{s} Opportunities" for s in display_stats if f"{s} Opportunities" in counts_df.columns]
    counts_df = counts_df.reindex(columns=cols)
    return counts_df
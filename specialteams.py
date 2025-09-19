# special teams — compute Box Out and Forced Turnover success rates for a fixed master list
from pathlib import Path
from typing import List
import re

import pandas as pd
import numpy as np

DEF_FOLDER = Path(__file__).parent / "Defense"

# canonical names used for special teams (ONLY these rows will be considered)
CANONICAL_PLAYERS = [
    "#21 LUKA TARLAC",
    "#2 BRADEN FREEMAN",
    "#3 JAKE DAVIS",
    "#4 KIERAN ELLIOTT",
    "#8 JACKSON MOSLEY",
    "#5 GUZMAN VASILIC",
    "#14 CAYDEN WARD",
    "#10 HAMAD MOUSA",
    "#11 AUSTIN GOODE",
    "#15 AARON PRICE JR.",
    "#13 TROY PLUMTREE"
]

# target stat base names to search for in defense CSV headers
TARGET_STATS = ["Box Out", "Forced Turnover"]
SHOOTING_FOUL = "Shooting Foul"


def list_defense_files() -> list[Path]:
    if not DEF_FOLDER.exists():
        return []
    return sorted(DEF_FOLDER.glob("*.csv"))


def _find_header_info(cols: List[str], targets: List[str]):
    """
    Return list of (colname, base, sign) for columns matching target stat names.
    sign is '+' or '-' or None.
    """
    header_info = []
    for col in cols:
        base = col.split(":")[-1].strip()
        # normalize trailing +/- in header
        m = re.search(r"([+-])\s*$", base)
        if m:
            sign = m.group(1)
            base_name = re.sub(r"\s*[+-]$", "", base).strip()
        else:
            # also check raw col for trailing +/-
            if col.strip().endswith("+"):
                sign = "+"
                base_name = re.sub(r"\s*\+\s*$", "", base).strip()
            elif col.strip().endswith("-"):
                sign = "-"
                base_name = re.sub(r"\s*\-\s*$", "", base).strip()
            else:
                sign = None
                base_name = base
        for t in targets:
            if base_name.lower().startswith(t.lower()):
                header_info.append((col, t, sign))
                break
    return header_info


# helper to require the jersey number be present in the source row.
# build canonical metadata: numeric jersey and normalized name (without number)
_CANON_META = []
for p in CANONICAL_PLAYERS:
    m = re.match(r"^\s*#?(\d+)\s+(.*)$", p)
    if m:
        num = int(m.group(1))
        name_norm = m.group(2).strip().lower()
    else:
        num = None
        name_norm = re.sub(r"^\s*#?\d+\s*", "", p).strip().lower()
    _CANON_META.append({"num": num, "name": name_norm, "canon": p})

def _internal_key_from_canonical(canon: str) -> str:
    return re.sub(r"^\s*#?\d+\s*", "", canon).strip().lower()

def _map_special_name(raw: str) -> str | None:
    """
    Map only when the raw row contains a leading jersey number that matches the
    canonical player's jersey AND the remaining name tokens match the canonical name.
    This prevents mapping plain "Braden" rows to "#2 BRADEN FREEMAN".
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.match(r"^\s*#?(\d+)\b\s*(.*)$", s)
    if not m:
        return None  # require leading jersey number in source row
    row_num = int(m.group(1))
    row_name = m.group(2).strip().lower()
    for meta in _CANON_META:
        if meta["num"] is None:
            continue
        if row_num != meta["num"]:
            continue
        # require that the canonical name tokens are a superset / match of the row tokens
        if meta["name"] == row_name or meta["name"].startswith(row_name) or row_name.startswith(meta["name"]) or set(meta["name"].split()) & set(row_name.split()):
            return meta["canon"]
    return None


def compute_def_special_rates_for_paths(paths: list[Path]) -> pd.DataFrame:
    """
    Returns DataFrame with Player and success-rate columns for TARGET_STATS (% 0-100 or NaN).
    Rounds to 1 decimal. ONLY includes rows for CANONICAL_PLAYERS (in that order).
    """
    if not paths:
        return pd.DataFrame()
    plus_totals: dict = {}   # keys are internal normalized keys (no number, lower)
    minus_totals: dict = {}
    sf_totals: dict = {}     # shooting fouls per player
    poss_totals: dict = {}   # possessions per player (Team:team1 + Team:team2 or defense_lab)
    observed_stats = set()

    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        header_info = _find_header_info(list(df.columns[1:]), TARGET_STATS)
        if not header_info:
            continue

        # detect team possession columns / defense_lab in file
        team_cols = [c for c in df.columns if c in ("Team:team1", "Team:team2", "defense_lab")]
        for _, row in df.iterrows():
            raw = row.get(name_col, "")
            player = _map_special_name(raw)
            # only consider rows that map to canonical players
            if not player or player not in CANONICAL_PLAYERS:
                continue
            ik = _internal_key_from_canonical(player)
            plus_totals.setdefault(ik, {})
            minus_totals.setdefault(ik, {})
            sf_totals.setdefault(ik, 0)
            poss_totals.setdefault(ik, 0)
            for col, base, sign in header_info:
                observed_stats.add(base)
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+" or (isinstance(col, str) and "+" in col):
                    plus_totals[ik][base] = plus_totals[ik].get(base, 0) + val
                elif sign == "-" or (isinstance(col, str) and "-" in col):
                    minus_totals[ik][base] = minus_totals[ik].get(base, 0) + val
                else:
                    continue
            # shooting foul (may be present in a variety of column names)
            for c in df.columns[1:]:
                if "shooting foul" in c.lower():
                    v = pd.to_numeric(row.get(c, 0), errors="coerce")
                    v = 0 if pd.isna(v) else v
                    sf_totals[ik] = sf_totals.get(ik, 0) + v
            # possessions: sum team cols (or defense_lab) on this row
            row_poss = 0
            for tc in team_cols:
                row_poss += int(pd.to_numeric(row.get(tc, 0), errors="coerce") or 0)
            poss_totals[ik] = poss_totals.get(ik, 0) + row_poss

    if not observed_stats:
        # still return canonical players with NaN rates to keep table shape
        rows = [{"Player": p, **{s: np.nan for s in TARGET_STATS}} for p in CANONICAL_PLAYERS]
        return pd.DataFrame(rows)

    # build rows using canonical display names but lookup via internal keys
    rows = []
    for canon in CANONICAL_PLAYERS:
        ik = _internal_key_from_canonical(canon)
        row = {"Player": canon}
        for s in TARGET_STATS:
            plus = plus_totals.get(ik, {}).get(s, 0)
            minus = minus_totals.get(ik, {}).get(s, 0)
            denom = plus + minus
            if denom > 0:
                rate = (plus / denom) * 100.0
                row[s] = round(rate, 1)
            else:
                row[s] = np.nan
        rows.append(row)

    # attach shooting fouls and shooting-fouls-per-100-pos to rows
    out = pd.DataFrame(rows)
    sf_list = []
    sf_per100_list = []
    for canon in out["Player"].tolist():
        ik = _internal_key_from_canonical(canon)
        sf = int(sf_totals.get(ik, 0))
        poss = int(poss_totals.get(ik, 0))
        sf_list.append(sf)
        if poss > 0:
            sf_per100_list.append(round(sf / poss * 100.0, 1))
        else:
            sf_per100_list.append(np.nan)
    out["Shooting Fouls"] = sf_list
    out["Shooting Fouls / 100 Poss"] = sf_per100_list
    cols = ["Player"] + TARGET_STATS
    # keep success rate cols first, additional columns follow
    out = out.reindex(columns=cols + ["Shooting Fouls", "Shooting Fouls / 100 Poss"])
    return out


def compute_def_special_player_counts(paths: list[Path], stats: List[str]) -> pd.DataFrame:
    if not paths or not stats:
        # still return canonical players with zeros
        rows = [{"Player": p, **{f"{s} Opportunities": 0 for s in stats}, "Shooting Fouls": 0, "Possessions": 0} for p in CANONICAL_PLAYERS]
        return pd.DataFrame(rows)

    plus_totals: dict = {}
    minus_totals: dict = {}
    sf_totals: dict = {}
    poss_totals: dict = {}

    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        header_info = _find_header_info(list(df.columns[1:]), stats)
        if not header_info:
            continue

        # detect shooting-foul columns and team possession columns (case-insensitive)
        sf_cols = [c for c in df.columns if "shooting foul" in c.lower()]
        team_cols = [c for c in df.columns if c.lower() in ("team:team1", "team:team2", "defense_lab")]

        for _, row in df.iterrows():
            raw = row.get(name_col, "")
            player = _map_special_name(raw)
            if not player or player not in CANONICAL_PLAYERS:
                continue
            ik = _internal_key_from_canonical(player)
            plus_totals.setdefault(ik, {})
            minus_totals.setdefault(ik, {})
            sf_totals.setdefault(ik, 0)
            poss_totals.setdefault(ik, 0)

            for col, base, sign in header_info:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+" or (isinstance(col, str) and "+" in col):
                    plus_totals[ik][base] = plus_totals[ik].get(base, 0) + val
                elif sign == "-" or (isinstance(col, str) and "-" in col):
                    minus_totals[ik][base] = minus_totals[ik].get(base, 0) + val

            # shooting fouls on this row
            row_sf = 0
            for c in sf_cols:
                v = pd.to_numeric(row.get(c, 0), errors="coerce")
                row_sf += 0 if pd.isna(v) else int(v)
            sf_totals[ik] = sf_totals.get(ik, 0) + row_sf

            # possessions on this row (sum team cols)
            row_poss = 0
            for tc in team_cols:
                v = pd.to_numeric(row.get(tc, 0), errors="coerce")
                row_poss += 0 if pd.isna(v) else int(v)
            poss_totals[ik] = poss_totals.get(ik, 0) + row_poss

    rows = []
    for canon in CANONICAL_PLAYERS:
        ik = _internal_key_from_canonical(canon)
        row = {"Player": canon}
        for s in stats:
            plus = plus_totals.get(ik, {}).get(s, 0)
            minus = minus_totals.get(ik, {}).get(s, 0)
            total = int(plus + minus)
            row[f"{s} Opportunities"] = total
        row["Shooting Fouls"] = int(sf_totals.get(ik, 0))
        row["Possessions"] = int(poss_totals.get(ik, 0))
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["Player"] + [f"{s} Opportunities" for s in stats] + ["Shooting Fouls", "Possessions"]
    out = out.reindex(columns=cols, fill_value=0)
    return out


def compute_def_special_team_summary(paths: list[Path], stats: List[str]) -> pd.DataFrame:
    """
    Aggregate team-level totals (plus/minus) for the given stats across the selected files,
    compute success % rounded to 1 decimal. Only aggregates rows that map to CANONICAL_PLAYERS.
    """
    if not paths or not stats:
        return pd.DataFrame()
    total_plus: dict = {}
    total_minus: dict = {}
    total_sf = 0
    total_poss = 0

    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        header_info = _find_header_info(list(df.columns[1:]), stats)
        if not header_info:
            continue
        name_col = df.columns[0]
        for _, row in df.iterrows():
            raw = row.get(name_col, "")
            player = _map_special_name(raw)
            if not player or player not in CANONICAL_PLAYERS:
                continue
            ik = _internal_key_from_canonical(player)
            for col, base, sign in header_info:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+" or (isinstance(col, str) and "+" in col):
                    total_plus[base] = total_plus.get(base, 0) + val
                elif sign == "-" or (isinstance(col, str) and "-" in col):
                    total_minus[base] = total_minus.get(base, 0) + val
            # shooting fouls
            for c in df.columns[1:]:
                if "shooting foul" in c.lower():
                    v = pd.to_numeric(row.get(c, 0), errors="coerce")
                    v = 0 if pd.isna(v) else v
                    total_sf += int(v)
            # possessions
            for tc in ("Team:team1", "Team:team2", "defense_lab"):
                total_poss += int(pd.to_numeric(row.get(tc, 0), errors="coerce") or 0)

    row = {"Player": "Team"}
    for s in stats:
        plus = total_plus.get(s, 0)
        minus = total_minus.get(s, 0)
        denom = plus + minus
        if denom > 0:
            rate = (plus / denom) * 100.0
            row[s] = round(rate, 1)
        else:
            row[s] = np.nan
    row["Shooting Fouls"] = int(total_sf)
    row["Shooting Fouls / 100 Poss"] = round(total_sf / total_poss * 100.0, 1) if total_poss > 0 else np.nan
    team_df = pd.DataFrame([row])
    cols = ["Player"] + [c for c in stats if c in team_df.columns] + ["Shooting Fouls", "Shooting Fouls / 100 Poss"]
    team_df = team_df.reindex(columns=cols)
    return team_df
# special teams — compute Box Out and Forced Turnover success rates for a fixed master list
from pathlib import Path
from typing import List
import re

import pandas as pd
import numpy as np
from defense import process_defense_file
from constants import MASTER_PLAYERS

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


# map canonical entry (e.g. "#21 LUKA TARLAC") -> master short name key (e.g. "luka")
def _canonical_to_master_key(canonical: str):
    if not canonical:
        return None
    s = canonical.lower()
    # try direct substring match against MASTER_PLAYERS
    for m in MASTER_PLAYERS:
        if m.lower() in s:
            return m.lower()
    # fallback: look for any word in canonical that equals a master name
    words = re.findall(r"[a-z]+", s)
    for w in words:
        for m in MASTER_PLAYERS:
            if w == m.lower():
                return m.lower()
    return None


def compute_def_special_rates_for_paths(paths: list[Path]) -> pd.DataFrame:
    """
    Returns DataFrame with Player and success-rate columns for TARGET_STATS (% 0-100 or NaN).
    Rounds to 1 decimal. Only includes rows mapped to CANONICAL_PLAYERS.
    Also returns Shooting Fouls (raw count) and Possessions and Shooting Fouls / Poss (rounded 1 decimal).
    """
    if not paths:
        return pd.DataFrame()
    plus_totals: dict = {}
    minus_totals: dict = {}
    sf_totals: dict = {}
    poss_totals: dict = {}
    observed_stats = set()

    for p in paths:
        # use process_defense_file to get canonical DefPoss per player (applies defense_lab or Team:team1+team2 logic)
        try:
            proc = process_defense_file(p, p.name)
        except Exception:
            proc = pd.DataFrame()

        # possession map keyed by master short-name (lowercased) -> DefPoss
        poss_map_master: dict = {}
        if not proc.empty:
            for _, prow in proc.iterrows():
                pname = str(prow["Player"])
                defposs = int(prow.get("DefPoss", 0) or 0)
                master_key = _canonical_to_master_key(pname)
                if master_key:
                    poss_map_master[master_key] = poss_map_master.get(master_key, 0) + defposs

        # read raw CSV to collect plus/minus and shooting foul counts (possessions taken from poss_map)
        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        header_info = _find_header_info(list(df.columns[1:]), TARGET_STATS)
        if not header_info:
            continue

        sf_cols = [c for c in df.columns if "shooting foul" in c.lower()]

        # per-file buffers
        file_plus = {}
        file_minus = {}
        file_sf = {}

        for _, row in df.iterrows():
            raw = row.get(name_col, "")
            player = _map_special_name(raw)
            if not player or player not in CANONICAL_PLAYERS:
                continue
            ik = _internal_key_from_canonical(player)
            file_plus.setdefault(ik, {})
            file_minus.setdefault(ik, {})
            file_sf.setdefault(ik, 0)

            for col, base, sign in header_info:
                observed_stats.add(base)
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+" or (isinstance(col, str) and "+" in col):
                    file_plus[ik][base] = file_plus[ik].get(base, 0) + val
                elif sign == "-" or (isinstance(col, str) and "-" in col):
                    file_minus[ik][base] = file_minus[ik].get(base, 0) + val

            # shooting fouls on this row
            row_sf = 0
            for c in sf_cols:
                v = pd.to_numeric(row.get(c, 0), errors="coerce")
                row_sf += 0 if pd.isna(v) else int(v)
            file_sf[ik] = file_sf.get(ik, 0) + row_sf

        # merge file buffers into global totals; possessions come from poss_map_master
        for ik in set(list(file_plus.keys()) + list(file_minus.keys()) + list(file_sf.keys()) + list(poss_map_master.keys())):
            plus_totals.setdefault(ik, {})
            minus_totals.setdefault(ik, {})
            sf_totals.setdefault(ik, 0)
            poss_totals.setdefault(ik, 0)
            for base, v in file_plus.get(ik, {}).items():
                plus_totals[ik][base] = plus_totals[ik].get(base, 0) + v
            for base, v in file_minus.get(ik, {}).items():
                minus_totals[ik][base] = minus_totals[ik].get(base, 0) + v
            sf_totals[ik] = sf_totals.get(ik, 0) + file_sf.get(ik, 0)
            # map canonical/internal key to master short-name to fetch possessions
            master_key_for_ik = _canonical_to_master_key(_canonical_from_internal_key(ik) if "_canonical_from_internal_key" in globals() else ik)
            poss_totals[ik] = poss_totals.get(ik, 0) + poss_map_master.get(master_key_for_ik, 0)

    if not observed_stats:
        # still return canonical players with NaN rates and zero SF/poss to keep table shape
        rows = []
        for p in CANONICAL_PLAYERS:
            rows.append({
                "Player": p,
                **{s: np.nan for s in TARGET_STATS},
                "Shooting Fouls": 0,
                "Possessions": 0,
                "Shooting Fouls / Poss": np.nan,
            })
        return pd.DataFrame(rows)

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
        # attach shooting fouls and possessions and per-pos (rounded 1 decimal)
        sf = int(sf_totals.get(ik, 0))
        poss = int(poss_totals.get(ik, 0))
        row["Shooting Fouls"] = sf
        row["Possessions"] = poss
        # express Shooting Fouls per 100 possessions
        row["Shooting Fouls / Poss"] = round((sf / poss) * 100, 1) if poss > 0 else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["Player"] + TARGET_STATS + ["Shooting Fouls", "Possessions", "Shooting Fouls / Poss"]
    out = out.reindex(columns=cols)
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
        # processed file to get DefPoss per player
        try:
            proc = process_defense_file(p, p.name)
        except Exception:
            proc = pd.DataFrame()

        poss_map_master = {}
        if not proc.empty:
            for _, prow in proc.iterrows():
                pname = str(prow["Player"])
                master_key = _canonical_to_master_key(pname)
                if master_key:
                    poss_map_master[master_key] = poss_map_master.get(master_key, 0) + int(prow.get("DefPoss", 0) or 0)

        df = pd.read_csv(p)
        if df.empty:
            continue
        name_col = df.columns[0]
        header_info = _find_header_info(list(df.columns[1:]), stats)
        if not header_info:
            continue

        sf_cols = [c for c in df.columns if "shooting foul" in c.lower()]

        file_plus = {}
        file_minus = {}
        file_sf = {}

        for _, row in df.iterrows():
            raw = row.get(name_col, "")
            player = _map_special_name(raw)
            if not player or player not in CANONICAL_PLAYERS:
                continue
            ik = _internal_key_from_canonical(player)
            file_plus.setdefault(ik, {})
            file_minus.setdefault(ik, {})
            file_sf.setdefault(ik, 0)

            for col, base, sign in header_info:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                val = 0 if pd.isna(val) else val
                if sign == "+" or (isinstance(col, str) and "+" in col):
                    file_plus[ik][base] = file_plus[ik].get(base, 0) + val
                elif sign == "-" or (isinstance(col, str) and "-" in col):
                    file_minus[ik][base] = file_minus[ik].get(base, 0) + val

            # shooting fouls on this row
            row_sf = 0
            for c in sf_cols:
                v = pd.to_numeric(row.get(c, 0), errors="coerce")
                row_sf += 0 if pd.isna(v) else int(v)
            file_sf[ik] = file_sf.get(ik, 0) + row_sf

        # merge per-file buffers into global totals; possessions from poss_map_master
        for ik in set(list(file_plus.keys()) + list(file_minus.keys()) + list(file_sf.keys()) + list(poss_map_master.keys())):
            plus_totals.setdefault(ik, {})
            minus_totals.setdefault(ik, {})
            sf_totals.setdefault(ik, 0)
            poss_totals.setdefault(ik, 0)
            for base, v in file_plus.get(ik, {}).items():
                plus_totals[ik][base] = plus_totals[ik].get(base, 0) + v
            for base, v in file_minus.get(ik, {}).items():
                minus_totals[ik][base] = minus_totals[ik].get(base, 0) + v
            sf_totals[ik] = sf_totals.get(ik, 0) + file_sf.get(ik, 0)
            master_key_for_ik = _canonical_to_master_key(_canonical_from_internal_key(ik) if "_canonical_from_internal_key" in globals() else ik)
            poss_totals[ik] = poss_totals.get(ik, 0) + poss_map_master.get(master_key_for_ik, 0)

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
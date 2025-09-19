from pathlib import Path
import pandas as pd
import numpy as np

from utils import clean_and_filter_players, coalesce_cols, parse_date_from_filename

def process_defense_file(file, filename: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = clean_and_filter_players(df)
    shot_cols = [
        "shot chart:RZ + (D)",
        "shot chart:TT + (D)",
        "shot chart:IO3 + (D)",
        "shot chart:NIO3 + (D)",
        "shot chart:FT 2 (D)",
        "shot chart:FT 3 (D)",
    ]
    team_cols = ["Team:team1", "Team:team2"]
    coalesce_cols(df, ["defense_lab"] + shot_cols + team_cols)

    for c in shot_cols + ["defense_lab"] + team_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["DefPoints"] = (
        2.0 * df["shot chart:RZ + (D)"] +
        2.0 * df["shot chart:TT + (D)"] +
        3.0 * df["shot chart:IO3 + (D)"] +
        3.0 * df["shot chart:NIO3 + (D)"] +
        1.5 * df["shot chart:FT 2 (D)"] +
        2.25 * df["shot chart:FT 3 (D)"]
    )

    if "defense_lab" in df.columns and df["defense_lab"].sum() > 0:
        df = df.rename(columns={"defense_lab": "DefPoss"})
    else:
        df["DefPoss"] = df.get("Team:team1", 0.0).fillna(0.0) + df.get("Team:team2", 0.0).fillna(0.0)
        df["DefPoss"] = df["DefPoss"].round().astype(int)

    df["Date"] = parse_date_from_filename(filename)
    grp_cols = ["Date", "Player"]
    numeric_cols = ["DefPoints", "DefPoss"]
    df = df.groupby(grp_cols, as_index=False)[numeric_cols].sum()
    return df[["Date", "Player", "DefPoints", "DefPoss"]]
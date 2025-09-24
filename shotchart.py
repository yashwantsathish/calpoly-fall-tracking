import re
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from utils import clean_and_filter_players, coalesce_cols, parse_date_from_filename

def process_shot_chart_file(file, filename: str) -> pd.DataFrame:
    """Process a single shot chart CSV file."""
    df = pd.read_csv(file)
    
    # Filter for "Player Stat" rows and Team rows
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame()
    
    # Get the player name column (usually first column)
    name_col = df.columns[0]
    # Include both " Stat" rows and Team 1/Team 2 rows
    mask = (df[name_col].astype(str).str.endswith(' Stat')) | (df[name_col].astype(str).isin(['Team 1', 'Team 2']))
    df = df[mask].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Clean the player names by removing " Stat" suffix (but keep Team 1/Team 2 as is)
    df[name_col] = df[name_col].astype(str).apply(lambda x: x.replace(' Stat', '') if x.endswith(' Stat') else x)
    df = clean_and_filter_players(df)
    
    if df.empty:
        return pd.DataFrame()
    
    # Expected shot chart columns (all possible columns, even if not in current CSV)
    shot_cols = [
        "shot chart:FT 2",
        "shot chart:FT 3", 
        "shot chart:IO3 +",
        "shot chart:IO3 -",
        "shot chart:NIO3 +",
        "shot chart:NIO3 -",
        "shot chart:RZ +",
        "shot chart:RZ -",
        "shot chart:TT +",
        "shot chart:TT -",
        "shot chart:Turnover -",
    ]
    
    # Player stats and team columns
    other_cols = ["player stats:PT +", "Team:team1", "Team:team2", "offense_lab"]
    
    coalesce_cols(df, shot_cols + other_cols)
    
    # Convert to numeric
    for c in shot_cols + other_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    
    # Calculate total shots made and missed (using all possible columns)
    made_shots = ["shot chart:IO3 +", "shot chart:NIO3 +", "shot chart:RZ +", "shot chart:TT +"]
    missed_shots = ["shot chart:IO3 -", "shot chart:NIO3 -", "shot chart:RZ -", "shot chart:TT -"]
    
    df["ShotsMade"] = sum(df.get(col, 0) for col in made_shots)
    df["ShotsMissed"] = sum(df.get(col, 0) for col in missed_shots)
    df["TotalShots"] = df["ShotsMade"] + df["ShotsMissed"]
    df["ShootingPct"] = np.where(df["TotalShots"] > 0, (df["ShotsMade"] / df["TotalShots"] * 100).round(1), np.nan)
    
    # Calculate 2-pointers and 3-pointers separately
    df["TwoPointMade"] = df.get("shot chart:RZ +", 0) + df.get("shot chart:TT +", 0)  # RZ + TT (2-pointers)
    df["TwoPointMissed"] = df.get("shot chart:RZ -", 0) + df.get("shot chart:TT -", 0)
    df["ThreePointMade"] = df.get("shot chart:IO3 +", 0) + df.get("shot chart:NIO3 +", 0)  # All 3-pointers
    df["ThreePointMissed"] = df.get("shot chart:IO3 -", 0) + df.get("shot chart:NIO3 -", 0)  # All 3-point misses
    df["TwoPointAttempts"] = df["TwoPointMade"] + df["TwoPointMissed"]
    df["ThreePointAttempts"] = df["ThreePointMade"] + df["ThreePointMissed"]
    df["FieldGoalAttempts"] = df["TwoPointAttempts"] + df["ThreePointAttempts"]
    df["FieldGoalMade"] = df["TwoPointMade"] + df["ThreePointMade"]
    
    # Calculate points from actual shot makes and free throws
    df["FieldGoalPoints"] = (
        2.0 * df.get("shot chart:RZ +", 0) +      # 2-pointers from restricted zone
        2.0 * df.get("shot chart:TT +", 0) +      # 2-pointers from tough two
        3.0 * df.get("shot chart:IO3 +", 0) +     # 3-pointers inside-out
        3.0 * df.get("shot chart:NIO3 +", 0)      # 3-pointers non-inside-out
    )
    
    # Estimate free throw attempts and makes
    # If FT data is available, use it; otherwise estimate from points
    df["FTMade2"] = df.get("shot chart:FT 2", 0)
    df["FTMade3"] = df.get("shot chart:FT 3", 0)
    df["EstimatedFTM"] = df["FTMade2"] + df["FTMade3"]
    
    # Total points = Field Goal Points + Free Throw Points
    df["Points"] = df["FieldGoalPoints"] + df["EstimatedFTM"]
    
    # If no FT data, estimate from remaining points after field goals
    if df["EstimatedFTM"].sum() == 0:
        df["EstimatedFTM"] = np.maximum(0, df["Points"] - (2 * df["TwoPointMade"]) - (3 * df["ThreePointMade"]))
    
    df["EstimatedFTA"] = np.where(df["EstimatedFTM"] > 0, df["EstimatedFTM"] / 0.75, 0)  # Assume 75% FT shooting
    
    # Effective Field Goal % = (FGM + 0.5 * 3PM) / FGA
    df["EffectiveFGPct"] = np.where(
        df["FieldGoalAttempts"] > 0,
        ((df["FieldGoalMade"] + 0.5 * df["ThreePointMade"]) / df["FieldGoalAttempts"] * 100).round(0),
        np.nan
    )
    
    # True Shooting % = Points / (2 * (FGA + 0.44 * FTA))
    df["TrueShootingPct"] = np.where(
        (df["FieldGoalAttempts"] + 0.44 * df["EstimatedFTA"]) > 0,
        (df["Points"] / (2 * (df["FieldGoalAttempts"] + 0.44 * df["EstimatedFTA"])) * 100).round(0),
        np.nan
    )
    
    # Points Per Possession (PPP) 
    # Use offense_lab for possessions (actual possessions from the data)
    df["Possessions"] = df.get("offense_lab", 0)
    
    df["PointsPerPossession"] = np.where(
        df["Possessions"] > 0,
        (df["Points"] / df["Possessions"]).round(2),
        np.nan
    )
    
    # New Efficiency Metrics
    # Red Zone %
    rz_attempts = df.get("shot chart:RZ +", 0) + df.get("shot chart:RZ -", 0)
    df["RedZonePct"] = np.where(
        rz_attempts > 0,
        (df.get("shot chart:RZ +", 0) / rz_attempts * 100).round(0),
        np.nan
    )
    
    # Tough Two %
    tt_attempts = df.get("shot chart:TT +", 0) + df.get("shot chart:TT -", 0)
    df["ToughTwoPct"] = np.where(
        tt_attempts > 0,
        (df.get("shot chart:TT +", 0) / tt_attempts * 100).round(0),
        np.nan
    )
    
    # Inside-Out 3 %
    io3_attempts = df.get("shot chart:IO3 +", 0) + df.get("shot chart:IO3 -", 0)
    df["InsideOut3Pct"] = np.where(
        io3_attempts > 0,
        (df.get("shot chart:IO3 +", 0) / io3_attempts * 100).round(0),
        np.nan
    )
    
    # Non Inside-Out 3 %
    nio3_attempts = df.get("shot chart:NIO3 +", 0) + df.get("shot chart:NIO3 -", 0)
    df["NonInsideOut3Pct"] = np.where(
        nio3_attempts > 0,
        (df.get("shot chart:NIO3 +", 0) / nio3_attempts * 100).round(0),
        np.nan
    )
    
    return df
    
    # 2PT% (Red Zone + Tough Two)
    df["TwoPointPct"] = np.where(
        df["TwoPointAttempts"] > 0,
        (df["TwoPointMade"] / df["TwoPointAttempts"] * 100).round(0),
        np.nan
    )
    
    # 3PT% (Inside-Out 3 + Non Inside-Out 3)
    df["ThreePointPct"] = np.where(
        df["ThreePointAttempts"] > 0,
        (df["ThreePointMade"] / df["ThreePointAttempts"] * 100).round(0),
        np.nan
    )
    
    # Total points calculation (already calculated above as Points)
    df["TotalPoints"] = df["Points"]
    
    # Add date
    df["Date"] = parse_date_from_filename(filename)
    
    return df

def list_shot_chart_files() -> list[Path]:
    """Get list of shot chart CSV files."""
    base = Path(__file__).parent
    folder = base / "Shot Chart"
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"))

def compute_shot_chart_stats_for_paths(paths: list[Path]) -> pd.DataFrame:
    """Compute shot chart statistics for selected files."""
    if not paths:
        return pd.DataFrame()
    
    all_data = []
    for path in paths:
        try:
            df = process_shot_chart_file(path, path.name)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Group by player and sum statistics
    numeric_cols = [
        "Team:team1", "Team:team2", "player stats:PT +", 
        "shot chart:FT 2", "shot chart:FT 3",
        "shot chart:IO3 +", "shot chart:IO3 -", 
        "shot chart:NIO3 +", "shot chart:NIO3 -",
        "shot chart:RZ +", "shot chart:RZ -",
        "shot chart:TT +", "shot chart:TT -",
        "shot chart:Turnover -", "offense_lab",
        "ShotsMade", "ShotsMissed", "TotalShots",
        "TwoPointMade", "TwoPointMissed", "TwoPointAttempts",
        "ThreePointMade", "ThreePointMissed", "ThreePointAttempts",
        "FieldGoalAttempts", "FieldGoalMade", "Points",
        "FTMade2", "FTMade3", "EstimatedFTM", "EstimatedFTA",
        "FieldGoalPoints", "TotalPoints", "Possessions"
    ]
    
    # Only include columns that exist in the dataframe
    available_cols = [col for col in numeric_cols if col in combined_df.columns]
    player_stats = combined_df.groupby("Player", as_index=False)[available_cols].sum()
    
    # Recalculate percentages after aggregation
    player_stats["ShootingPct"] = np.where(
        player_stats["TotalShots"] > 0, 
        (player_stats["ShotsMade"] / player_stats["TotalShots"] * 100).round(1), 
        np.nan
    )
    
    # Recalculate Effective FG%
    player_stats["EffectiveFGPct"] = np.where(
        player_stats["FieldGoalAttempts"] > 0,
        ((player_stats["FieldGoalMade"] + 0.5 * player_stats["ThreePointMade"]) / player_stats["FieldGoalAttempts"] * 100).round(1),
        np.nan
    )
    
    # Recalculate True Shooting %
    true_shooting_denominator = 2 * (player_stats["FieldGoalAttempts"] + 0.44 * player_stats["EstimatedFTA"])
    player_stats["TrueShootingPct"] = np.where(
        true_shooting_denominator > 0,
        (player_stats["Points"] / true_shooting_denominator * 100).round(1),
        np.nan
    )
    
    # Recalculate Points Per Possession
    player_stats["PointsPerPossession"] = np.where(
        player_stats["Possessions"] > 0,
        (player_stats["Points"] / player_stats["Possessions"]).round(2),
        np.nan
    )
    
    # Recalculate new efficiency metrics
    # Red Zone %
    rz_attempts = player_stats.get("shot chart:RZ +", 0) + player_stats.get("shot chart:RZ -", 0)
    player_stats["RedZonePct"] = np.where(
        rz_attempts > 0,
        (player_stats.get("shot chart:RZ +", 0) / rz_attempts * 100).round(0),
        np.nan
    )
    
    # Tough Two %
    tt_attempts = player_stats.get("shot chart:TT +", 0) + player_stats.get("shot chart:TT -", 0)
    player_stats["ToughTwoPct"] = np.where(
        tt_attempts > 0,
        (player_stats.get("shot chart:TT +", 0) / tt_attempts * 100).round(0),
        np.nan
    )
    
    # Inside-Out 3 %
    io3_attempts = player_stats.get("shot chart:IO3 +", 0) + player_stats.get("shot chart:IO3 -", 0)
    player_stats["InsideOut3Pct"] = np.where(
        io3_attempts > 0,
        (player_stats.get("shot chart:IO3 +", 0) / io3_attempts * 100).round(0),
        np.nan
    )
    
    # Non Inside-Out 3 %
    nio3_attempts = player_stats.get("shot chart:NIO3 +", 0) + player_stats.get("shot chart:NIO3 -", 0)
    player_stats["NonInsideOut3Pct"] = np.where(
        nio3_attempts > 0,
        (player_stats.get("shot chart:NIO3 +", 0) / nio3_attempts * 100).round(0),
        np.nan
    )
    
    # 2PT% and 3PT%
    player_stats["TwoPointPct"] = np.where(
        player_stats["TwoPointAttempts"] > 0,
        (player_stats["TwoPointMade"] / player_stats["TwoPointAttempts"] * 100).round(0),
        np.nan
    )
    
    player_stats["ThreePointPct"] = np.where(
        player_stats["ThreePointAttempts"] > 0,
        (player_stats["ThreePointMade"] / player_stats["ThreePointAttempts"] * 100).round(0),
        np.nan
    )
    
    return player_stats

def compute_team_shot_chart_summary(paths: list[Path]) -> pd.DataFrame:
    """Compute team-level shot chart summaries."""
    if not paths:
        return pd.DataFrame()
    
    all_data = []
    for path in paths:
        try:
            df = process_shot_chart_file(path, path.name)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Use the actual Team 1/Team 2 summary rows from the CSV data
    team_rows = combined_df[combined_df["Player"].isin(["Team 1", "Team 2"])].copy()
    
    if team_rows.empty:
        return pd.DataFrame()
    
    return team_rows


def compute_overall_team_summary(paths: list[Path]) -> pd.DataFrame:
    """Compute overall team summary combining Team 1 and Team 2."""
    team_stats = compute_team_shot_chart_summary(paths)
    
    if team_stats.empty:
        return pd.DataFrame()
    
    # Sum all team stats
    numeric_cols = [
        "Team:team1", "Team:team2", "player stats:PT +", 
        "shot chart:FT 2", "shot chart:FT 3",
        "shot chart:IO3 +", "shot chart:IO3 -", 
        "shot chart:NIO3 +", "shot chart:NIO3 -",
        "shot chart:RZ +", "shot chart:RZ -",
        "shot chart:TT +", "shot chart:TT -",
        "shot chart:Turnover -", "offense_lab",
        "ShotsMade", "ShotsMissed", "TotalShots",
        "TwoPointMade", "TwoPointMissed", "TwoPointAttempts",
        "ThreePointMade", "ThreePointMissed", "ThreePointAttempts",
        "FieldGoalAttempts", "FieldGoalMade", "Points",
        "FTMade2", "FTMade3", "EstimatedFTM", "EstimatedFTA",
        "FieldGoalPoints", "TotalPoints", "Possessions"
    ]
    
    # Only include columns that exist in the dataframe
    available_cols = [col for col in numeric_cols if col in team_stats.columns]
    overall_stats = team_stats[available_cols].sum().to_frame().T
    overall_stats["Player"] = "Overall Team"
    
    # Recalculate percentages
    overall_stats["ShootingPct"] = np.where(
        overall_stats["TotalShots"] > 0, 
        (overall_stats["ShotsMade"] / overall_stats["TotalShots"] * 100).round(1), 
        np.nan
    )
    
    # Recalculate Effective FG%
    overall_stats["EffectiveFGPct"] = np.where(
        overall_stats["FieldGoalAttempts"] > 0,
        ((overall_stats["FieldGoalMade"] + 0.5 * overall_stats["ThreePointMade"]) / overall_stats["FieldGoalAttempts"] * 100).round(1),
        np.nan
    )
    
    # Recalculate True Shooting %
    true_shooting_denominator = 2 * (overall_stats["FieldGoalAttempts"] + 0.44 * overall_stats["EstimatedFTA"])
    overall_stats["TrueShootingPct"] = np.where(
        true_shooting_denominator > 0,
        (overall_stats["Points"] / true_shooting_denominator * 100).round(1),
        np.nan
    )
    
    # Recalculate Points Per Possession using offense_lab
    overall_stats["PointsPerPossession"] = np.where(
        overall_stats["Possessions"] > 0,
        (overall_stats["Points"] / overall_stats["Possessions"]).round(2),
        np.nan
    )
    
    # Recalculate new efficiency metrics
    # Red Zone %
    rz_attempts = overall_stats.get("shot chart:RZ +", 0) + overall_stats.get("shot chart:RZ -", 0)
    overall_stats["RedZonePct"] = np.where(
        rz_attempts > 0,
        (overall_stats.get("shot chart:RZ +", 0) / rz_attempts * 100).round(1),
        np.nan
    )
    
    # Tough Two %
    tt_attempts = overall_stats.get("shot chart:TT +", 0) + overall_stats.get("shot chart:TT -", 0)
    overall_stats["ToughTwoPct"] = np.where(
        tt_attempts > 0,
        (overall_stats.get("shot chart:TT +", 0) / tt_attempts * 100).round(1),
        np.nan
    )
    
    # Inside-Out 3 %
    io3_attempts = overall_stats.get("shot chart:IO3 +", 0) + overall_stats.get("shot chart:IO3 -", 0)
    overall_stats["InsideOut3Pct"] = np.where(
        io3_attempts > 0,
        (overall_stats.get("shot chart:IO3 +", 0) / io3_attempts * 100).round(1),
        np.nan
    )
    
    # Non Inside-Out 3 %
    nio3_attempts = team_stats.get("shot chart:NIO3 +", 0) + team_stats.get("shot chart:NIO3 -", 0)
    team_stats["NonInsideOut3Pct"] = np.where(
        nio3_attempts > 0,
        (team_stats.get("shot chart:NIO3 +", 0) / nio3_attempts * 100).round(0),
        np.nan
    )
    
    # 2PT% and 3PT%
    team_stats["TwoPointPct"] = np.where(
        team_stats["TwoPointAttempts"] > 0,
        (team_stats["TwoPointMade"] / team_stats["TwoPointAttempts"] * 100).round(0),
        np.nan
    )
    
    team_stats["ThreePointPct"] = np.where(
        team_stats["ThreePointAttempts"] > 0,
        (team_stats["ThreePointMade"] / team_stats["ThreePointAttempts"] * 100).round(0),
        np.nan
    )
    
    return team_stats

def compute_overall_team_summary(paths: list[Path]) -> pd.DataFrame:
    """Compute overall team summary combining Team 1 and Team 2."""
    team_stats = compute_team_shot_chart_summary(paths)
    
    if team_stats.empty:
        return pd.DataFrame()
    
    # Sum all team stats
    numeric_cols = [
        "Team:team1", "Team:team2", "player stats:PT +", 
        "shot chart:FT 2", "shot chart:FT 3",
        "shot chart:IO3 +", "shot chart:IO3 -", 
        "shot chart:NIO3 +", "shot chart:NIO3 -",
        "shot chart:RZ +", "shot chart:RZ -",
        "shot chart:TT +", "shot chart:TT -",
        "shot chart:Turnover -", "offense_lab",
        "ShotsMade", "ShotsMissed", "TotalShots",
        "TwoPointMade", "TwoPointMissed", "TwoPointAttempts",
        "ThreePointMade", "ThreePointMissed", "ThreePointAttempts",
        "FieldGoalAttempts", "FieldGoalMade", "Points",
        "FTMade2", "FTMade3", "EstimatedFTM", "EstimatedFTA",
        "FieldGoalPoints", "TotalPoints", "Possessions"
    ]
    
    # Only include columns that exist in the dataframe
    available_cols = [col for col in numeric_cols if col in team_stats.columns]
    overall_stats = team_stats[available_cols].sum().to_frame().T
    overall_stats["Player"] = "Overall Team"
    
    # Recalculate percentages
    overall_stats["ShootingPct"] = np.where(
        overall_stats["TotalShots"] > 0, 
        (overall_stats["ShotsMade"] / overall_stats["TotalShots"] * 100).round(1), 
        np.nan
    )
    
    # Recalculate Effective FG%
    overall_stats["EffectiveFGPct"] = np.where(
        overall_stats["FieldGoalAttempts"] > 0,
        ((overall_stats["FieldGoalMade"] + 0.5 * overall_stats["ThreePointMade"]) / overall_stats["FieldGoalAttempts"] * 100).round(1),
        np.nan
    )
    
    # Recalculate True Shooting %
    true_shooting_denominator = 2 * (overall_stats["FieldGoalAttempts"] + 0.44 * overall_stats["EstimatedFTA"])
    overall_stats["TrueShootingPct"] = np.where(
        true_shooting_denominator > 0,
        (overall_stats["Points"] / true_shooting_denominator * 100).round(1),
        np.nan
    )
    
    # Recalculate Points Per Possession using offense_lab
    overall_stats["PointsPerPossession"] = np.where(
        overall_stats["Possessions"] > 0,
        (overall_stats["Points"] / overall_stats["Possessions"]).round(2),
        np.nan
    )
    
    # Recalculate new efficiency metrics
    # Red Zone %
    rz_attempts = overall_stats.get("shot chart:RZ +", 0) + overall_stats.get("shot chart:RZ -", 0)
    overall_stats["RedZonePct"] = np.where(
        rz_attempts > 0,
        (overall_stats.get("shot chart:RZ +", 0) / rz_attempts * 100).round(0),
        np.nan
    )
    
    # Tough Two %
    tt_attempts = overall_stats.get("shot chart:TT +", 0) + overall_stats.get("shot chart:TT -", 0)
    overall_stats["ToughTwoPct"] = np.where(
        tt_attempts > 0,
        (overall_stats.get("shot chart:TT +", 0) / tt_attempts * 100).round(0),
        np.nan
    )
    
    # Inside-Out 3 %
    io3_attempts = overall_stats.get("shot chart:IO3 +", 0) + overall_stats.get("shot chart:IO3 -", 0)
    overall_stats["InsideOut3Pct"] = np.where(
        io3_attempts > 0,
        (overall_stats.get("shot chart:IO3 +", 0) / io3_attempts * 100).round(0),
        np.nan
    )
    
    # Non Inside-Out 3 %
    nio3_attempts = overall_stats.get("shot chart:NIO3 +", 0) + overall_stats.get("shot chart:NIO3 -", 0)
    overall_stats["NonInsideOut3Pct"] = np.where(
        nio3_attempts > 0,
        (overall_stats.get("shot chart:NIO3 +", 0) / nio3_attempts * 100).round(0),
        np.nan
    )
    
    # 2PT% and 3PT%
    overall_stats["TwoPointPct"] = np.where(
        overall_stats["TwoPointAttempts"] > 0,
        (overall_stats["TwoPointMade"] / overall_stats["TwoPointAttempts"] * 100).round(0),
        np.nan
    )
    
    overall_stats["ThreePointPct"] = np.where(
        overall_stats["ThreePointAttempts"] > 0,
        (overall_stats["ThreePointMade"] / overall_stats["ThreePointAttempts"] * 100).round(0),
        np.nan
    )
    
    # Reorder columns to include Player first
    cols = ["Player"] + available_cols + ["ShootingPct", "EffectiveFGPct", "TrueShootingPct", "PointsPerPossession", 
            "RedZonePct", "ToughTwoPct", "InsideOut3Pct", "NonInsideOut3Pct", "TwoPointPct", "ThreePointPct"]
    cols = [col for col in cols if col in overall_stats.columns]  # Only include existing columns
    return overall_stats[cols]

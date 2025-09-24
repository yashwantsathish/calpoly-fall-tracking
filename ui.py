from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from constants import CATEGORIES
from utils import list_offense_files, aggregate_by_player
from offense import (
    compute_offense_success_rates_for_paths,
    compute_offense_player_counts,
    compute_offense_team_summary,
    compute_offense_team_counts,
    process_offense_file,
)
from defense import process_defense_file
from specialteams import (
    list_defense_files,
    compute_def_special_rates_for_paths,
    compute_def_special_player_counts,
    compute_def_special_team_summary,
)
from shotchart import (
    list_shot_chart_files,
    compute_shot_chart_stats_for_paths,
    compute_team_shot_chart_summary,
    compute_overall_team_summary,
)
# NOTE: render_ui expects agg to be precomputed and off/def dfs available if needed
def render_ui(off_df: pd.DataFrame, def_df: pd.DataFrame, agg: pd.DataFrame):
    tab_tracking, tab_special, tab_shot_chart, tab_counting = st.tabs(["Tracking Data", "Special Teams", "Shot Chart", "Counting Stats"])

    with tab_tracking:
        offense_paths = list_offense_files()
        offense_names = [p.name for p in offense_paths]

        st.subheader("Offense")
        st.caption(
            "Select one or more practice dates and the categories to view. "
        )

        if not offense_paths:
            st.info("No files found in Offense/ folder.")
        else:
            selected = st.multiselect("Dates", options=offense_names, default=offense_names, key="off_files")
            selected_paths = [p for p in offense_paths if p.name in selected]

            cat_keys = list(CATEGORIES.keys())
            btn_cols = st.columns([1, 1, 6])
            if "offense_cat_selected" not in st.session_state:
                st.session_state.offense_cat_selected = cat_keys.copy()
            if btn_cols[0].button("Select All", key="off_select_all"):
                st.session_state.offense_cat_selected = cat_keys.copy()
            if btn_cols[1].button("Clear", key="off_clear"):
                st.session_state.offense_cat_selected = []

            box_cols = st.columns(len(cat_keys))
            for col, cat in zip(box_cols, cat_keys):
                checked = cat in st.session_state.offense_cat_selected
                new_checked = col.checkbox(cat, value=checked, key=f"cat_chk_{cat}")
                if new_checked and cat not in st.session_state.offense_cat_selected:
                    st.session_state.offense_cat_selected.append(cat)
                if not new_checked and cat in st.session_state.offense_cat_selected:
                    st.session_state.offense_cat_selected.remove(cat)

            selected_cats = st.session_state.offense_cat_selected

            if not selected_paths:
                st.info("Select one or more files to view aggregated success rates.")
            else:
                rates_df = compute_offense_success_rates_for_paths(selected_paths)
                if rates_df.empty:
                    st.info("No player-stat rows found in selected files.")
                else:
                    to_drop = [c for c in rates_df.columns if isinstance(c, str) and c.strip().lower().replace(" ", "") in ("offenselab", "offense_lab", "offenselab")]
                    if to_drop:
                        rates_df = rates_df.drop(columns=to_drop)

                    show_stats = []
                    for cat in selected_cats:
                        show_stats.extend(CATEGORIES.get(cat, []))
                    seen = set()
                    show_stats_ordered = []
                    for s in show_stats:
                        if s not in seen:
                            seen.add(s)
                            show_stats_ordered.append(s)

                    available_stats = [s for s in show_stats_ordered if s in rates_df.columns]
                    if "off_visible_stats" not in st.session_state:
                        st.session_state.off_visible_stats = available_stats.copy()
                    default_visible = [s for s in st.session_state.off_visible_stats if s in available_stats]
                    if not default_visible:
                        default_visible = available_stats.copy()

                    st.write("Show / hide stats (uncheck to remove from view):")
                    visible_stats = st.multiselect(
                        "Visible stats",
                        options=available_stats,
                        default=default_visible,
                        key="off_visible_stats_multiselect"
                    )
                    st.session_state.off_visible_stats = [s for s in visible_stats if s in available_stats]
                    cols_to_show = ["Player"] + st.session_state.off_visible_stats

                    if len(cols_to_show) == 1:
                        st.info("No stats from the selected categories are present in the chosen file(s).")
                        st.dataframe(rates_df[["Player"]], width="stretch")
                    else:
                        display_df = rates_df[cols_to_show].copy()
                        for c in display_df.columns:
                            if c == "Player":
                                continue
                            display_df[c] = pd.to_numeric(display_df[c], errors="coerce")

                        stats_for_players = [s for s in show_stats_ordered if s in rates_df.columns and s in st.session_state.off_visible_stats]
                        cols = [("", "Player")]
                        for s in stats_for_players:
                            cols.append((s, "Success Rate"))
                            cols.append((s, "Opportunities"))
                        mcols = pd.MultiIndex.from_tuples(cols)

                        player_counts = compute_offense_player_counts(selected_paths, stats_for_players)
                        player_counts = player_counts.set_index("Player") if not player_counts.empty else pd.DataFrame()

                        combined_rows = []
                        for _, prow in display_df.iterrows():
                            player = prow["Player"]
                            row = [player]
                            for s in stats_for_players:
                                pct = prow.get(s, np.nan)
                                opp = (
                                    player_counts.loc[player, f"{s} Opportunities"]
                                    if (player in player_counts.index and f"{s} Opportunities" in player_counts.columns)
                                    else np.nan
                                )
                                row.append(pct)
                                row.append(int(opp) if not pd.isna(opp) else np.nan)
                            combined_rows.append(row)

                        combined_df = pd.DataFrame(combined_rows, columns=mcols)

                        def fmt_pct(v):
                            return "—" if pd.isna(v) else f"{v:.1f}%"
                        def fmt_opp(v):
                            return "—" if pd.isna(v) else f"{int(v):d}"
                        fmt_map = {}
                        for s in stats_for_players:
                            fmt_map[(s, "Success Rate")] = fmt_pct
                            fmt_map[(s, "Opportunities")] = fmt_opp
                        fmt_map[("", "Player")] = lambda v: v

                        styler = combined_df.style.format(fmt_map)

                        pct_cols = [(s, "Success Rate") for s in stats_for_players]
                        def _cell_style(v):
                            if pd.isna(v):
                                return ""
                            try:
                                val = float(v)
                            except Exception:
                                return ""
                            if val >= 90:
                                return "background-color: #0f5132; color: #ffffff; font-weight: 600"
                            if val < 70:
                                return "background-color: #6b0f0f; color: #ffffff; font-weight: 600"
                            return "background-color: #b07b00; color: #000000; font-weight: 600"
                        if pct_cols:
                            styler = styler.applymap(_cell_style, subset=pct_cols)
                        st.dataframe(styler, width="stretch")

                        display_stats_for_team = [s for s in show_stats_ordered if s in rates_df.columns and s in st.session_state.off_visible_stats]
                        team_df = compute_offense_team_summary(selected_paths, display_stats_for_team)
                        counts_df = compute_offense_team_counts(selected_paths, display_stats_for_team)

                        if (not team_df.empty) or (not counts_df.empty):
                            cols = [("", "Player")]
                            for s in display_stats_for_team:
                                cols.append((s, "Success Rate"))
                                cols.append((s, "Opportunities"))
                            mcols = pd.MultiIndex.from_tuples(cols)

                            team_row = ["Team"]
                            for s in display_stats_for_team:
                                pct = team_df.iloc[0].get(s, np.nan) if (not team_df.empty and s in team_df.columns) else np.nan
                                opp = counts_df.iloc[0].get(f"{s} Opportunities", np.nan) if (not counts_df.empty and f"{s} Opportunities" in counts_df.columns) else np.nan
                                team_row.append(pct)
                                team_row.append(int(opp) if not pd.isna(opp) else np.nan)
                            # team shooting fouls and per-100 poss
                            team_sf = team_df.iloc[0].get("Shooting Fouls", np.nan) if (not team_df.empty and "Shooting Fouls" in team_df.columns) else np.nan
                            team_sf_per100 = team_df.iloc[0].get("Shooting Fouls / 100 Poss", np.nan) if (not team_df.empty and "Shooting Fouls / 100 Poss" in team_df.columns) else np.nan
                            team_row.append(int(team_sf) if not pd.isna(team_sf) else np.nan)
                            team_row.append(team_sf_per100)
                            # make sure team_row length matches number of columns (mcols)
                            needed = len(mcols)
                            if len(team_row) < needed:
                                team_row += [np.nan] * (needed - len(team_row))
                            elif len(team_row) > needed:
                                team_row = team_row[:needed]

                            combined_df_team = pd.DataFrame([team_row], columns=mcols)

                            def fmt_pct_team(v):
                                return "—" if pd.isna(v) else f"{v:.1f}%"
                            def fmt_opp_team(v):
                                return "—" if pd.isna(v) else f"{int(v):d}"

                            fmt_map_team = {}
                            for s in display_stats_for_team:
                                fmt_map_team[(s, "Success Rate")] = fmt_pct_team
                                fmt_map_team[(s, "Opportunities")] = fmt_opp_team
                            fmt_map_team[("", "Player")] = lambda v: v

                            styler_team = combined_df_team.style.format(fmt_map_team)

                            pct_cols_team = [(s, "Success Rate") for s in display_stats_for_team]
                            def _cell_style_team(v):
                                if pd.isna(v):
                                    return ""
                                try:
                                    val = float(v)
                                except Exception:
                                    return ""
                                if val >= 90:
                                    return "background-color: #0f5132; color: #ffffff; font-weight: 600"
                                if val < 70:
                                    return "background-color: #6b0f0f; color: #ffffff; font-weight: 600"
                                return "background-color: #b07b00; color: #000000; font-weight: 600"
                            if pct_cols_team:
                                styler_team = styler_team.applymap(_cell_style_team, subset=pct_cols_team)
                            st.markdown("### Team Summary")
                            st.dataframe(styler_team, width="stretch")

    with tab_special:
        st.subheader("Special Teams")
        def_paths = list_defense_files()
        def_names = [p.name for p in def_paths]
        if not def_paths:
            st.info("No files found in Defense/ folder.")
        else:
            selected = st.multiselect("Dates", options=def_names, default=def_names, key="def_files_special")
            selected_paths = [p for p in def_paths if p.name in selected]
            if not selected_paths:
                st.info("Select one or more defense files to view special teams rates.")
            else:
                stats_df = compute_def_special_rates_for_paths(selected_paths)
                if stats_df.empty:
                    st.info("No special-team stat rows found in selected files.")
                else:
                    # player counts
                    stats_list = ["Box Out", "Forced Turnover"]
                    counts_df = compute_def_special_player_counts(selected_paths, stats_list)

                    # Build display MultiIndex like other tables
                    cols = [("", "Player")]
                    for s in stats_list:
                        cols.append((s, "Success Rate"))
                        cols.append((s, "Opportunities"))
                    # shooting fouls: show Count, Possessions, and % of Poss (sf / poss)
                    cols.append(("Shooting Fouls", "Count"))
                    cols.append(("Shooting Fouls", "Possessions"))
                    cols.append(("Shooting Fouls", "% of Poss"))
                    mcols = pd.MultiIndex.from_tuples(cols)

                    player_counts = counts_df.set_index("Player") if not counts_df.empty else pd.DataFrame()

                    combined_rows = []
                    for _, r in stats_df.iterrows():
                        player = r["Player"]
                        row = [player]
                        for s in stats_list:
                            pct = r.get(s, np.nan)
                            opp = player_counts.loc[player, f"{s} Opportunities"] if (player in player_counts.index and f"{s} Opportunities" in player_counts.columns) else np.nan
                            row.append(pct)
                            row.append(int(opp) if not pd.isna(opp) else np.nan)

                        # shooting fouls: prefer counts_df values, fall back to stats_df values from specialteams
                        if (player in player_counts.index) and "Shooting Fouls" in player_counts.columns:
                            sf_count = int(player_counts.loc[player, "Shooting Fouls"])
                        else:
                            sf_count = int(r.get("Shooting Fouls", 0) if not pd.isna(r.get("Shooting Fouls", np.nan)) else 0)

                        if (player in player_counts.index) and "Possessions" in player_counts.columns:
                            poss = int(player_counts.loc[player, "Possessions"])
                        else:
                            poss = int(r.get("Possessions", 0) if not pd.isna(r.get("Possessions", np.nan)) else 0)

                        # append raw count, raw possessions, and per-pos (rounded 1 decimal)
                        # express as per 100 possessions
                        sf_per_pos = (sf_count / poss) * 100 if (poss and poss > 0) else np.nan
                        row.append(sf_count if sf_count is not None else np.nan)
                        row.append(poss if poss is not None else np.nan)
                        row.append(round(sf_per_pos, 1) if not pd.isna(sf_per_pos) else np.nan)

                        combined_rows.append(row)

                    combined_df = pd.DataFrame(combined_rows, columns=mcols)

                    def fmt_pct(v):
                        return "—" if pd.isna(v) else f"{v:.1f}%"
                    def fmt_opp(v):
                        return "—" if pd.isna(v) else f"{int(v):d}"
                    def fmt_sf_count(v):
                        return "—" if pd.isna(v) else f"{int(v):d}"
                    def fmt_sf_poss(v):
                        return "—" if pd.isna(v) else f"{int(v):d}"
                    def fmt_sf_perpos(v):
                        return "—" if pd.isna(v) else f"{v:.1f}"
                    fmt_map = {("", "Player"): (lambda v: v)}
                    for s in stats_list:
                        fmt_map[(s, "Success Rate")] = fmt_pct
                        fmt_map[(s, "Opportunities")] = fmt_opp
                    fmt_map[("Shooting Fouls", "Count")] = fmt_sf_count
                    fmt_map[("Shooting Fouls", "Possessions")] = fmt_sf_poss
                    fmt_map[("Shooting Fouls", "% of Poss")] = fmt_sf_perpos

                    styler = combined_df.style.format(fmt_map)
                    pct_cols = [(s, "Success Rate") for s in stats_list]

                    def _cell_style(v):
                        if pd.isna(v):
                            return ""
                        try:
                            val = float(v)
                        except Exception:
                            return ""
                        if val >= 90:
                            return "background-color: #0f5132; color: #ffffff; font-weight: 600"
                        if val < 70:
                            return "background-color: #6b0f0f; color: #ffffff; font-weight: 600"
                        return "background-color: #b07b00; color: #000000; font-weight: 600"

                    if pct_cols:
                        styler = styler.applymap(_cell_style, subset=pct_cols)
                    st.dataframe(styler, width="stretch")

                    # Team summary (Box Out / Forced Turnover)
                    team_df = compute_def_special_team_summary(selected_paths, stats_list)
                    if not team_df.empty:
                        team_cols = [("", "Player")]
                        for s in stats_list:
                            team_cols.append((s, "Success Rate"))
                            team_cols.append((s, "Opportunities"))
                        # shooting fouls: show Count, Possessions, and % of Poss (sf / poss)
                        team_cols.append(("Shooting Fouls", "Count"))
                        team_cols.append(("Shooting Fouls", "Possessions"))
                        team_cols.append(("Shooting Fouls", "% of Poss"))
                        mcols_team = pd.MultiIndex.from_tuples(team_cols)

                        # build combined team row using team_df and counts
                        counts_team = compute_def_special_player_counts(selected_paths, stats_list)

                        team_row = ["Team"]
                        for s in stats_list:
                            pct = team_df.iloc[0].get(s, np.nan) if s in team_df.columns else np.nan
                            if not counts_team.empty and f"{s} Opportunities" in counts_team.columns:
                                opp = counts_team[f"{s} Opportunities"].sum()
                                opp = int(opp) if opp > 0 else np.nan
                            else:
                                opp = np.nan
                            team_row.append(pct)
                            team_row.append(opp)

                        # team shooting fouls count and possessions and per-pos: prefer counts_team
                        team_sf = team_df.iloc[0].get("Shooting Fouls", np.nan) if "Shooting Fouls" in team_df.columns else 0
                        if not counts_team.empty and "Possessions" in counts_team.columns:
                            team_poss = int(counts_team["Possessions"].sum())
                        else:
                            # fallback: sum DefPoss from agg for special players
                            special_players = stats_df["Player"].tolist()
                            team_poss = int(agg.loc[agg["Player"].isin(special_players), "DefPoss"].sum()) if not agg.empty else 0
                        team_sf = int(team_sf) if not pd.isna(team_sf) else 0
                        # express team SF as per 100 possessions
                        team_sf_perpos = (team_sf / team_poss) * 100 if (team_poss and team_poss > 0) else np.nan
                        team_row.append(team_sf)
                        team_row.append(team_poss)
                        team_row.append(round(team_sf_perpos, 1) if not pd.isna(team_sf_perpos) else np.nan)

                        # ensure team_row length matches number of columns (mcols_team)
                        needed = len(mcols_team)
                        if len(team_row) < needed:
                            team_row += [np.nan] * (needed - len(team_row))
                        elif len(team_row) > needed:
                            team_row = team_row[:needed]

                        combined_team_df = pd.DataFrame([team_row], columns=mcols_team)

                        def fmt_pct_team(v):
                            return "—" if pd.isna(v) else f"{v:.1f}%"
                        def fmt_opp_team(v):
                            return "—" if pd.isna(v) else f"{int(v):d}"
                        def fmt_sf_count_team(v):
                            return "—" if pd.isna(v) else f"{int(v):d}"
                        def fmt_sf_poss_team(v):
                            return "—" if pd.isna(v) else f"{int(v):d}"
                        def fmt_sf_team_perpos(v):
                            return "—" if pd.isna(v) else f"{v:.1f}"

                        fmt_map_team = {("", "Player"): (lambda v: v)}
                        for s in stats_list:
                            fmt_map_team[(s, "Success Rate")] = fmt_pct_team
                            fmt_map_team[(s, "Opportunities")] = fmt_opp_team
                        fmt_map_team[("Shooting Fouls", "Count")] = fmt_sf_count_team
                        fmt_map_team[("Shooting Fouls", "Possessions")] = fmt_sf_poss_team
                        fmt_map_team[("Shooting Fouls", "% of Poss")] = fmt_sf_team_perpos

                        styler_team = combined_team_df.style.format(fmt_map_team)

                        pct_cols_team = [(s, "Success Rate") for s in stats_list]
                        def _cell_style_team(v):
                            if pd.isna(v):
                                return ""
                            try:
                                val = float(v)
                            except Exception:
                                return ""
                            if val >= 90:
                                return "background-color: #0f5132; color: #ffffff; font-weight: 600"
                            if val < 70:
                                return "background-color: #6b0f0f; color: #ffffff; font-weight: 600"
                            return "background-color: #b07b00; color: #000000; font-weight: 600"
                        if pct_cols_team:
                            styler_team = styler_team.applymap(_cell_style_team, subset=pct_cols_team)
                        st.markdown("### Team Summary")
                        st.dataframe(styler_team, width="stretch")

    with tab_shot_chart:
        st.subheader("Shot Chart Analytics")
        st.caption("Shooting statistics by zone and player")
        
        # Get shot chart files
        shot_chart_paths = list_shot_chart_files()
        shot_chart_names = [p.name for p in shot_chart_paths]
        
        if not shot_chart_paths:
            st.info("No files found in Shot Chart/ folder.")
        else:
            selected_shot_chart = st.multiselect(
                "Shot Chart Dates", 
                options=shot_chart_names, 
                default=shot_chart_names, 
                key="shot_chart_files"
            )
            selected_shot_chart_paths = [p for p in shot_chart_paths if p.name in selected_shot_chart]
            
            if not selected_shot_chart_paths:
                st.info("Select one or more shot chart files to view statistics.")
            else:
                # Get individual player stats
                player_stats = compute_shot_chart_stats_for_paths(selected_shot_chart_paths)
                
                if player_stats.empty:
                    st.info("No shot chart data found in selected files.")
                else:
                    # Table View Selector
                    st.markdown("### Shot Chart Data Views")
                    view_option = st.selectbox(
                        "View Options:",
                        ["Efficiency (EFG%, TS%, PPP)", "Shot Types", "Select your own columns"],
                        key="view_option_shot_chart"
                    )
                    
                    # Filter out team rows for individual display (used by both team summary and individual display)
                    individual_stats = player_stats[~player_stats["Player"].isin(["Team 1", "Team 2"])].copy()
                    
                    # Get Team 1 and Team 2 rows for breakdown
                    team_rows = player_stats[player_stats["Player"].isin(["Team 1", "Team 2"])].copy()
                    
                    # Overall Team Statistics (updates with view selection)
                    st.markdown("### Overall Team Statistics")
                    
                    # Calculate overall team stats by summing all individual players
                    # Calculate team totals (exclude percentage columns from sum)
                    percentage_cols = ['ShootingPct', 'EffectiveFGPct', 'TrueShootingPct', 'RedZonePct', 'ToughTwoPct', 
                                     'InsideOut3Pct', 'NonInsideOut3Pct', 'TwoPointPct', 'ThreePointPct']
                    numeric_cols_to_sum = individual_stats.select_dtypes(include=[np.number]).columns.difference(percentage_cols)
                    team_totals = individual_stats[numeric_cols_to_sum].sum()
                    team_totals["Player"] = "Overall Team"
                    
                    # Recalculate team percentages as whole numbers
                    if team_totals["TotalShots"] > 0:
                        team_totals["ShootingPct"] = round((team_totals["ShotsMade"] / team_totals["TotalShots"] * 100), 0)
                    else:
                        team_totals["ShootingPct"] = np.nan
                        
                    if team_totals["FieldGoalAttempts"] > 0:
                        team_totals["EffectiveFGPct"] = round(((team_totals["FieldGoalMade"] + 0.5 * team_totals["ThreePointMade"]) / team_totals["FieldGoalAttempts"] * 100), 0)
                    else:
                        team_totals["EffectiveFGPct"] = np.nan
                        
                    ts_denominator = 2 * (team_totals["FieldGoalAttempts"] + 0.44 * team_totals["EstimatedFTA"])
                    if ts_denominator > 0:
                        team_totals["TrueShootingPct"] = round((team_totals["Points"] / ts_denominator * 100), 0)
                    else:
                        team_totals["TrueShootingPct"] = np.nan
                        
                    if team_totals["Possessions"] > 0:
                        team_totals["PointsPerPossession"] = round((team_totals["Points"] / team_totals["Possessions"]), 2)
                    else:
                        team_totals["PointsPerPossession"] = np.nan
                    
                    # Recalculate new efficiency metrics from raw totals
                    # Red Zone %
                    rz_attempts = team_totals.get("shot chart:RZ +", 0) + team_totals.get("shot chart:RZ -", 0)
                    if rz_attempts > 0:
                        team_totals["RedZonePct"] = round((team_totals.get("shot chart:RZ +", 0) / rz_attempts * 100), 0)
                    else:
                        team_totals["RedZonePct"] = np.nan
                    
                    # Tough Two %
                    tt_attempts = team_totals.get("shot chart:TT +", 0) + team_totals.get("shot chart:TT -", 0)
                    if tt_attempts > 0:
                        team_totals["ToughTwoPct"] = round((team_totals.get("shot chart:TT +", 0) / tt_attempts * 100), 0)
                    else:
                        team_totals["ToughTwoPct"] = np.nan
                    
                    # Inside-Out 3 %
                    io3_attempts = team_totals.get("shot chart:IO3 +", 0) + team_totals.get("shot chart:IO3 -", 0)
                    if io3_attempts > 0:
                        team_totals["InsideOut3Pct"] = round((team_totals.get("shot chart:IO3 +", 0) / io3_attempts * 100), 0)
                    else:
                        team_totals["InsideOut3Pct"] = np.nan
                    
                    # Non Inside-Out 3 %
                    nio3_attempts = team_totals.get("shot chart:NIO3 +", 0) + team_totals.get("shot chart:NIO3 -", 0)
                    if nio3_attempts > 0:
                        team_totals["NonInsideOut3Pct"] = round((team_totals.get("shot chart:NIO3 +", 0) / nio3_attempts * 100), 0)
                    else:
                        team_totals["NonInsideOut3Pct"] = np.nan
                    
                    # 2PT% and 3PT%
                    if team_totals["TwoPointAttempts"] > 0:
                        team_totals["TwoPointPct"] = round((team_totals["TwoPointMade"] / team_totals["TwoPointAttempts"] * 100), 0)
                    else:
                        team_totals["TwoPointPct"] = np.nan
                    
                    if team_totals["ThreePointAttempts"] > 0:
                        team_totals["ThreePointPct"] = round((team_totals["ThreePointMade"] / team_totals["ThreePointAttempts"] * 100), 0)
                    else:
                        team_totals["ThreePointPct"] = np.nan
                    
                    # Create team display based on selected view
                    if view_option == "Efficiency (EFG%, TS%, PPP)":
                        team_display_cols = [
                            "Player", "TotalShots", "ShotsMade", "Points", "Possessions",
                            "EffectiveFGPct", "TrueShootingPct", "PointsPerPossession",
                            "RedZonePct", "ToughTwoPct", "InsideOut3Pct", "NonInsideOut3Pct"
                        ]
                    elif view_option == "Shot Types":
                        team_display_cols = [
                            "Player", "TotalShots", "TwoPointMade", "TwoPointAttempts", "ThreePointMade", "ThreePointAttempts",
                            "shot chart:RZ +", "shot chart:RZ -", "shot chart:TT +", "shot chart:TT -",
                            "shot chart:IO3 +", "shot chart:IO3 -", "shot chart:NIO3 +", "shot chart:NIO3 -",
                            "shot chart:FT 2", "shot chart:FT 3", "shot chart:Turnover -"
                        ]
                    else:  # Custom view - use same columns as individual players will show
                        all_available_cols = [col for col in individual_stats.columns if col != "Player"]
                        selected_cols = st.multiselect(
                            "Select columns to display:",
                            options=all_available_cols,
                            default=["TotalShots", "ShotsMade", "RedZonePct", "ToughTwoPct", "TwoPointPct", "ThreePointPct"],
                            help="Choose which columns to show in the table",
                            key="team_columns"
                        )
                        team_display_cols = ["Player"] + selected_cols
                    
                    # Filter team columns that exist
                    available_team_cols = [col for col in team_display_cols if col in team_totals.index or col == "Player"]
                    team_df = pd.DataFrame([team_totals[available_team_cols[1:]]]).reset_index(drop=True)
                    team_df.insert(0, "Player", "Overall Team")
                    team_df.columns = available_team_cols
                    
                    # Format team dataframe
                    team_format_dict = {
                        "TotalShots": "{:.0f}",
                        "ShotsMade": "{:.0f}",
                        "ShotsMissed": "{:.0f}",
                        "Points": "{:.1f}",
                        "Possessions": "{:.0f}",
                        "PointsPerPossession": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                        "ShootingPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "EffectiveFGPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "TrueShootingPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "RedZonePct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "ToughTwoPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "InsideOut3Pct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "NonInsideOut3Pct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "TwoPointPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "ThreePointPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                        "TwoPointMade": "{:.0f}",
                        "TwoPointAttempts": "{:.0f}",
                        "ThreePointMade": "{:.0f}",
                        "ThreePointAttempts": "{:.0f}",
                        "shot chart:RZ +": "{:.0f}",
                        "shot chart:RZ -": "{:.0f}",
                        "shot chart:IO3 +": "{:.0f}",
                        "shot chart:IO3 -": "{:.0f}",
                        "shot chart:NIO3 +": "{:.0f}",
                        "shot chart:NIO3 -": "{:.0f}",
                        "shot chart:TT +": "{:.0f}",
                        "shot chart:TT -": "{:.0f}",
                        "shot chart:FT 2": "{:.0f}",
                        "shot chart:FT 3": "{:.0f}",
                        "shot chart:Turnover -": "{:.0f}",
                        "TotalPoints": "{:.1f}",
                        "offense_lab": "{:.0f}"
                    }
                    
                    # Only include format rules for columns that exist
                    active_team_format_dict = {k: v for k, v in team_format_dict.items() if k in team_df.columns}
                    team_styler = team_df.style.format(active_team_format_dict, na_rep="—")
                    
                    st.dataframe(team_styler, use_container_width=True)
                    
                    # Team 1 vs Team 2 Breakdown
                    if not team_rows.empty:
                        st.markdown("#### Team 1 vs Team 2 Breakdown")
                        
                        # Process team rows for display based on selected view
                        team_breakdown_df = team_rows.copy()
                        
                        # Create team breakdown display based on selected view
                        if view_option == "Efficiency (EFG%, TS%, PPP)":
                            team_breakdown_cols = [
                                "Player", "TotalShots", "ShotsMade", "Points", "Possessions",
                                "EffectiveFGPct", "TrueShootingPct", "PointsPerPossession",
                                "RedZonePct", "ToughTwoPct", "InsideOut3Pct", "NonInsideOut3Pct"
                            ]
                        elif view_option == "Shot Types":
                            team_breakdown_cols = [
                                "Player", "TotalShots", "TwoPointMade", "TwoPointAttempts", "ThreePointMade", "ThreePointAttempts",
                                "shot chart:RZ +", "shot chart:RZ -", "shot chart:TT +", "shot chart:TT -",
                                "shot chart:IO3 +", "shot chart:IO3 -", "shot chart:NIO3 +", "shot chart:NIO3 -",
                                "shot chart:FT 2", "shot chart:FT 3", "shot chart:Turnover -"
                            ]
                        else:  # Custom view
                            if 'team_columns' in st.session_state:
                                selected_cols = st.session_state.team_columns
                            else:
                                selected_cols = ["TotalShots", "ShotsMade", "RedZonePct", "ToughTwoPct", "TwoPointPct", "ThreePointPct"]
                            team_breakdown_cols = ["Player"] + selected_cols
                        
                        # Filter columns that exist in team breakdown
                        available_breakdown_cols = [col for col in team_breakdown_cols if col in team_breakdown_df.columns]
                        team_breakdown_display = team_breakdown_df[available_breakdown_cols].copy()
                        
                        # Format team breakdown dataframe using same formatting rules
                        active_breakdown_format = {k: v for k, v in team_format_dict.items() if k in team_breakdown_display.columns}
                        breakdown_styler = team_breakdown_display.style.format(active_breakdown_format, na_rep="—")
                        
                        st.dataframe(breakdown_styler, use_container_width=True)
                    
                    # Individual Player Statistics
                    st.markdown("### Individual Player Statistics")
                    
                    if not individual_stats.empty:
                        # Define column sets for different views
                        if view_option == "Efficiency (EFG%, TS%, PPP)":
                            display_cols = [
                                "Player", "TotalShots", "ShotsMade", "Points", 
                                "EffectiveFGPct", "TrueShootingPct", "PointsPerPossession",
                                "RedZonePct", "ToughTwoPct", "InsideOut3Pct", "NonInsideOut3Pct"
                            ]
                        elif view_option == "Shot Types":
                            display_cols = [
                                "Player", "TotalShots", "TwoPointMade", "TwoPointAttempts", "ThreePointMade", "ThreePointAttempts",
                                "shot chart:RZ +", "shot chart:RZ -", "shot chart:TT +", "shot chart:TT -",
                                "shot chart:IO3 +", "shot chart:IO3 -", "shot chart:NIO3 +", "shot chart:NIO3 -",
                                "shot chart:FT 2", "shot chart:FT 3", "shot chart:Turnover -"
                            ]
                        else:  # "Select your own columns"
                            # Use the same columns selected for team display
                            if 'team_columns' in st.session_state:
                                selected_cols = st.session_state.team_columns
                            else:
                                selected_cols = ["TotalShots", "ShotsMade", "RedZonePct", "ToughTwoPct", "TwoPointPct", "ThreePointPct"]
                            display_cols = ["Player"] + selected_cols
                        
                        # Filter columns that exist in the data
                        available_cols = [col for col in display_cols if col in individual_stats.columns]
                        display_df = individual_stats[available_cols].copy()
                        
                        # Sort by appropriate metric based on view
                        if view_option == "Efficiency (EFG%, TS%, PPP)" and "PointsPerPossession" in display_df.columns:
                            display_df = display_df.sort_values("PointsPerPossession", ascending=False)
                        elif "TrueShootingPct" in display_df.columns:
                            display_df = display_df.sort_values("TrueShootingPct", ascending=False)
                        elif "TotalPoints" in display_df.columns:
                            display_df = display_df.sort_values("TotalPoints", ascending=False)
                        
                        # Format the dataframe
                        format_dict = {
                            "TotalShots": "{:.0f}",
                            "ShotsMade": "{:.0f}",
                            "ShotsMissed": "{:.0f}",
                            "Points": "{:.1f}",
                            "Possessions": "{:.0f}",
                            "PointsPerPossession": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
                            "ShootingPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "EffectiveFGPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "TrueShootingPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "RedZonePct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "ToughTwoPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "InsideOut3Pct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "NonInsideOut3Pct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "TwoPointPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "ThreePointPct": lambda v: "—" if pd.isna(v) else f"{int(v)}%",
                            "TwoPointMade": "{:.0f}",
                            "TwoPointAttempts": "{:.0f}",
                            "ThreePointMade": "{:.0f}",
                            "ThreePointAttempts": "{:.0f}",
                            "shot chart:RZ +": "{:.0f}",
                            "shot chart:RZ -": "{:.0f}",
                            "shot chart:IO3 +": "{:.0f}",
                            "shot chart:IO3 -": "{:.0f}",
                            "shot chart:NIO3 +": "{:.0f}",
                            "shot chart:NIO3 -": "{:.0f}",
                            "shot chart:TT +": "{:.0f}",
                            "shot chart:TT -": "{:.0f}",
                            "shot chart:FT 2": "{:.0f}",
                            "shot chart:FT 3": "{:.0f}",
                            "shot chart:Turnover -": "{:.0f}",
                            "TotalPoints": "{:.1f}",
                            "offense_lab": "{:.0f}"
                        }
                        
                        # Only include format rules for columns that exist
                        active_format_dict = {k: v for k, v in format_dict.items() if k in display_df.columns}
                        
                        styler = display_df.style.format(active_format_dict, na_rep="—")
                        
                        st.dataframe(styler, use_container_width=True)
                    
                    # Advanced Shooting Efficiency Charts
                    if not individual_stats.empty:
                        # Create two columns for charts
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            if "EffectiveFGPct" in individual_stats.columns:
                                efg_df = individual_stats[individual_stats["EffectiveFGPct"].notna()].copy()
                                if not efg_df.empty:
                                    fig_efg = px.bar(
                                        efg_df.sort_values("EffectiveFGPct", ascending=True),
                                        x="Player",
                                        y="EffectiveFGPct",
                                        color="EffectiveFGPct",
                                        color_continuous_scale="Blues",
                                        title="Effective Field Goal % by Player",
                                        height=400
                                    )
                                    fig_efg.update_layout(yaxis_title="eFG%", xaxis_title="")
                                    st.plotly_chart(fig_efg, use_container_width=True)
                        
                        with chart_col2:
                            if "TrueShootingPct" in individual_stats.columns:
                                ts_df = individual_stats[individual_stats["TrueShootingPct"].notna()].copy()
                                if not ts_df.empty:
                                    fig_ts = px.bar(
                                        ts_df.sort_values("TrueShootingPct", ascending=True),
                                        x="Player",
                                        y="TrueShootingPct",
                                        color="TrueShootingPct",
                                        color_continuous_scale="Viridis",
                                        title="True Shooting % by Player",
                                        height=400
                                    )
                                    fig_ts.update_layout(yaxis_title="TS%", xaxis_title="")
                                    st.plotly_chart(fig_ts, use_container_width=True)

    with tab_counting:
        st.subheader("Defensive Ratings")
        st.caption("Defensive rating = points allowed per 100 defensive possessions")
        
        # Get file lists for both offense and defense
        offense_paths = list_offense_files()
        defense_paths = list_defense_files()
        offense_names = [p.name for p in offense_paths]
        defense_names = [p.name for p in defense_paths]
        
        # Create two columns for file selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Offense Files**")
            if not offense_paths:
                st.info("No files found in Offense/ folder.")
                selected_offense = []
            else:
                selected_offense = st.multiselect(
                    "Offense Dates", 
                    options=offense_names, 
                    default=offense_names, 
                    key="counting_off_files"
                )
        
        with col2:
            st.write("**Defense Files**")
            if not defense_paths:
                st.info("No files found in Defense/ folder.")
                selected_defense = []
            else:
                selected_defense = st.multiselect(
                    "Defense Dates", 
                    options=defense_names, 
                    default=defense_names, 
                    key="counting_def_files"
                )
        
        # Process selected files
        selected_offense_paths = [p for p in offense_paths if p.name in selected_offense]
        selected_defense_paths = [p for p in defense_paths if p.name in selected_defense]
        
        # Process offense files
        offense_dfs = []
        for path in selected_offense_paths:
            try:
                df = process_offense_file(path, path.name)
                if df is not None and not df.empty:
                    offense_dfs.append(df)
            except Exception as e:
                st.warning(f"Error processing offense file {path.name}: {e}")
        
        # Process defense files
        defense_dfs = []
        for path in selected_defense_paths:
            try:
                df = process_defense_file(path, path.name)
                if df is not None and not df.empty:
                    defense_dfs.append(df)
            except Exception as e:
                st.warning(f"Error processing defense file {path.name}: {e}")
        
        # Combine processed data
        processed_off_df = pd.concat(offense_dfs, ignore_index=True) if offense_dfs else pd.DataFrame()
        processed_def_df = pd.concat(defense_dfs, ignore_index=True) if defense_dfs else pd.DataFrame()
        
        # Recompute aggregation with selected data
        filtered_agg = aggregate_by_player(processed_off_df, processed_def_df)
        
        if filtered_agg.empty or filtered_agg["DefPoss"].sum() == 0:
            st.info("No defensive data found for selected files.")
        else:
            # Display player defensive ratings table
            def_table_raw = filtered_agg[["Player", "DefPoss", "DefPoints", "ShotsAllowed", "DefRating"]].copy()
            display_raw = def_table_raw[def_table_raw["DefRating"].notna()].copy()
            sorted_raw = display_raw.sort_values(by="DefRating", ascending=True, na_position="last").reset_index(drop=True)
     
            # create a Styler to format DefRating for display while keeping numeric dtype for sorting logic
            styler = sorted_raw.style.format({
                "DefPoss": "{:.0f}",
                "DefPoints": "{:.0f}",
                "ShotsAllowed": "{:.0f}",
                "DefRating": lambda v: "—" if pd.isna(v) else f"{v:.1f}"
            }, na_rep="—")
     
            st.dataframe(styler, width="stretch")
     
            # -------------------
            # Team Defensive Summary (mapped names: Green / White)
            # -------------------
            team_keys = ["Green", "White"]
            team_rows = processed_def_df[processed_def_df["Player"].isin(team_keys)].copy() if not processed_def_df.empty else pd.DataFrame()
            if not team_rows.empty:
                team_agg = team_rows.groupby("Player", as_index=False)[["DefPoss", "DefPoints", "ShotsAllowed"]].sum()
                team_agg["DefRating"] = np.where(team_agg["DefPoss"] > 0, (team_agg["DefPoints"] / team_agg["DefPoss"] * 100).round(1), np.nan)
                # keep requested order
                team_agg["Player"] = pd.Categorical(team_agg["Player"], categories=team_keys, ordered=True)
                team_agg = team_agg.sort_values("Player").reset_index(drop=True)
                team_styler = team_agg.style.format({
                    "DefPoss": "{:.0f}",
                    "DefPoints": "{:.0f}",
                    "ShotsAllowed": "{:.0f}",
                    "DefRating": lambda v: "—" if pd.isna(v) else f"{v:.1f}"
                }, na_rep="—")
                st.markdown("### Team Defensive Summary")
                st.dataframe(team_styler, width="stretch")
            else:
                st.info("No Green / White team summary rows found in selected defense files.")
     
            # Defensive Rating Chart
            chart_df = filtered_agg[filtered_agg["DefPoss"] > 0].sort_values("DefRating", ascending=True)
            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x="Player",
                    y="DefRating",
                    color="DefRating",
                    color_continuous_scale="Blues",
                    title="Defensive Rating (points allowed per 100 defensive possessions)",
                    height=420
                )
                fig.update_layout(yaxis_title="Def Rating", xaxis_title="")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No defensive possessions found to chart.")
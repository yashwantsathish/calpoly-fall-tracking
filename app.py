# Cal Poly MBB — Summer 5v5 Ratings Dashboard
# Streamlit app to upload offensive & defensive CSVs, compute player ratings, and visualize with date filters

import re
import io
import math
import datetime as dt
from typing import List
from pathlib import Path
import streamlit as st

from constants import CALPOLY_GREEN, CALPOLY_GOLD, BG_LIGHT
from utils import read_folder
from offense import process_offense_file as proc_offense_file
from defense import process_defense_file as proc_defense_file
from utils import aggregate_by_player
from ui import render_ui

st.set_page_config(
    page_title="Cal Poly MBB: 2025 Fall Stat Site",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Cal Poly MBB: 2025 Fall Stat Site")

# Load Offense and Defense folders (if present)
off_df = read_folder("Offense", proc_offense_file)
def_df = read_folder("Defense", proc_defense_file)

# compute aggregation once and reuse in UI
agg = aggregate_by_player(off_df, def_df)

# hand off rendering to ui module
render_ui(off_df, def_df, agg)

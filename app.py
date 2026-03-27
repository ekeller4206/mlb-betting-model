import re
from datetime import date
import requests
import pandas as pd
import streamlit as st
from pybaseball import pitching_stats, team_batting, team_pitching

from datetime import date

def season_blend_weights(game_dt):
    m = game_dt.month

    if m <= 4:
        return 0.15, 0.85
    elif m == 5:
        return 0.35, 0.65
    elif m == 6:
        return 0.50, 0.50
    elif m == 7:
        return 0.65, 0.35
    else:
        return 0.80, 0.20

def blend_vals(current_val, prev_val, current_w, prev_w, fallback):
    vals = []
    weights = []

    if current_val is not None:
        vals.append(current_val)
        weights.append(current_w)

    if prev_val is not None:
        vals.append(prev_val)
        weights.append(prev_w)

    if not vals:
        return fallback

    return sum(v*w for v, w in zip(vals, weights)) / sum(weights)

st.set_page_config(page_title="MLB Betting Model", layout="wide")
st.title("MLB Betting Model")
st.caption("Auto-fill stats + DraftKings odds, fair odds, edge, Kelly, and best bet")

# ----------------------------
# Helpers
# ----------------------------
TEAM_CODE_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}

DEFAULT_PARK_FACTORS = {
    "Arizona Diamondbacks": 1.00,
    "Atlanta Braves": 1.00,
    "Baltimore Orioles": 1.00,
    "Boston Red Sox": 1.00,
    "Chicago Cubs": 1.00,
    "Chicago White Sox": 1.00,
    "Cincinnati Reds": 1.00,
    "Cleveland Guardians": 1.00,
    "Colorado Rockies": 1.10,
    "Detroit Tigers": 0.98,
    "Houston Astros": 1.00,
    "Kansas City Royals": 0.98,
    "Los Angeles Angels": 1.00,
    "Los Angeles Dodgers": 0.99,
    "Miami Marlins": 0.97,
    "Milwaukee Brewers": 1.01,
    "Minnesota Twins": 1.00,
    "New York Mets": 0.99,
    "New York Yankees": 1.03,
    "Athletics": 0.98,
    "Philadelphia Phillies": 1.02,
    "Pittsburgh Pirates": 0.98,
    "San Diego Padres": 0.93,
    "San Francisco Giants": 0.92,
    "Seattle Mariners": 0.96,
    "St. Louis Cardinals": 1.00,
    "Tampa Bay Rays": 0.98,
    "Texas Rangers": 1.03,
    "Toronto Blue Jays": 1.01,
    "Washington Nationals": 1.00,
}

def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def american_to_implied_prob(odds: float) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def prob_to_american(prob: float):
    if prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)

def kelly_fraction(prob: float, odds: float) -> float:
    if prob <= 0 or prob >= 1:
        return 0.0
    b = odds / 100 if odds > 0 else 100 / abs(odds)
    if b <= 0:
        return 0.0
    raw = ((b * prob) - (1 - prob)) / b
    return max(0.0, raw)

def moneyline_win_prob(
    away_sp_fip, home_sp_fip,
    away_sp_xfip, home_sp_xfip,
    away_woba, home_woba,
    away_wrcp, home_wrcp,
    park_factor
):
    away_pitch = (4.20 / ((away_sp_fip + away_sp_xfip) / 2))
    home_pitch = (4.20 / ((home_sp_fip + home_sp_xfip) / 2))

    away_off = ((away_woba / 0.315) * 0.55) + ((away_wrcp / 100) * 0.45)
    home_off = ((home_woba / 0.315) * 0.55) + ((home_wrcp / 100) * 0.45)

    away_strength = away_off * away_pitch
    home_strength = home_off * home_pitch

    away_base = away_strength / (away_strength + home_strength)
    home_base = 1 - away_base

    home_field = 0.015 * park_factor
    home_prob = clamp(home_base + home_field, 0.05, 0.95)
    away_prob = 1 - home_prob
    return away_prob, home_prob

def projected_runs(
    away_woba, home_woba,
    away_wrcp, home_wrcp,
    away_sp_fip, home_sp_fip,
    away_sp_xfip, home_sp_xfip,
    away_bullpen_xfip, home_bullpen_xfip,
    park_factor
):
    away_off_index = ((away_woba / 0.315) * 0.5) + ((away_wrcp / 100) * 0.5)
    home_off_index = ((home_woba / 0.315) * 0.5) + ((home_wrcp / 100) * 0.5)

    away_pitch_prev = 0.7 * ((home_sp_fip + home_sp_xfip) / 2) + 0.3 * home_bullpen_xfip
    home_pitch_prev = 0.7 * ((away_sp_fip + away_sp_xfip) / 2) + 0.3 * away_bullpen_xfip

    away_runs = 4.40 * away_off_index * (away_pitch_prev / 4.20) * park_factor
    home_runs = 4.40 * home_off_index * (home_pitch_prev / 4.20) * park_factor

    away_runs = clamp(away_runs, 1.5, 8.5)
    home_runs = clamp(home_runs, 1.5, 8.5)

    return away_runs, home_runs, away_runs + home_runs

def runline_cover_probs(away_win_prob, away_runs, home_runs):
    margin = away_runs - home_runs
    away_minus_1_5 = clamp(away_win_prob - 0.12 + (margin * 0.06), 0.05, 0.90)
    home_plus_1_5 = 1 - away_minus_1_5

    home_minus_1_5 = clamp((1 - away_win_prob) - 0.12 + ((-margin) * 0.06), 0.05, 0.90)
    away_plus_1_5 = 1 - home_minus_1_5

    return away_minus_1_5, home_plus_1_5, away_plus_1_5, home_minus_1_5

def over_under_probs(projected_total, total_line):
    diff = projected_total - total_line
    over_prob = clamp(0.50 + (diff * 0.09), 0.05, 0.95)
    under_prob = 1 - over_prob
    return over_prob, under_prob

# ----------------------------
# Web/data loaders
# ----------------------------
@st.cache_data(ttl=1800)
def load_schedule(game_date: str):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": game_date, "hydrate": "probablePitcher,team"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            away_prob = g["teams"]["away"].get("probablePitcher", {})
            home_prob = g["teams"]["home"].get("probablePitcher", {})
            games.append({
                "label": f"{away} @ {home}",
                "away_team": away,
                "home_team": home,
                "away_pitcher": away_prob.get("fullName", ""),
                "away_pitcher_id": away_prob.get("id"),
                "home_pitcher": home_prob.get("fullName", ""),
                "home_pitcher_id": home_prob.get("id"),
            })
    return games

@st.cache_data(ttl=21600)
def load_pitching_stats(season: int):
    return pitching_stats(season, qual=0)

@st.cache_data(ttl=21600)
def load_team_batting(season: int):
    return team_batting(season)

@st.cache_data(ttl=21600)
def load_team_pitching(season: int):
    return team_pitching(season)

@st.cache_data(ttl=21600)
def load_pitcher_hand(player_id: int):
    if not player_id:
        return ""
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    people = r.json().get("people", [])
    if not people:
        return ""
    return people[0].get("pitchHand", {}).get("code", "")

@st.cache_data(ttl=300)
def load_draftkings_odds():
    api_key = st.secrets.get("ODDS_API_KEY", None)
    if not api_key:
        return []

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    events = r.json()

    cleaned = []
    for event in events:
        home_team = event.get("home_team")
        away_team = next((t for t in event.get("teams", []) if t != home_team), None)

        dk = None
        for bookmaker in event.get("bookmakers", []):
            key = bookmaker.get("key", "").lower()
            title = bookmaker.get("title", "").lower()
            if key == "draftkings" or title == "draftkings":
                dk = bookmaker
                break

        if not dk:
            continue

        record = {
            "home_team": home_team,
            "away_team": away_team,
            "away_ml_odds": None,
            "home_ml_odds": None,
            "away_rl_odds": None,
            "home_rl_odds": None,
            "away_rl_point": None,
            "home_rl_point": None,
            "total_line": None,
            "over_odds": None,
            "under_odds": None,
        }

        for market in dk.get("markets", []):
            key = market.get("key")

            if key == "h2h":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == away_team:
                        record["away_ml_odds"] = outcome.get("price")
                    elif outcome.get("name") == home_team:
                        record["home_ml_odds"] = outcome.get("price")

            elif key == "spreads":
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if name == away_team:
                        record["away_rl_point"] = point
                        record["away_rl_odds"] = price
                    elif name == home_team:
                        record["home_rl_point"] = point
                        record["home_rl_odds"] = price

            elif key == "totals":
                for outcome in market.get("outcomes", []):
                    name = str(outcome.get("name", "")).lower()
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if point is not None:
                        record["total_line"] = point
                    if name == "over":
                        record["over_odds"] = price
                    elif name == "under":
                        record["under_odds"] = price

        cleaned.append(record)

    return cleaned

def find_pitcher_row(df: pd.DataFrame, pitcher_name: str):
    target = norm_name(pitcher_name)
    if not target or "Name" not in df.columns:
        return None
    names = df["Name"].astype(str)
    norms = names.map(norm_name)
    exact = df[norms == target]
    if not exact.empty:
        return exact.iloc[0]
    contains = df[norms.str.contains(target, na=False)]
    if not contains.empty:
        return contains.iloc[0]
    return None

def find_team_row(df: pd.DataFrame, team_name: str):
    code = TEAM_CODE_MAP.get(team_name, team_name)
    cols = [c for c in ["Team", "Tm"] if c in df.columns]
    if not cols:
        return None
    col = cols[0]
    s = df[col].astype(str)
    exact_code = df[s == code]
    if not exact_code.empty:
        return exact_code.iloc[0]
    exact_name = df[s.str.lower() == team_name.lower()]
    if not exact_name.empty:
        return exact_name.iloc[0]
    contains = df[s.str.contains(team_name.split()[-1], case=False, na=False)]
    if not contains.empty:
        return contains.iloc[0]
    return None

def safe_stat(row, col, fallback):
    try:
        val = row[col]
        if pd.isna(val):
            return fallback
        return float(val)
    except Exception:
        return fallback

def find_event_odds(away_team: str, home_team: str, odds_events: list):
    for event in odds_events:
        if event["away_team"] == away_team and event["home_team"] == home_team:
            return event
    return None

def auto_fill_for_game(game: dict, game_dt):
    season = game_dt.year
prev_season = season - 1
current_w, prev_w = season_blend_weights(game_dt)
p_df_cur = load_pitching_stats(season)
p_df_prev = load_pitching_stats(prev_season)
p_df = p_df_prev
tb_df = tb_df_prev
tp_df = tp_df_prev

tb_df_cur = load_team_batting(season)
tb_df_prev = load_team_batting(prev_season)

tp_df_cur = load_team_pitching(season)
tp_df_prev = load_team_pitching(prev_season)

    away_pitch_row = find_pitcher_row(p_df, game["away_pitcher"])
    home_pitch_row = find_pitcher_row(p_df, game["home_pitcher"])
    away_team_row = find_team_row(tb_df, game["away_team"])
    home_team_row = find_team_row(tb_df, game["home_team"])
    away_team_pitch_row = find_team_row(tp_df, game["away_team"])
    home_team_pitch_row = find_team_row(tp_df, game["home_team"])

    away_hand = load_pitcher_hand(game.get("away_pitcher_id"))
    home_hand = load_pitcher_hand(game.get("home_pitcher_id"))

    filled = {
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "away_sp": game["away_pitcher"] or "",
        "home_sp": game["home_pitcher"] or "",
        "away_sp_hand": away_hand or "",
        "home_sp_hand": home_hand or "",
        "away_sp_fip": safe_stat(away_pitch_row, "FIP", 4.20) if away_pitch_row is not None else 4.20,
        "away_sp_xfip": safe_stat(away_pitch_row, "xFIP", 4.20) if away_pitch_row is not None else 4.20,
        "home_sp_fip": safe_stat(home_pitch_row, "FIP", 4.20) if home_pitch_row is not None else 4.20,
        "home_sp_xfip": safe_stat(home_pitch_row, "xFIP", 4.20) if home_pitch_row is not None else 4.20,
        "away_bullpen_xfip": safe_stat(away_team_pitch_row, "xFIP", 4.20) if away_team_pitch_row is not None else 4.20,
        "home_bullpen_xfip": safe_stat(home_team_pitch_row, "xFIP", 4.20) if home_team_pitch_row is not None else 4.20,
        "away_woba": safe_stat(away_team_row, "wOBA", 0.315) if away_team_row is not None else 0.315,
        "away_wrcp": safe_stat(away_team_row, "wRC+", 100) if away_team_row is not None else 100,
        "home_woba": safe_stat(home_team_row, "wOBA", 0.315) if home_team_row is not None else 0.315,
        "home_wrcp": safe_stat(home_team_row, "wRC+", 100) if home_team_row is not None else 100,
        "park_factor": DEFAULT_PARK_FACTORS.get(game["home_team"], 1.00),
    }

    odds_events = load_draftkings_odds()
    odds = find_event_odds(game["away_team"], game["home_team"], odds_events)
    if odds:
        if odds["away_ml_odds"] is not None:
            filled["away_ml_odds"] = odds["away_ml_odds"]
        if odds["home_ml_odds"] is not None:
            filled["home_ml_odds"] = odds["home_ml_odds"]
        if odds["away_rl_odds"] is not None:
            filled["away_rl_odds"] = odds["away_rl_odds"]
        if odds["home_rl_odds"] is not None:
            filled["home_rl_odds"] = odds["home_rl_odds"]
        if odds["total_line"] is not None:
            filled["total_line"] = odds["total_line"]
        if odds["over_odds"] is not None:
            filled["over_odds"] = odds["over_odds"]
        if odds["under_odds"] is not None:
            filled["under_odds"] = odds["under_odds"]

    return filled

# ----------------------------
# Session defaults
# ----------------------------
defaults = {
    "away_team": "Yankees",
    "home_team": "Giants",
    "away_sp": "Cam Schlittler",
    "home_sp": "Robbie Ray",
    "away_sp_hand": "R",
    "home_sp_hand": "L",
    "away_sp_fip": 3.06,
    "away_sp_xfip": 4.12,
    "home_sp_fip": 4.19,
    "home_sp_xfip": 4.19,
    "away_bullpen_xfip": 3.85,
    "home_bullpen_xfip": 4.10,
    "away_woba": 0.338,
    "away_wrcp": 118,
    "home_woba": 0.315,
    "home_wrcp": 98,
    "park_factor": 0.92,
    "away_ml_odds": -126,
    "home_ml_odds": 104,
    "away_rl_odds": 135,
    "home_rl_odds": -163,
    "total_line": 7.5,
    "over_odds": -118,
    "under_odds": -102,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ----------------------------
# Auto-fill UI
# ----------------------------
st.subheader("Auto-fill")
c1, c2, c3 = st.columns([1, 2, 1])

with c1:
    game_date = st.date_input("Game Date", value=date.today())

games = load_schedule(str(game_date))
labels = ["Manual entry"] + [g["label"] for g in games]

with c2:
    selected_label = st.selectbox("Choose a game", labels)

with c3:
    st.write("")
    st.write("")
    if st.button("Auto-fill stats + DK odds", use_container_width=True):
        if selected_label != "Manual entry":
            game = next(g for g in games if g["label"] == selected_label)
            filled = auto_fill_for_game(game, game_date)
            for k, v in filled.items():
                st.session_state[k] = v
            st.success("Auto-fill complete.")
            st.rerun()

# ----------------------------
# Sidebar inputs
# ----------------------------
with st.sidebar:
    st.header("Game Inputs")
    st.text_input("Away Team", key="away_team")
    st.text_input("Home Team", key="home_team")

    st.subheader("Pitchers")
    st.text_input("Away SP", key="away_sp")
    st.text_input("Home SP", key="home_sp")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Away SP Hand", key="away_sp_hand")
        st.number_input("Away SP FIP", key="away_sp_fip", step=0.01)
        st.number_input("Away SP xFIP", key="away_sp_xfip", step=0.01)
        st.number_input("Away Bullpen xFIP", key="away_bullpen_xfip", step=0.01)
        st.number_input("Away Team wOBA", key="away_woba", step=0.001, format="%.3f")
        st.number_input("Away Team wRC+", key="away_wrcp", step=1)
    with col2:
        st.text_input("Home SP Hand", key="home_sp_hand")
        st.number_input("Home SP FIP", key="home_sp_fip", step=0.01)
        st.number_input("Home SP xFIP", key="home_sp_xfip", step=0.01)
        st.number_input("Home Bullpen xFIP", key="home_bullpen_xfip", step=0.01)
        st.number_input("Home Team wOBA", key="home_woba", step=0.001, format="%.3f")
        st.number_input("Home Team wRC+", key="home_wrcp", step=1)

    st.subheader("Market")
    st.number_input("Park Factor", key="park_factor", step=0.01)
    st.number_input(f"{st.session_state.away_team} ML Odds", key="away_ml_odds", step=1)
    st.number_input(f"{st.session_state.home_team} ML Odds", key="home_ml_odds", step=1)
    st.number_input(f"{st.session_state.away_team} -1.5 Odds", key="away_rl_odds", step=1)
    st.number_input(f"{st.session_state.home_team} +1.5 Odds", key="home_rl_odds", step=1)
    st.number_input("Total Line", key="total_line", step=0.5)
    st.number_input("Over Odds", key="over_odds", step=1)
    st.number_input("Under Odds", key="under_odds", step=1)

away_team = st.session_state.away_team
home_team = st.session_state.home_team

away_win_prob, home_win_prob = moneyline_win_prob(
    st.session_state.away_sp_fip, st.session_state.home_sp_fip,
    st.session_state.away_sp_xfip, st.session_state.home_sp_xfip,
    st.session_state.away_woba, st.session_state.home_woba,
    st.session_state.away_wrcp, st.session_state.home_wrcp,
    st.session_state.park_factor
)

away_runs, home_runs, projected_total = projected_runs(
    st.session_state.away_woba, st.session_state.home_woba,
    st.session_state.away_wrcp, st.session_state.home_wrcp,
    st.session_state.away_sp_fip, st.session_state.home_sp_fip,
    st.session_state.away_sp_xfip, st.session_state.home_sp_xfip,
    st.session_state.away_bullpen_xfip, st.session_state.home_bullpen_xfip,
    st.session_state.park_factor
)

away_rl_cover, home_plus_cover, away_plus_cover, home_rl_cover = runline_cover_probs(
    away_win_prob, away_runs, home_runs
)
over_prob, under_prob = over_under_probs(projected_total, st.session_state.total_line)

away_ml_fair = prob_to_american(away_win_prob)
home_ml_fair = prob_to_american(home_win_prob)
away_rl_fair = prob_to_american(away_rl_cover)
home_rl_fair = prob_to_american(home_plus_cover)
over_fair = prob_to_american(over_prob)
under_fair = prob_to_american(under_prob)

away_ml_edge = away_win_prob - american_to_implied_prob(st.session_state.away_ml_odds)
home_ml_edge = home_win_prob - american_to_implied_prob(st.session_state.home_ml_odds)
away_rl_edge = away_rl_cover - american_to_implied_prob(st.session_state.away_rl_odds)
home_rl_edge = home_plus_cover - american_to_implied_prob(st.session_state.home_rl_odds)
over_edge = over_prob - american_to_implied_prob(st.session_state.over_odds)
under_edge = under_prob - american_to_implied_prob(st.session_state.under_odds)

markets = [
    {"bet": f"{away_team} ML", "edge": away_ml_edge, "kelly": min(0.5, kelly_fraction(away_win_prob, st.session_state.away_ml_odds) * 0.25)},
    {"bet": f"{home_team} ML", "edge": home_ml_edge, "kelly": min(0.5, kelly_fraction(home_win_prob, st.session_state.home_ml_odds) * 0.25)},
    {"bet": f"{away_team} -1.5", "edge": away_rl_edge, "kelly": min(0.5, kelly_fraction(away_rl_cover, st.session_state.away_rl_odds) * 0.25)},
    {"bet": f"{home_team} +1.5", "edge": home_rl_edge, "kelly": min(0.5, kelly_fraction(home_plus_cover, st.session_state.home_rl_odds) * 0.25)},
    {"bet": "Over", "edge": over_edge, "kelly": min(0.5, kelly_fraction(over_prob, st.session_state.over_odds) * 0.25)},
    {"bet": "Under", "edge": under_edge, "kelly": min(0.5, kelly_fraction(under_prob, st.session_state.under_odds) * 0.25)},
]

best = max(markets, key=lambda x: x["edge"])
best_bet = best["bet"] if best["edge"] >= 0.02 else "No Bet"
best_edge = best["edge"] if best["edge"] >= 0.02 else 0.0
best_kelly = best["kelly"] if best["edge"] >= 0.02 else 0.0
note = "OK" if best_bet != "No Bet" else "No qualifying edge"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projected Score", f"{away_team} {away_runs:.2f} - {home_runs:.2f} {home_team}")
c2.metric(f"{away_team} Win %", f"{away_win_prob:.1%}")
c3.metric(f"{home_team} Win %", f"{home_win_prob:.1%}")
c4.metric("Projected Total", f"{projected_total:.2f}")

st.subheader("Best Bet")
b1, b2, b3 = st.columns(3)
b1.metric("Best Bet", best_bet)
b2.metric("Best Edge %", f"{best_edge:.1%}")
b3.metric("Kelly Units", f"{best_kelly:.2f}")
st.write(f"**Notes:** {note}")

df = pd.DataFrame([
    {"Market": f"{away_team} ML", "Model Prob": away_win_prob, "Book Odds": st.session_state.away_ml_odds, "Fair Odds": away_ml_fair, "Edge %": away_ml_edge},
    {"Market": f"{home_team} ML", "Model Prob": home_win_prob, "Book Odds": st.session_state.home_ml_odds, "Fair Odds": home_ml_fair, "Edge %": home_ml_edge},
    {"Market": f"{away_team} -1.5", "Model Prob": away_rl_cover, "Book Odds": st.session_state.away_rl_odds, "Fair Odds": away_rl_fair, "Edge %": away_rl_edge},
    {"Market": f"{home_team} +1.5", "Model Prob": home_plus_cover, "Book Odds": st.session_state.home_rl_odds, "Fair Odds": home_rl_fair, "Edge %": home_rl_edge},
    {"Market": "Over", "Model Prob": over_prob, "Book Odds": st.session_state.over_odds, "Fair Odds": over_fair, "Edge %": over_edge},
    {"Market": "Under", "Model Prob": under_prob, "Book Odds": st.session_state.under_odds, "Fair Odds": under_fair, "Edge %": under_edge},
])
df["Model Prob"] = df["Model Prob"].map(lambda x: f"{x:.1%}")
df["Edge %"] = df["Edge %"].map(lambda x: f"{x:.1%}")

st.subheader("Market Breakdown")
st.dataframe(df, use_container_width=True)

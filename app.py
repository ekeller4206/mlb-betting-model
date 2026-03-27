import math
import streamlit as st
import pandas as pd

st.set_page_config(page_title="MLB Betting Model", layout="wide")

st.title("MLB Betting Model")
st.caption("Fair odds, edge, Kelly, and best bet for a single MLB game")

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
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)
    if b <= 0:
        return 0.0
    raw = ((b * prob) - (1 - prob)) / b
    return max(0.0, raw)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

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

with st.sidebar:
    st.header("Game Inputs")
    away_team = st.text_input("Away Team", "Yankees")
    home_team = st.text_input("Home Team", "Giants")

    st.subheader("Pitchers")
    away_sp = st.text_input("Away SP", "Cam Schlittler")
    home_sp = st.text_input("Home SP", "Robbie Ray")

    col1, col2 = st.columns(2)
    with col1:
        away_sp_fip = st.number_input("Away SP FIP", value=3.06, step=0.01)
        away_sp_xfip = st.number_input("Away SP xFIP", value=4.12, step=0.01)
        away_bullpen_xfip = st.number_input("Away Bullpen xFIP", value=3.85, step=0.01)
        away_woba = st.number_input("Away Team wOBA", value=0.338, step=0.001, format="%.3f")
        away_wrcp = st.number_input("Away Team wRC+", value=118, step=1)
    with col2:
        home_sp_fip = st.number_input("Home SP FIP", value=4.19, step=0.01)
        home_sp_xfip = st.number_input("Home SP xFIP", value=4.19, step=0.01)
        home_bullpen_xfip = st.number_input("Home Bullpen xFIP", value=4.10, step=0.01)
        home_woba = st.number_input("Home Team wOBA", value=0.315, step=0.001, format="%.3f")
        home_wrcp = st.number_input("Home Team wRC+", value=98, step=1)

    st.subheader("Market")
    park_factor = st.number_input("Park Factor", value=0.92, step=0.01)

    away_ml_odds = st.number_input(f"{away_team} ML Odds", value=-126, step=1)
    home_ml_odds = st.number_input(f"{home_team} ML Odds", value=104, step=1)

    away_rl_odds = st.number_input(f"{away_team} -1.5 Odds", value=135, step=1)
    home_rl_odds = st.number_input(f"{home_team} +1.5 Odds", value=-163, step=1)

    total_line = st.number_input("Total Line", value=7.5, step=0.5)
    over_odds = st.number_input("Over Odds", value=-118, step=1)
    under_odds = st.number_input("Under Odds", value=-102, step=1)

away_win_prob, home_win_prob = moneyline_win_prob(
    away_sp_fip, home_sp_fip,
    away_sp_xfip, home_sp_xfip,
    away_woba, home_woba,
    away_wrcp, home_wrcp,
    park_factor
)

away_runs, home_runs, projected_total = projected_runs(
    away_woba, home_woba,
    away_wrcp, home_wrcp,
    away_sp_fip, home_sp_fip,
    away_sp_xfip, home_sp_xfip,
    away_bullpen_xfip, home_bullpen_xfip,
    park_factor
)

away_rl_cover, home_plus_cover, away_plus_cover, home_rl_cover = runline_cover_probs(
    away_win_prob, away_runs, home_runs
)

over_prob, under_prob = over_under_probs(projected_total, total_line)

away_ml_fair = prob_to_american(away_win_prob)
home_ml_fair = prob_to_american(home_win_prob)
away_rl_fair = prob_to_american(away_rl_cover)
home_rl_fair = prob_to_american(home_plus_cover)
over_fair = prob_to_american(over_prob)
under_fair = prob_to_american(under_prob)

away_ml_edge = away_win_prob - american_to_implied_prob(away_ml_odds)
home_ml_edge = home_win_prob - american_to_implied_prob(home_ml_odds)
away_rl_edge = away_rl_cover - american_to_implied_prob(away_rl_odds)
home_rl_edge = home_plus_cover - american_to_implied_prob(home_rl_odds)
over_edge = over_prob - american_to_implied_prob(over_odds)
under_edge = under_prob - american_to_implied_prob(under_odds)

markets = [
    {"bet": f"{away_team} ML", "edge": away_ml_edge, "kelly": min(0.5, kelly_fraction(away_win_prob, away_ml_odds) * 0.25)},
    {"bet": f"{home_team} ML", "edge": home_ml_edge, "kelly": min(0.5, kelly_fraction(home_win_prob, home_ml_odds) * 0.25)},
    {"bet": f"{away_team} -1.5", "edge": away_rl_edge, "kelly": min(0.5, kelly_fraction(away_rl_cover, away_rl_odds) * 0.25)},
    {"bet": f"{home_team} +1.5", "edge": home_rl_edge, "kelly": min(0.5, kelly_fraction(home_plus_cover, home_rl_odds) * 0.25)},
    {"bet": "Over", "edge": over_edge, "kelly": min(0.5, kelly_fraction(over_prob, over_odds) * 0.25)},
    {"bet": "Under", "edge": under_edge, "kelly": min(0.5, kelly_fraction(under_prob, under_odds) * 0.25)},
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
    {
        "Market": f"{away_team} ML",
        "Model Prob": away_win_prob,
        "Book Odds": away_ml_odds,
        "Fair Odds": away_ml_fair,
        "Edge %": away_ml_edge,
    },
    {
        "Market": f"{home_team} ML",
        "Model Prob": home_win_prob,
        "Book Odds": home_ml_odds,
        "Fair Odds": home_ml_fair,
        "Edge %": home_ml_edge,
    },
    {
        "Market": f"{away_team} -1.5",
        "Model Prob": away_rl_cover,
        "Book Odds": away_rl_odds,
        "Fair Odds": away_rl_fair,
        "Edge %": away_rl_edge,
    },
    {
        "Market": f"{home_team} +1.5",
        "Model Prob": home_plus_cover,
        "Book Odds": home_rl_odds,
        "Fair Odds": home_rl_fair,
        "Edge %": home_rl_edge,
    },
    {
        "Market": "Over",
        "Model Prob": over_prob,
        "Book Odds": over_odds,
        "Fair Odds": over_fair,
        "Edge %": over_edge,
    },
    {
        "Market": "Under",
        "Model Prob": under_prob,
        "Book Odds": under_odds,
        "Fair Odds": under_fair,
        "Edge %": under_edge,
    },
])

df["Model Prob"] = df["Model Prob"].map(lambda x: f"{x:.1%}")
df["Edge %"] = df["Edge %"].map(lambda x: f"{x:.1%}")

st.subheader("Market Breakdown")
st.dataframe(df, use_container_width=True)

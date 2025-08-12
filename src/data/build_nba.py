import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# --------------------------
# Scrape opponent stats
# --------------------------
def scrape_fantasypros_position_stats():
    url = "https://www.fantasypros.com/daily-fantasy/nba/fanduel-defense-vs-position.php"
    tables = pd.read_html(url)
    df = tables[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    df = df.rename(columns={"Team": "opponent_team"})

    team_name_map = {
        "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BRK", "Charlotte": "CHO",
        "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
        "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
        "LA Clippers": "LAC", "LA Lakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
        "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
        "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHO",
        "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
        "Utah": "UTA", "Washington": "WAS"
    }

    df["opponent_abbr"] = df["opponent_team"].map(team_name_map)
    return df

# --------------------------
# Player helpers
# --------------------------
def get_player_id(player_name):
    for p in players.get_active_players():
        if p["full_name"].lower() == player_name.lower():
            return p["id"]
    return None

def get_player_game_data(player_name, season="2023-24"):
    pid = get_player_id(player_name)
    if not pid:
        return pd.DataFrame()

    try:
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season)
        df = gamelog.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["is_home"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
        df["opponent"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
        df["avg_points_last_5"] = df["PTS"].rolling(5).mean().shift(1)
        df["avg_assists_last_5"] = df["AST"].rolling(5).mean().shift(1)
        df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 5)
        df["player_name"] = player_name
        return df
    except:
        return pd.DataFrame()

# --------------------------
# Enrichment: mock pace, teammate ppg, games missed
# --------------------------
def load_mock_team_stats():
    return pd.DataFrame({
        "team": ["LAL", "BOS", "GSW", "MIL"],
        "pace": [100.5, 98.3, 102.7, 96.1],
        "def_rating": [112.3, 107.9, 114.1, 105.5]
    })

def estimate_games_missed(game_dates):
    season_start = pd.to_datetime("2023-10-24")
    played = pd.to_datetime(game_dates).dt.date
    full_schedule = pd.date_range(start=season_start, end=max(game_dates), freq='D')
    missed = len([d for d in full_schedule if d.weekday() < 6 and d.date() not in played.values])
    return missed

def get_top_teammates(player_name, team_abbr, season="2023-24"):
    teammate_stats = []
    all_players = players.get_active_players()

    for p in all_players:
        name = p["full_name"]
        if name == player_name:
            continue
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=p["id"], season=season)
            df = gamelog.get_data_frames()[0]
            avg = df["PTS"].mean()
            last5 = df["PTS"].rolling(5).mean().iloc[-1]
            if df["MATCHUP"].astype(str).str.contains(team_abbr).any():
                teammate_stats.append((name, avg, last5))
        except:
            continue
        time.sleep(0.4)

    top2 = sorted(teammate_stats, key=lambda x: x[1], reverse=True)[:2]
    data = {}
    for i, (name, avg, last5) in enumerate(top2):
        data[f"teammate_{i+1}_name"] = name
        data[f"teammate_{i+1}_ppg"] = round(avg, 2)
        data[f"teammate_{i+1}_last5"] = round(last5, 2)
    return data

def enrich(df):
    if df.empty:
        return df

    team = df["MATCHUP"].iloc[0].split()[0]
    player = df["player_name"].iloc[0]
    team_stats = load_mock_team_stats()
    stats_row = team_stats[team_stats["team"] == team]
    if not stats_row.empty:
        df["team_pace"] = stats_row["pace"].values[0]
        df["team_def_rating"] = stats_row["def_rating"].values[0]
    df["games_missed"] = estimate_games_missed(df["GAME_DATE"])

    teammates = get_top_teammates(player, team)
    for k, v in teammates.items():
        df[k] = v

    return df

# --------------------------
# Main Build
# --------------------------
def build_dataset():
    fantasy_df = scrape_fantasypros_position_stats()
    all_players = players.get_active_players()
    player_dfs = []

    for p in all_players[:20]:  # limit for testing
        name = p["full_name"]
        df = get_player_game_data(name)
        if not df.empty:
            df = df.merge(fantasy_df, left_on="opponent", right_on="opponent_abbr", how="left")
            df = enrich(df)
            player_dfs.append(df)
        time.sleep(0.6)

    final = pd.concat(player_dfs, ignore_index=True)
    final.to_csv("full_nba_player_dataset.csv", index=False)
    print("✅ Saved full_nba_player_dataset.csv")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    build_dataset()

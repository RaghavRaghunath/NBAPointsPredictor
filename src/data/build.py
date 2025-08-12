import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import players, teams
import time
import os

def build_dataset(resume_from=0, output_csv="nba_player_augmented_dataset.csv"):
    import os
    import time

    fantasy_df = scrape_fantasypros_position_stats()
    position_defense = scrape_points_allowed_by_position()
    all_players = players.get_active_players()

    existing_players = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        existing_players = set(existing_df["player_name"].unique())

    for i, p in enumerate(all_players[resume_from:], start=resume_from):
        name = p["full_name"]

        if name in existing_players:
            print(f"⏩ Skipping {name} (already in dataset)")
            continue

        print(f"🔄 Processing ({i + 1}/{len(all_players)}): {name}")
        df = get_player_game_data(name)
        if df.empty:
            continue

        # Example: Assign a default position (replace with real logic if needed)
        df["position"] = "SG"  # You can replace with actual mapping per player

        try:
            # Merge in FantasyPros team matchup stats
            df = df.merge(fantasy_df, left_on="opponent", right_on="opponent_abbr", how="left")

            # Add opponent points allowed to this player's position
            df["opp_points_allowed_to_position"] = df.apply(
                lambda row: position_defense.get(row["opponent"], {}).get(row["position"], None),
                axis=1
            )

            # Add last-3-game trend stat
            df["trend_last_3_pts"] = df["PTS"].diff().rolling(3).mean().shift(1)

            # Add existing enrichment (pace, defense, teammates, etc.)
            df = enrich(df)

        except Exception as e:
            print(f"⚠️ Skipping {name} due to error: {e}")
            continue

        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode="a", index=False, header=False)
        else:
            df.to_csv(output_csv, index=False)

        print(f"✅ Added {name} to {output_csv}")
        time.sleep(0.6)


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
    all_players = players.get_active_players()
    for i, p in enumerate(all_players):
        print(f"🔄 Processing: {player_name}")
        if p["full_name"].lower() == player_name.lower():
            return p["id"]
        i+=1
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
    all_teams = teams.get_teams()
    teammate_stats = []

    # Step 1: Get team ID from abbreviation
    team_id = next((team["id"] for team in all_teams if team["abbreviation"] == team_abbr), None)
    if not team_id:
        print(f"❌ Could not find team ID for {team_abbr}")
        return {}

    # Step 2: Get roster
    try:
        roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        teammate_names = roster_df["PLAYER"].tolist()
    except Exception as e:
        print(f"❌ Failed to get roster for team {team_abbr}: {e}")
        return {}

    # Step 3: Loop over teammates
    active_players = players.get_active_players()
    player_id_map = {p["full_name"]: p["id"] for p in active_players}

    for name in teammate_names:
        if name == player_name or name not in player_id_map:
            continue
        try:
            print(f"   📊 Fetching stats for teammate: {name}")
            log = playergamelog.PlayerGameLog(player_id=player_id_map[name], season=season)
            df = log.get_data_frames()[0]
            if df.empty or "PTS" not in df.columns:
                continue
            avg_ppg = df["PTS"].astype(float).mean()
            last5 = df["PTS"].astype(float).rolling(5).mean().iloc[-1]
            teammate_stats.append((name, avg_ppg, last5))
        except Exception as e:
            print(f"   ⚠️ Skipping {name}: {e}")
            continue
        time.sleep(0.4)

    if not teammate_stats:
        print(f"⚠️ No usable teammates found for {player_name} ({team_abbr})")
        return {}

    # Step 4: Sort and return top 2
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
def build_dataset(resume_from=160):
    fantasy_df = scrape_fantasypros_position_stats()
    all_players = players.get_active_players()
    player_dfs = []

    for i, p in enumerate(all_players[resume_from:], start=resume_from):
        name = p["full_name"]
        print(f"\n🔄 Processing ({i + 1}/{len(all_players)}): {name}")
        df = get_player_game_data(name)
        if df.empty:
            continue

        try:
            df = df.merge(fantasy_df, left_on="opponent", right_on="opponent_abbr", how="left")
            df = enrich(df)
        except Exception as e:
            print(f"⚠️ Skipping {name} due to error: {e}")
            continue

        if os.path.exists("full_nba_player_dataset.csv"):
            df.to_csv("full_nba_player_dataset.csv", mode="a", index=False, header=False)
        else:
            df.to_csv("full_nba_player_dataset.csv", index=False)

        print(f"✅ Added {name} to CSV")
        time.sleep(0.6)

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    build_dataset()

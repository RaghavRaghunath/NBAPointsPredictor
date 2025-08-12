from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time

print("Loading NBA player game data...")
def get_player_id(player_name):
    player_dict = players.find_players_by_full_name(player_name)
    return player_dict[0]['id'] if player_dict else None

def get_player_game_data(player_name, season="2023-24"):
    player_id = get_player_id(player_name)
    if not player_id:
        print(f"❌ Player not found: {player_name}")
        return pd.DataFrame()

    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]

        print(f"\n✅ {player_name} data loaded — Columns:")
        print(df.columns.tolist())

        required = ["GAME_DATE", "MATCHUP", "PTS", "MIN", "FGA", "AST", "REB"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"⚠️ Skipping {player_name}: missing columns: {missing}")
            return pd.DataFrame()

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["is_home"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
        df["opponent"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
        df["avg_points_last_5"] = df["PTS"].rolling(5).mean().shift(1)
        df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 5)
        df["team"] = player_name

        try:
            return df[[
                "GAME_DATE", "team", "opponent", "is_home", "PTS",
                 "avg_points_last_5", "MIN", "FGA", "AST", "REB", "days_rest"
            ]].rename(columns={"PTS": "actual_points"})
        except KeyError as e:
            print(f"❌ KeyError on final return for {player_name}: {e}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error fetching data for {player_name}: {e}")
        return pd.DataFrame()

# Players to collect
players_to_pull = ["LeBron James", "Stephen Curry", "Kevin Durant"]

# Safely build the combined dataset
player_dfs = []
for player in players_to_pull:
    df = get_player_game_data(player)
    if not df.empty:
        player_dfs.append(df)
    time.sleep(0.6)  # Delay to avoid rate limiting

# Merge all valid DataFrames
all_data = pd.concat(player_dfs, ignore_index=True)
all_data.to_csv("nba_player_game_data.csv", index=False)

print("\n✅ Saved to nba_player_game_data.csv")

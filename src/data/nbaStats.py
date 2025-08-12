from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd

def get_player_id(player_name):
    player_dict = players.find_players_by_full_name(player_name)
    return player_dict[0]['id'] if player_dict else None

def get_player_game_data(player_name, season="2023-24"):
    player_id = get_player_id(player_name)
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
    df["opponent"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
    df["avg_points_last_5"] = df["PTS"].rolling(5).mean().shift(1)
    df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 5)

    return df[[
        "GAME_DATE", "TEAM_ABBREVIATION", "opponent", "is_home", "PTS",
        "avg_points_last_5", "MIN", "FGA", "AST", "REB", "days_rest"
    ]].rename(columns={"PTS": "actual_points", "TEAM_ABBREVIATION": "team"})

# Run this for a few players
players_to_pull = ["LeBron James", "Stephen Curry", "Kevin Durant"]
all_data = pd.concat([get_player_game_data(p) for p in players_to_pull], ignore_index=True)

# Save locally
all_data.to_csv("nba_player_game_data.csv", index=False)
print("Saved nba_player_game_data.csv")

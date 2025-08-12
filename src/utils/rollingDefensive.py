import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.library.parameters import SeasonTypeAllStar

# Load your player dataset
df = pd.read_csv("nba_merged_with_defense_final.csv")
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Get list of unique opponent teams and season years
df['SEASON'] = df['SEASON_ID'].astype(str).str[-4:].astype(int)
teams_seasons = df[['opponent', 'SEASON']].drop_duplicates()

# Step 1: Build a game mapping from NBA API
game_id_map = []

print("📥 Fetching official NBA games...")
for _, row in teams_seasons.iterrows():
    team_abbr = row['opponent']
    season = row['SEASON']
    try:
        finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=None,
                                                   season_nullable=f"{season-1}-{str(season)[-2:]}",
                                                   season_type_nullable=SeasonTypeAllStar.regular)
        games = finder.get_data_frames()[0]
        games = games[games['TEAM_ABBREVIATION'] == team_abbr]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        game_id_map.append(games[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION']])
        time.sleep(0.6)
    except Exception as e:
        print(f"Failed to fetch for {team_abbr} {season}: {e}")

# Combine all games
official_games_df = pd.concat(game_id_map)
official_games_df.rename(columns={'TEAM_ABBREVIATION': 'opponent'}, inplace=True)

# Step 2: Merge official game IDs with your dataset
df_merged_ids = pd.merge(df, official_games_df, on=['opponent', 'GAME_DATE'], how='left')

# Step 3: Fetch opponent points using valid Game_IDs
opponent_pts_cache = {}

def get_opponent_points(game_id, team_abbr):
    if pd.isna(game_id) or not str(game_id).startswith("002"):
        return None
    if game_id in opponent_pts_cache:
        return opponent_pts_cache[game_id].get(team_abbr, None)
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        df_box = box.get_data_frames()[0]
        team_pts = df_box.groupby("TEAM_ABBREVIATION")["PTS"].sum().to_dict()
        opponent_abbr = [abbr for abbr in team_pts if abbr != team_abbr]
        if opponent_abbr:
            opponent_pts_cache[game_id] = team_pts
            time.sleep(0.6)
            return team_pts[opponent_abbr[0]]
        return None
    except Exception as e:
        print(f"Failed for {game_id}: {e}")
        return None

# Step 4: Apply opponent points fetching
print("📊 Fetching opponent points...")
df_merged_ids['OPPONENT_PTS'] = df_merged_ids.apply(
    lambda row: get_opponent_points(row['GAME_ID'], row['opponent']), axis=1
)

# Step 5: Compute EMA opponent points
df_merged_ids.sort_values(by=['opponent', 'GAME_DATE'], inplace=True)
df_merged_ids['EMA_OPP_PTS_ALLOWED'] = df_merged_ids.groupby('opponent')['OPPONENT_PTS'].transform(
    lambda x: x.ewm(span=5, adjust=False).mean()
)

# Step 6: Save result
df_merged_ids.to_csv("nba_dataset_with_official_gameids_ema.csv", index=False)
print("✅ Saved to nba_dataset_with_official_gameids_ema.csv")

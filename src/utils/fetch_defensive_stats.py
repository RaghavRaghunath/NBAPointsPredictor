import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2
import time
from pathlib import Path
import json

def fetch_game_points(game_id, team_abbr):
    """Fetch points scored by opponent in a specific game"""
    try:
        # Add NBA prefix if missing
        if not game_id.startswith('002'):
            game_id = f'002{game_id}'
        
        # Fetch box score data
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        df_box = box.get_data_frames()[0]
        
        # Get opponent points
        if team_abbr not in df_box['TEAM_ABBREVIATION'].values:
            print(f"❌ Team {team_abbr} not found in box score for game {game_id}")
            print("Available teams:", df_box['TEAM_ABBREVIATION'].unique())
            return None
            
        opponent_team = df_box[df_box['TEAM_ABBREVIATION'] != team_abbr]
        if len(opponent_team) != 1:
            print(f"❌ Unexpected number of opponent teams for game {game_id}")
            return None
            
        opponent_points = opponent_team['PTS'].values[0]
        return opponent_points
    except Exception as e:
        print(f"❌ Error fetching game {game_id} for {team_abbr}: {e}")
        try:
            # Try to print available data for debugging
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            df_box = box.get_data_frames()[0]
            print("\n=== Available Teams ===")
            print(df_box['TEAM_ABBREVIATION'].unique())
            print("\n=== Available Points ===")
            print(df_box[['TEAM_ABBREVIATION', 'PTS']])
        except:
            pass
        return None

def main():
    # Load input data
    df = pd.read_csv('nba_merged_with_defense_final.csv')
    
    # Create output directory if it doesn't exist
    output_dir = Path('defensive_stats')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize cache
    cache_file = output_dir / 'points_cache.json'
    points_cache = {}
    
    # Load existing cache if it exists
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                points_cache = json.load(f)
            print(f"✅ Loaded cache with {len(points_cache)} games")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
    
    # Process games in batches of 100
    batch_size = 100
    total_games = len(df)
    processed = 0
    
    for start_idx in range(0, total_games, batch_size):
        end_idx = min(start_idx + batch_size, total_games)
        batch = df.iloc[start_idx:end_idx]
        print(f"\n=== Processing batch {start_idx}-{end_idx} of {total_games} ===")
        
        for idx, row in batch.iterrows():
            game_id = str(row['Game_ID'])
            team_abbr = row['opponent']
            
            # Skip if already in cache
            if game_id in points_cache:
                continue
                
            # Fetch points
            points = fetch_game_points(game_id, team_abbr)
            
            if points is not None:
                points_cache[game_id] = points
                print(f"✅ Fetched points for game {game_id} ({team_abbr}): {points}")
            else:
                print(f"❌ Failed to fetch points for game {game_id} ({team_abbr})")
            
            processed += 1
            
            # Save cache after every 10 games
            if processed % 10 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(points_cache, f, indent=2)
                print(f"💾 Saved cache with {len(points_cache)} games")
                
            # Add a small delay to avoid API rate limiting
            time.sleep(0.5)
            
            # Print progress every 10 games
            if processed % 10 == 0:
                print(f"📊 Progress: {processed}/{total_games} games processed")
                print(f"✅ Successfully fetched: {len(points_cache)} games")
                print(f"❌ Failed to fetch: {processed - len(points_cache)} games")
    
    # Save final cache
    with open(cache_file, 'w') as f:
        json.dump(points_cache, f, indent=2)
    print(f"\n=== Final Summary ===")
    print(f"✅ Total games processed: {processed}")
    print(f"✅ Successfully fetched: {len(points_cache)} games")
    print(f"❌ Failed to fetch: {processed - len(points_cache)} games")
    print(f"💾 Final cache saved with {len(points_cache)} games")

if __name__ == "__main__":
    main()

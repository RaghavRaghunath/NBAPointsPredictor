import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('defensive_stats.log'),
        logging.StreamHandler()
    ]
)

def format_game_id(game_id):
    """Ensure game ID has NBA prefix"""
    if not str(game_id).startswith('002'):
        return f'002{game_id}'
    return str(game_id)

def fetch_game_points(game_id, team_abbr):
    """Fetch opponent points for a single game"""
    try:
        # Add NBA prefix
        game_id = format_game_id(game_id)
        
        # Fetch box score
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        df_box = box.get_data_frames()[0]
        
        # Get opponent points
        if team_abbr not in df_box['TEAM_ABBREVIATION'].values:
            logging.error(f"Team {team_abbr} not found in box score for game {game_id}")
            return None
            
        opponent_team = df_box[df_box['TEAM_ABBREVIATION'] != team_abbr]
        if len(opponent_team) != 1:
            logging.error(f"Unexpected number of opponent teams for game {game_id}")
            return None
            
        opponent_points = opponent_team['PTS'].values[0]
        return opponent_points
    except Exception as e:
        logging.error(f"Error fetching game {game_id} for {team_abbr}: {str(e)}")
        return None

def process_game(game_id, team_abbr, cache_file):
    """Process a single game and save result"""
    points = fetch_game_points(game_id, team_abbr)
    if points is not None:
        # Load cache
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except:
            cache = {}
            
        # Update cache
        cache[game_id] = points
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
            
        logging.info(f"✅ Successfully fetched points for game {game_id}")
    else:
        logging.error(f"❌ Failed to fetch points for game {game_id}")
    
    return points is not None

def main():
    # Create output directory
    output_dir = Path('defensive_stats_v2')
    output_dir.mkdir(exist_ok=True)
    
    # Cache file
    cache_file = output_dir / 'points_cache.json'
    
    # Load input data
    df = pd.read_csv('nba_merged_with_defense_final.csv')
    total_games = len(df)
    
    # Load existing cache
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logging.info(f"✅ Loaded cache with {len(cache)} games")
        except:
            logging.warning("⚠️ Could not load cache")
    
    # Get games to process (exclude cached games)
    games_to_process = df[~df['Game_ID'].astype(str).isin(cache.keys())]
    logging.info(f"Total games to process: {len(games_to_process)}")
    
    # Process games in parallel
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor() as executor:
        # Create progress bar
        with tqdm(total=len(games_to_process), desc="Processing games") as pbar:
            # Process games in chunks
            chunk_size = 100
            for i in range(0, len(games_to_process), chunk_size):
                chunk = games_to_process.iloc[i:i + chunk_size]
                
                # Submit tasks
                futures = []
                for _, row in chunk.iterrows():
                    future = executor.submit(
                        process_game,
                        str(row['Game_ID']),
                        row['opponent'],
                        cache_file
                    )
                    futures.append(future)
                
                # Wait for results
                for future in futures:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                    pbar.update(1)
                    
                # Add small delay between chunks to avoid API rate limiting
                time.sleep(1)
    
    # Print summary
    logging.info("\n=== Summary ===")
    logging.info(f"Total games processed: {successful + failed}")
    logging.info(f"✅ Successfully fetched: {successful}")
    logging.info(f"❌ Failed to fetch: {failed}")
    logging.info(f"💾 Final cache saved with {successful} games")

if __name__ == "__main__":
    main()

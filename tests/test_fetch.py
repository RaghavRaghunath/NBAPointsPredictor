import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv2
import json
import time
from pathlib import Path
from datetime import datetime

def format_nba_game_id(game_id):
    """Convert game ID to NBA API format (002YYNNNNN)"""
    game_id = str(game_id).strip()
    
    # If it's already in NBA format, return as is
    if game_id.startswith('002') and len(game_id) == 10:
        return game_id
    
    # If it's a date-based ID (YYMMDDII), convert to NBA format
    if len(game_id) == 8:  # YYMMDDII format
        season_year = int(game_id[:2])
        game_num = game_id[6:]
        return f'002{season_year:02d}000{game_num}'
    
    # If it's just a number, try to use it as the game number
    try:
        game_num = int(game_id)
        # Use current season year as fallback
        current_year = datetime.now().year
        season_year = current_year - 2000  # Convert to 2-digit year
        return f'002{season_year:02d}000{game_num:04d}'
    except:
        return None

def fetch_game_points(raw_game_id, team_abbr):
    """Fetch points scored by opponent in a specific game"""
    try:
        # Format the game ID for the NBA API
        game_id = format_nba_game_id(raw_game_id)
        if not game_id:
            print(f"❌ Could not format game ID: {raw_game_id}")
            return None
            
        print(f"\n=== Processing game {raw_game_id} (as {game_id}) for team {team_abbr} ===")
        
        # Try to get the season from the game ID
        try:
            season = 2000 + int(game_id[3:5])  # Extract YY from 002YY...
            print(f"Game season: {season}-{season+1}")
        except:
            print("⚠️ Could not determine season from game ID")
        
        # Fetch game summary with timeout
        print(f"Fetching game summary for {game_id}...")
        try:
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=10)
        except Exception as e:
            print(f"❌ Error fetching game summary: {e}")
            return None
        
        # Get the line score data with error handling
        try:
            # Try to get line score directly
            line_score = summary.line_score.get_data_frame()
            print("Successfully retrieved line score data")
            
        except Exception as e:
            print(f"❌ Error getting line score: {e}")
            print("Trying to get game summary instead...")
            try:
                game_summary = summary.game_summary.get_data_frame()
                print("Game summary:", game_summary)
                return None  # Can't get line score from here
            except Exception as e2:
                print(f"❌ Could not get game summary: {e2}")
                print("Raw JSON response:", summary.get_json())
                return None
                
            print("Successfully retrieved line score data")
            
        except Exception as e:
            print(f"❌ Error getting line score: {e}")
            try:
                print("Available methods:", dir(summary))
                print("Raw JSON response:", summary.get_json())
            except:
                print("Could not get additional error details")
            return None
        
        # Show full game information
        print("\n=== GAME INFORMATION ===")
        print(f"Game ID: {game_id}")
        
        # Try to get game date and teams
        try:
            game_date = line_score['GAME_DATE_EST'].iloc[0] if 'GAME_DATE_EST' in line_score.columns else 'Unknown date'
            print(f"Game date: {game_date}")
        except:
            pass
            
        # Show detailed team information
        print("\nTEAMS IN THIS GAME:")
        for _, row in line_score.iterrows():
            team_info = []
            for col in ['TEAM_ABBREVIATION', 'TEAM_CITY_NAME', 'TEAM_NICKNAME']:
                if col in line_score.columns:
                    team_info.append(f"{col.split('_')[-1]}: {row[col]}")
            print(" - ", ", ".join(team_info))
        
        # Show points for both teams
        print("\nFINAL SCORE:")
        for _, row in line_score.iterrows():
            print(f"{row['TEAM_ABBREVIATION']}: {row['PTS']} points")
        
        # Check if team is in the game (case-insensitive)
        team_abbr_upper = team_abbr.strip().upper()
        available_teams = line_score['TEAM_ABBREVIATION'].str.strip().str.upper().tolist()
        
        if team_abbr_upper not in available_teams:
            print(f"\n❌ Team {team_abbr} not found in game {game_id}")
            print("Available teams:", ", ".join(line_score['TEAM_ABBREVIATION'].tolist()))
            print("\nThis means the team you're looking for didn't play in this game.")
            print("Possible reasons:")
            print("1. The game ID in your data might be incorrect")
            print("2. The team abbreviation might be incorrect")
            print("3. The game might have been rescheduled or the teams changed")
            print("\nPlease check your data source for game ID {game_id} and team {team_abbr}")
            return None
        
        # Get opponent points (case-insensitive match)
        try:
            # Convert team abbreviations to uppercase for case-insensitive comparison
            line_score['TEAM_ABBREVIATION'] = line_score['TEAM_ABBREVIATION'].str.strip().str.upper()
            team_abbr_upper = team_abbr.strip().upper()
            
            # Check if our team is in the game
            if team_abbr_upper not in line_score['TEAM_ABBREVIATION'].values:
                print(f"❌ Team {team_abbr} not found in game {game_id}")
                print("Available teams:", line_score['TEAM_ABBREVIATION'].tolist())
                return None
            
            # Get the opponent's points
            opponent_team = line_score[line_score['TEAM_ABBREVIATION'] != team_abbr_upper]
            
            if len(opponent_team) != 1:
                print(f"❌ Could not find single opponent for {team_abbr} in game {game_id}")
                print("All teams in game:", line_score['TEAM_ABBREVIATION'].tolist())
                return None
            
            opponent_points = int(opponent_team['PTS'].values[0])
            opponent_team_abbr = opponent_team['TEAM_ABBREVIATION'].values[0]
            
            print(f"✅ {team_abbr} vs {opponent_team_abbr}: {opponent_points} points allowed by {team_abbr}")
            return opponent_points
            
        except Exception as e:
            print(f"❌ Error processing team data: {e}")
            print("Line score data:", line_score)
            return None
        
    except Exception as e:
        import traceback
        print(f"❌ Unexpected error in fetch_game_points: {e}")
        print("Stack trace:", traceback.format_exc())
        return None

def main():
    # Load just the first 10 games for testing
    input_file = 'nba_merged_with_defense_final.csv'
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file).head(10)
    
    print("\n=== Sample of data being processed ===")
    print(df[['Game_ID', 'opponent']].head())
    print("\n")
    
    # Create output directory
    output_dir = Path('defensive_stats_test')
    output_dir.mkdir(exist_ok=True)
    cache_file = output_dir / 'test_cache.json'
    
    # Load existing cache
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"✅ Loaded cache with {len(cache)} games")
    except FileNotFoundError:
        cache = {}
        print("Starting with empty cache (no existing cache file found)")
    except json.JSONDecodeError:
        cache = {}
        print("⚠️ Cache file exists but is not valid JSON, starting with empty cache")
    except Exception as e:
        cache = {}
        print(f"⚠️ Error loading cache: {e}, starting with empty cache")
    
    # Process each game
    total_games = len(df)
    print(f"\nProcessing {total_games} games...\n")
    
    for idx, row in df.iterrows():
        game_id = str(row['Game_ID']).strip()
        team_abbr = row['opponent'].strip()
        
        print(f"\n{'='*50}")
        print(f"Processing game {idx+1}/{total_games}: ID={game_id}, Team={team_abbr}")
        
        # Skip if already in cache
        if game_id in cache:
            print(f"ℹ️  Game {game_id} already in cache with value: {cache[game_id]}")
            continue
            
        # Fetch points
        print(f"🔍 Fetching points for game {game_id}...")
        points = fetch_game_points(game_id, team_abbr)
        
        # Save to cache if successful
        if points is not None:
            cache[game_id] = points
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
                print(f"💾 Saved to cache: {game_id} = {points}")
            except Exception as e:
                print(f"⚠️ Error saving to cache: {e}")
        else:
            print(f"❌ Failed to fetch points for game {game_id}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Print final results
    print("\n" + "="*50)
    print("=== Final Results ===")
    print(f"Total games processed: {total_games}")
    print(f"Successfully fetched: {len(cache)} games")
    print("\nCache contents:")
    print(json.dumps(cache, indent=2))

if __name__ == "__main__":
    main()

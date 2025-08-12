import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, boxscoresummaryv2
from nba_api.stats.static import teams
import time
from tqdm import tqdm
import traceback
import json
from pathlib import Path

def get_team_abbreviation(team_name):
    """Convert team name to abbreviation"""
    # Handle None, NaN, or non-string values
    if pd.isna(team_name) or not isinstance(team_name, str):
        print(f"⚠️ Warning: Invalid team name: {team_name}")
        return "UNK"  # Return a default value for unknown teams
    
    # Standardize common variations
    team_name = team_name.strip()
    
    # Map of team names to abbreviations
    nba_teams = teams.get_teams()
    team_dict = {team['full_name']: team['abbreviation'] for team in nba_teams}
    
    # Add common variations and handle special cases
    team_dict.update({
        'LA Clippers': 'LAC',
        'LA Lakers': 'LAL',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'New Orleans Pelicans': 'NOP',
        'New Orleans/Oklahoma City Hornets': 'NOP',
        'New York Knicks': 'NYK',
        'Golden State Warriors': 'GSW',
        'Portland Trail Blazers': 'POR',
        'Portland Trailblazers': 'POR',
        'Philadelphia 76ers': 'PHI',
        'Philadelphia Seventy Sixers': 'PHI',
        'Phoenix Suns': 'PHX',
        'San Antonio Spurs': 'SAS',
        'Oklahoma City Thunder': 'OKC',
        'Oklahoma City': 'OKC',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS',
        'Washington Bullets': 'WAS',
        'New Jersey Nets': 'BKN',
        'Brooklyn Nets': 'BKN',
        'New Jersey': 'BKN',
        'Charlotte Bobcats': 'CHA',
        'Charlotte Hornets': 'CHA',
        'New Orleans Hornets': 'NOP',
        'New Orleans/Oklahoma City': 'NOP',
        'Seattle SuperSonics': 'OKC',
        'Seattle': 'OKC',
        'Vancouver Grizzlies': 'MEM',
        'Vancouver': 'MEM',
        'New Orleans/Oklahoma City Hornets': 'NOP',
        'New Orleans/Oklahoma City': 'NOP'
    })
    
    # Try exact match first
    if team_name in team_dict:
        return team_dict[team_name]
    
    # Try case-insensitive match
    for full_name, abbr in team_dict.items():
        if full_name.lower() == team_name.lower():
            return abbr
    
    # Try to extract from the last word if all else fails
    try:
        return team_name.split()[-1].upper()
    except (AttributeError, IndexError):
        print(f"⚠️ Could not determine abbreviation for: {team_name}")
        return "UNK"

def get_season_games(season='2023-24'):
    """Get all games for a season using league game finder"""
    cache_file = Path(f'nba_games_{season}.json')
    
    # Try to load from cache first
    if cache_file.exists():
        print(f"Loading games from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return pd.DataFrame(json.load(f))
    
    print(f"Fetching games for {season} season...")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_type_nullable='Regular Season')
    games = gamefinder.get_data_frames()[0]
    
    # Process the data
    games = games[games.SEASON_ID.str.endswith(season.split('-')[1])]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    
    # Save to cache
    games.to_json(cache_file, orient='records')
    return games

def get_game_scores(games_df):
    """Get scores for all games"""
    scores = {}
    
    for _, game in tqdm(games_df.iterrows(), total=len(games_df), desc="Processing games"):
        game_id = '00' + str(game['GAME_ID'])
        team_abbr = get_team_abbreviation(game['TEAM_NAME'])
        opponent_abbr = get_team_abbreviation(game['MATCHUP'].split()[-1])
        
        if game_id not in scores:
            scores[game_id] = {}
        
        scores[game_id][team_abbr] = game['PTS']
        scores[game_id][opponent_abbr] = game['PTS']
    
    return scores

def main():
    print("=== Starting NBA Defensive Stats Calculation ===")
    start_time = time.time()
    
    # Load your dataset
    print("\nLoading dataset...")
    df_players = pd.read_csv("nba_merged_with_defense_final.csv")
    print(f"✅ Loaded dataset with {len(df_players)} rows")
    
    # Get all games for the season
    season_games = get_season_games('2023-24')
    game_scores = get_game_scores(season_games)
    
    # Add opponent points to each row
    print("\nAdding opponent points to dataset...")
    
    def get_opponent_points(row):
        try:
            game_id = str(row['Game_ID']).zfill(11)  # Ensure proper game ID format
            team_abbr = get_team_abbreviation(row['Team'])
            
            if team_abbr == "UNK":
                print(f"⚠️ Unknown team: {row['Team']} in game {game_id}")
                return None
                
            if game_id not in game_scores:
                print(f"⚠️ No scores found for game {game_id}")
                return None
            
            # Find opponent's points
            for team, points in game_scores[game_id].items():
                if team != team_abbr:
                    return points
            return None
                    
        except Exception as e:
            print(f"⚠️ Error processing row: {row}\nError: {str(e)}")
            return None
    
    # Add opponent points column
    tqdm.pandas(desc="Calculating opponent points")
    df_players['OPPONENT_POINTS'] = df_players.progress_apply(get_opponent_points, axis=1)
    
    # Calculate EMA for defensive stats
    print("\nCalculating EMA for defensive stats...")
    print("\nAvailable columns in the dataset:")
    print(df_players.columns.tolist())
    
    # Try to identify the correct column names
    player_col = next((col for col in ['PLAYER_ID', 'PLAYER', 'PLAYER_NAME', 'NAME'] if col in df_players.columns), None)
    date_col = next((col for col in ['GAME_DATE', 'DATE', 'GAME_DATE_EST'] if col in df_players.columns), None)
    
    if not player_col or not date_col:
        raise ValueError(f"Could not find required columns. Player column: {player_col}, Date column: {date_col}")
        
    print(f"\nUsing columns - Player: '{player_col}', Date: '{date_col}'")
    
    # Sort by player and date
    df_players.sort_values([player_col, date_col], inplace=True)
    
    # Calculate EMAs - only use columns that exist in the dataframe
    all_columns = df_players.columns.tolist()
    possible_ema_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'MIN', 'OPPONENT_POINTS']
    ema_columns = [col for col in possible_ema_columns if col in all_columns]
    
    print(f"\nCalculating EMAs for columns: {ema_columns}")
    
    if not ema_columns:
        raise ValueError("No valid columns found for EMA calculation")
        
    ema_spans = [5, 10, 20]  # EMA spans for different time windows
    
    for span in ema_spans:
        for col in ema_columns:
            df_players[f'{col}_EMA_{span}'] = df_players.groupby(player_col)[col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
    
    # Save the results
    output_file = 'nba_players_with_defense.csv'
    df_players.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print(traceback.format_exc())

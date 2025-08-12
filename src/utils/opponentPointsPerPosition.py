import pandas as pd
import numpy as np
from nba_api.stats.endpoints import boxscoreplayertrackv3, leaguegamefinder, playergamelogs
from nba_api.stats.static import teams
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import requests
import time
import random


# Configuration
MINUTES_THRESHOLD = 12  # Minimum minutes played to be included
EMA_SPAN = 10  # Number of games for EMA calculation

# Position mapping to standard groups
POSITION_MAPPING = {
    'G': 'G',
    'G-F': 'G',
    'F-G': 'F',
    'F': 'F',
    'F-C': 'F',
    'C-F': 'C',
    'C': 'C',
    'C-C': 'C',
    'PF': 'F',
    'SF': 'F',
    'PF-C': 'F',
    'C-PF': 'C',
    'SF-SG': 'F',
    'SG-SF': 'G',
    'PG': 'G',
    'SG': 'G',
    'PG-SG': 'G',
    'SG-PG': 'G'
}

# Maps each player to either G, F, or C.
def get_primary_position(position):
    if pd.isna(position):
        return 'G'
    return POSITION_MAPPING.get(str(position).strip().upper(), 'G')

# Gets all of the games for the season specified in the parameter.
def fetch_season_games(season='2023-24'):
    try:
        print(f"\nFetching games for {season} season...")
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season',
            timeout=30
        )
        # Set the games datafram equal to variable games.
        games = gamefinder.get_data_frames()[0]
        
        if games.empty:
            print("No games found for the specified season.")
            return pd.DataFrame()
        
        # Clean the data so that the dates are readable, the game_id and team_id are expressed as strings.
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        games = games.sort_values('GAME_DATE')
        games['GAME_ID'] = games['GAME_ID'].astype(str)
        games['TEAM_ID'] = games['TEAM_ID'].astype(str)
        
        print(f"Successfully fetched {len(games)} game records")
        return games
        
    except Exception as e:
        print(f"Error fetching games: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# This method processes each game by taking in variables team, opponent, game_id, game_date, player_logs_df, position_stats, and retries
def process_team_game(team, opponent, game_id, game_date, player_logs_df, position_stats, max_retries=3):
    team_abbr = team['TEAM_ABBREVIATION']
    team_id = str(team['TEAM_ID'])
    
    try:
        # DEBUG Statement.
        print(f"  Processing {team_abbr}...")
        
        max_retries = 3
        player_stats = None
        for attempt in range(max_retries):
            try:
                # Tries to get the game from the box score api endpoint, and then get the player stats from that game in the max retries specified.
                print(f"    Fetching box score (attempt {attempt + 1}/{max_retries})...")
                box_score = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id, timeout=30)
                player_stats = box_score.player_stats.get_data_frame()
                
                # If the player_stats were successfully obtained, then break out of the try, except block. Otherwise, return position_pts.
                if player_stats is not None and not player_stats.empty:
                    print(f"    Successfully retrieved box score with {len(player_stats)} players")
                    break

                return position_pts
            # This is the error indication for if it reached max requests.
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    print(f"    Failed after {max_retries + 1} attempts: {str(e)}")
                    return None

                wait_time = (2 ** attempt) + (random.random() * 0.5)
                print(f"    Request failed, retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            # Error message if it didn't process the team.
            except Exception as e:
                print(f"    Error processing team {team_abbr}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        if player_stats is None or player_stats.empty:
            print(f"    No player stats found for {team_abbr}")
            return None
            
        # Debug: Print available columns
        print("    Available columns:", player_stats.columns.tolist())
        
        # Inconsistencies between the api column names and the desired column names rewuires us to provide this mapping.
        column_mapping = {
            'teamId': 'TEAM_ID',
            'personId': 'PLAYER_ID',
            'position': 'POSITION',
            'min': 'MINUTES',
            'minSeconds': 'MINUTES',
            'mins': 'MINUTES',
            'minutes': 'MINUTES',
            'timeOnCourt': 'MINUTES',
            'timeOnCourtMinutes': 'MINUTES'
        }
        
        # This takes the above column mappings and applies them to the player_stats dataframe.
        player_stats = player_stats.rename(columns={k: v for k, v in column_mapping.items() 
                                                  if k in player_stats.columns})
        
        # If the specific column we are looking for "MINUTES" doesn't exist, look for a possible time or minutes column.
        if 'MINUTES' not in player_stats.columns:
            for col in player_stats.columns:
                if 'min' in str(col).lower() or 'time' in str(col).lower():
                    player_stats = player_stats.rename(columns={col: 'MINUTES'})
                    print(f"    Using column '{col}' as minutes played")
                    break
        
        # If "MINUTES" column still isn't there, then skip the team.
        if 'MINUTES' not in player_stats.columns:
            print(f"    Could not find minutes column for {team_abbr}")
            print(f"    Available columns: {player_stats.columns.tolist()}")
            return None
        
        # Same thing we have been doing, making team id and player id readable strings.
        player_stats = player_stats.astype({
            'TEAM_ID': str,
            'PLAYER_ID': str
        })
        
        # We are filtering player_stats to look specifically for the team id that we are passing in the parameter, and creating a copy of that, storing it in new variable.
        team_players = player_stats[player_stats['TEAM_ID'] == team_id].copy()
        if team_players.empty:
            print(f"    No players found for {team_abbr} (ID: {team_id})")
            return None
            
        # Process minutes played - handle different time formats
        try:
            # If minutes are in 'MM:SS' format
            if team_players['MINUTES'].astype(str).str.contains(':', na=False).any():
                print("    Detected MM:SS format for minutes")
                team_players['minutes_played'] = pd.to_numeric(
                    team_players['MINUTES'].astype(str).str.split(':').str[0],
                    errors='coerce'
                ) + pd.to_numeric(
                    team_players['MINUTES'].astype(str).str.split(':').str[1],
                    errors='coerce'
                ) / 60.0
            else:
                # If minutes are in seconds, convert to minutes
                print("    Detected seconds format for minutes")
                team_players['minutes_played'] = pd.to_numeric(
                    team_players['MINUTES'],
                    errors='coerce'
                ) / 60.0
            
            # Handle any remaining NaN values
            team_players['minutes_played'] = team_players['minutes_played'].fillna(0)
            
            print(f"    Minutes sample: {team_players['minutes_played'].head().tolist()}")
            
        except Exception as e:
            print(f"    Error processing minutes: {str(e)}")
            print(f"    Sample MINUTES values: {team_players['MINUTES'].head().tolist()}")
            return None
        
        # Filter by minutes threshold
        team_players = team_players[team_players['minutes_played'] >= MINUTES_THRESHOLD].copy()
        
        if team_players.empty:
            print(f"    No players met the {MINUTES_THRESHOLD} minute threshold")
            return None
            
        print(f"    Found {len(team_players)} players with ≥{MINUTES_THRESHOLD} min")
        
        # Save a copy of the player logs from that game to a new variable
        game_player_logs = player_logs_df[
            (player_logs_df['GAME_ID'] == str(game_id)) & 
            (player_logs_df['TEAM_ID'] == team_id)
        ].copy()
        
        if game_player_logs.empty:
            print(f"    No game logs found for {team_abbr} in game {game_id}")
            return None
            
        # Like always, make sure that everything has a type of string.
        game_player_logs['PLAYER_ID'] = game_player_logs['PLAYER_ID'].astype(str)
        
        # Merge with player stats to get points
        merged_stats = pd.merge(
            team_players,
            game_player_logs[['PLAYER_ID', 'PTS']],
            on='PLAYER_ID',
            how='left'
        )
        
        # Map positions to standard groups
        merged_stats['POSITION_GROUP'] = merged_stats['POSITION'].apply(
            lambda x: get_primary_position(x) if pd.notnull(x) else 'G'
        )
        
        # Group by position and sum points
        position_pts = merged_stats.groupby('POSITION_GROUP')['PTS'].sum().reset_index()
        
        # Add game and team info
        position_pts['GAME_ID'] = game_id
        position_pts['GAME_DATE'] = game_date
        position_pts['TEAM_ID'] = team_id
        position_pts['TEAM_ABBREVIATION'] = team_abbr
        position_pts['OPPONENT_TEAM_ID'] = str(opponent['TEAM_ID'])
        position_pts['OPPONENT_ABBREVIATION'] = opponent['TEAM_ABBREVIATION']
        
        # Print and store results
        print("    Points by position:")
        for _, row in position_pts.iterrows():
            print(f"      {row['POSITION_GROUP']}: {row['PTS']:.1f}")
            
        return position_pts
        
    except requests.exceptions.RequestException as e:
        print(f"    HTTP Error: {str(e)}")
        raise  # Re-raise to handle in the main loop
    except Exception as e:
        print(f"    Error processing {team_abbr}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Take in parameters of game logs, player logs, the start game, the amount to process, and retries, and return a dataframe of positional defensive stats.
def calculate_position_stats(game_logs, player_logs_df, start_game=0, batch_size=20, max_retries=3):
    all_position_stats = []
    
    # Ensure consistent data types
    game_logs = game_logs.copy()
    game_logs['GAME_ID'] = game_logs['GAME_ID'].astype(str)
    game_logs['TEAM_ID'] = game_logs['TEAM_ID'].astype(str)
    
    # Use the drop duplicates function to ensure that there are no similar game ids or matchups of any sort.
    unique_games = game_logs[['GAME_ID', 'GAME_DATE', 'MATCHUP']].drop_duplicates()
    
    # Calculate the end index for a current batch of games, and use the min() function to ensure that we don't go past the end of the games list.
    # Hey is to process the data in batches because of how much data we have
    end_game = min(start_game + batch_size, len(unique_games))
    games_to_process = unique_games.iloc[start_game:end_game]
    
    print(f"\nProcessing games {start_game} to {end_game-1} of {len(unique_games)-1}")
    print(f"Processing {len(games_to_process)} games in this batch")
    
    # Define output file path
    output_file = 'nba_defense_by_position_latest.csv'
    
    # Initialize or clear the output file at the start
    if os.path.exists(output_file):
        os.remove(output_file)

    
    for game_idx, (_, game) in enumerate(games_to_process.iterrows(), 1):
        game_id = game['GAME_ID']
        game_date = game['GAME_DATE']
        
        try:
            print(f"\n=== Game {start_game + game_idx - 1}: {game['MATCHUP']} ({game_date}) ===")
            
            # Get teams for this game
            game_teams = game_logs[game_logs['GAME_ID'] == game_id]
            if len(game_teams) < 2:
                print(f"  Not enough teams for game {game_id}")
                continue
                
            # Process each team's game
            game_position_stats = []
            for i in range(min(2, len(game_teams))):  # Only process first 2 teams if more than 2
                team = game_teams.iloc[i]
                opponent = game_teams.iloc[1 - i]  # Get the other team
                
                # Process team game with retries
                team_stats = process_team_game(
                    team, 
                    opponent, 
                    game_id, 
                    game_date, 
                    player_logs_df, 
                    game_position_stats,
                    max_retries=max_retries
                )
                
                if team_stats is not None:
                    game_position_stats.append(team_stats)
            
            # If we have valid stats for this game
            if game_position_stats:
                # Combine stats for this game
                game_df = pd.concat(game_position_stats, ignore_index=True)
                
                # Add to our main list
                all_position_stats.extend(game_position_stats)
                
                # Determine if we need to write header (only for first game or new file)
                header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
                
                # Append to the CSV file
                game_df.to_csv(output_file, mode='a', header=header, index=False)
                print(f"  Saved results for game {game_id} to {output_file} ({len(game_df)} records)")
            
            # Add a small delay between games to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing game {game_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save partial results if we have any
            if all_position_stats:
                try:
                    partial_df = pd.concat(all_position_stats, ignore_index=True)
                    error_file = f'error_results_game_{game_id}.csv'
                    partial_df.to_csv(error_file, index=False)
                    print(f"Saved error recovery results to {error_file}")
                except Exception as save_error:
                    print(f"Failed to save error recovery file: {str(save_error)}")
            
            # Continue to next game instead of breaking
            continue
    
    if not all_position_stats:
        print("No position stats were calculated in this batch.")
        return pd.DataFrame()
    
    # Combine all position stats for return
    result_df = pd.concat(all_position_stats, ignore_index=True)
    
    # Calculate rolling stats if we have data
    if not result_df.empty:
        print("\nCalculating rolling statistics...")
        try:
            result_df = calculate_rolling_stats(result_df)
            # Ensure we have all expected columns
            for pos in ['C', 'F', 'G']:
                if f'ROLLING_{pos}_AVG' not in result_df.columns:
                    result_df[f'ROLLING_{pos}_AVG'] = np.nan
                    result_df[f'{pos}_DIFF'] = np.nan
        except Exception as e:
            print(f"Error in rolling stats calculation: {e}")
            import traceback
            traceback.print_exc()
    
    # Save final combined results
    if not result_df.empty:
        final_output_file = f'nba_defense_by_position_{start_game}_to_{end_game-1}.csv'
        result_df.to_csv(final_output_file, index=False)
        print(f"\nFinal results saved to {final_output_file} ({len(result_df)} records)")
    
    return result_df



def calculate_rolling_stats(position_stats):
    """Calculate rolling average points for each team and include all position groups in each row."""
    if position_stats.empty:
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_cols = ['TEAM_ID', 'POSITION_GROUP', 'GAME_DATE', 'PTS', 'OPPONENT_TEAM_ID', 'GAME_ID']
    if not all(col in position_stats.columns for col in required_cols):
        print("Missing required columns for rolling stats calculation")
        return position_stats
    
    try:
        # Sort by team and date
        position_stats = position_stats.sort_values(['TEAM_ID', 'GAME_DATE'])
        
        # Pivot the data to get position groups as columns
        pivot_df = position_stats.pivot_table(
            index=['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'OPPONENT_TEAM_ID'],
            columns='POSITION_GROUP',
            values='PTS',
            aggfunc='first'  # Should be one row per team-game-position
        ).reset_index()
        
        # Rename columns for clarity
        pivot_df.columns = [f'{col}_PTS' if col in ['C', 'F', 'G'] else col for col in pivot_df.columns]
        
        # Calculate rolling averages for each position
        for pos in ['C', 'F', 'G']:
            col_name = f'{pos}_PTS'
            if col_name in pivot_df.columns:
                # Group by team and calculate EMA for each position
                pivot_df[f'ROLLING_{pos}_AVG'] = pivot_df.groupby('TEAM_ID')[col_name]\
                    .transform(lambda x: x.ewm(span=EMA_SPAN, min_periods=1).mean())
                
                # Calculate difference from rolling average
                pivot_df[f'{pos}_DIFF'] = pivot_df[col_name] - pivot_df[f'ROLLING_{pos}_AVG']
        
        # Merge back with original data to get all columns
        result = pd.merge(
            position_stats,
            pivot_df,
            on=['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'OPPONENT_TEAM_ID'],
            how='left'
        )
        
        # Add games played counter
        result['GAMES_PLAYED'] = result.groupby(['TEAM_ID', 'POSITION_GROUP']).cumcount() + 1
        
        # Rename the original rolling average columns for backward compatibility
        if 'ROLLING_AVG_PTS' in result.columns:
            result = result.rename(columns={
                'ROLLING_AVG_PTS': 'ROLLING_AVG_PTS_OLD',
                'PTS_DIFF': 'PTS_DIFF_OLD'
            })
        
        # Create the main rolling average column based on position group
        result['ROLLING_AVG_PTS'] = result.apply(
            lambda row: row[f'ROLLING_{row["POSITION_GROUP"]}_AVG'], 
            axis=1
        )
        result['PTS_DIFF'] = result['PTS'] - result['ROLLING_AVG_PTS']
        
        return result
        
    except Exception as e:
        print(f"Error calculating rolling stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return position_stats  # Return original if error occurs
        return position_stats

def process_game_batch(games, player_logs_df, start_idx, batch_size):
    """Process a batch of games and return position stats."""
    print(f"\nProcessing batch: games {start_idx} to {start_idx + batch_size - 1}")
    
    # Get the subset of games to process
    unique_games = games[['GAME_ID', 'GAME_DATE', 'MATCHUP']].drop_duplicates()
    if start_idx >= len(unique_games):
        print("Start index exceeds number of available games")
        return pd.DataFrame()
        
    end_idx = min(start_idx + batch_size, len(unique_games))
    game_batch = unique_games.iloc[start_idx:end_idx]
    
    # Process the batch
    position_stats = []
    for _, game in game_batch.iterrows():
        game_id = game['GAME_ID']
        game_teams = games[games['GAME_ID'] == game_id]
        
        if len(game_teams) < 2:
            continue
            
        team1, team2 = game_teams.iloc[0], game_teams.iloc[1]
        
        # Process each team's game
        for team, opponent in [(team1, team2), (team2, team1)]:
            team_stats = process_team_game(team, opponent, game_id, game['GAME_DATE'], player_logs_df, position_stats)
            if team_stats is not None:
                position_stats.append(team_stats)
    
    return pd.concat(position_stats, ignore_index=True) if position_stats else pd.DataFrame()

def main(start_game=0, batch_size=20):
    try:
        # Set up argument parsing
        import argparse
        parser = argparse.ArgumentParser(description='Calculate NBA defensive stats by position.')
        parser.add_argument('--start', type=int, default=0, help='Starting game index (0-based)')
        parser.add_argument('--batch-size', type=int, default=20, help='Number of games to process in this batch')
        args = parser.parse_args()
        
        print(f"Starting NBA Defensive Stats Calculation")
        print(f"Processing games {args.start} to {args.start + args.batch_size - 1}")
        
        # Fetch all games for the season
        print("\nFetching season games...")
        try:
            games = fetch_season_games(season='2023-24')
            
            if games.empty:
                print("No games found. Exiting.")
                return
                
            # Ensure we have the required columns
            required_cols = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP']
            if not all(col in games.columns for col in required_cols):
                print("Missing required columns in games data")
                print("Available columns:", games.columns.tolist())
                return
                
        except Exception as e:
            print(f"Error fetching season games: {str(e)}")
            import traceback
            traceback.print_exc()
            return
            
        # Fetch player game logs
        print("\nFetching player game logs...")
        try:
            player_logs = playergamelogs.PlayerGameLogs(
                season_nullable='2023-24',
                season_type_nullable='Regular Season',
                timeout=120
            ).get_data_frames()[0]
            
            if player_logs.empty:
                print("No player game logs found")
                return
                
            # Ensure consistent data types
            for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']:
                if col in player_logs.columns:
                    player_logs[col] = player_logs[col].astype(str)
            
            print(f"Fetched {len(player_logs)} player game logs")
            
        except Exception as e:
            print(f"Error fetching player logs: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Process games
        print("\nProcessing games...")
        try:
            position_stats = calculate_position_stats(
                games, 
                player_logs, 
                start_game=args.start, 
                batch_size=args.batch_size,
                max_retries=3
            )
            
            if position_stats.empty:
                print("\nNo position stats were calculated.")
                return
                
            # Save results with batch information
            output_file = f'nba_defense_by_position_{args.start}_to_{args.start + len(position_stats) - 1}.csv'
            position_stats.to_csv(output_file, index=False)
            
            print(f"\nResults saved to {output_file} ({len(position_stats)} records)")
            print("\nSample of the data:")
            print(position_stats.head())
            
            print("\nSummary statistics by position:")
            print(position_stats.groupby('POSITION_GROUP')['PTS'].agg(['count', 'mean', 'min', 'max']))
            print(f"\nSuccessfully processed and saved {len(position_stats)} game records.")
            
        except Exception as e:
            print(f"\nError processing games: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save partial results if available
            if 'position_stats' in locals() and not position_stats.empty:
                output_file = f'nba_defense_by_position_error_{args.start}.csv'
                position_stats.to_csv(output_file, index=False)
                print(f"\nPartial results saved to {output_file}")
            return
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        if 'position_stats' in locals() and not position_stats.empty:
            output_file = f'nba_defense_by_position_partial_{args.start}.csv'
            position_stats.to_csv(output_file, index=False)
            print(f"\nPartial results saved to {output_file}")
    
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save partial results if available
        if 'position_stats' in locals() and not position_stats.empty:
            output_file = f'nba_defense_by_position_error_{args.start}.csv'
            position_stats.to_csv(output_file, index=False)
            print(f"\nPartial results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Process NBA defensive stats.')
    parser.add_argument('--start', type=int, default=0, help='Starting game index (0-based)')
    parser.add_argument('--batch', type=int, default=20, help='Number of games to process in this batch')
    
    args = parser.parse_args()
    
    main(start_game=args.start, batch_size=args.batch)

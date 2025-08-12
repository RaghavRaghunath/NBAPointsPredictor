import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, teamgamelog, commonallplayers
from nba_api.stats.static import players, teams
import time
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime

def get_all_players(active_only=True):
    """Get all NBA players"""
    nba_players = players.get_players()
    if active_only:
        return [p for p in nba_players if p['is_active']]
    return nba_players

def get_team_abbreviation(team_name):
    """Convert team name to abbreviation"""
    if pd.isna(team_name) or not isinstance(team_name, str):
        return "UNK"
    
    nba_teams = teams.get_teams()
    team_map = {team['full_name']: team['abbreviation'] for team in nba_teams}
    team_map.update({
        'LA Clippers': 'LAC',
        'LA Lakers': 'LAL',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Golden State Warriors': 'GSW',
        'Portland Trail Blazers': 'POR',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHX',
        'San Antonio Spurs': 'SAS',
        'Oklahoma City Thunder': 'OKC',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS',
        'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA'
    })
    
    # Try exact match first
    if team_name in team_map:
        return team_map[team_name]
    
    # Try case-insensitive match
    for full_name, abbr in team_map.items():
        if full_name.lower() == team_name.lower():
            return abbr
    
    # Try to extract from the last word if all else fails
    try:
        return team_name.split()[-1].upper()
    except (AttributeError, IndexError):
        return "UNK"

def get_team_defensive_stats(season='2023-24'):
    """Get team defensive stats for the season"""
    cache_file = f'team_defense_{season.replace("-", "")}.csv'
    
    if os.path.exists(cache_file):
        print(f"Loading team defense data from cache: {cache_file}")
        df = pd.read_csv(cache_file)
        # Ensure we have the required columns
        required_cols = ['TEAM_ABBREVIATION', 'GAME_DATE', 'OPP_PTS', 'OPP_FG_PCT', 'OPP_FG3_PCT', 'OPP_REB', 'OPP_AST', 'OPP_TOV']
        if all(col in df.columns for col in required_cols):
            print("All required columns found in cached data")
            return df
        else:
            print("Cached data is missing some required columns. Refetching...")
    
    print("\nFetching team defensive stats...")
    all_teams = teams.get_teams()
    all_defense = []
    
    for team in tqdm(all_teams, desc="Fetching team stats"):
        try:
            # Get team game log - this includes both team and opponent stats
            gamelog = teamgamelog.TeamGameLog(
                team_id=team['id'],
                season=season,
                season_type_all_star='Regular Season',
                league_id_nullable='00',  # NBA
                date_from_nullable=None,
                date_to_nullable=None
            )
            
            # Get the game logs
            df = gamelog.get_data_frames()[0]
            
            # Add team info
            df['TEAM_ABBREVIATION'] = team['abbreviation']
            
            # Rename columns to indicate these are opponent stats
            rename_dict = {
                'PTS': 'OPP_PTS',
                'FGM': 'OPP_FGM',
                'FGA': 'OPP_FGA',
                'FG_PCT': 'OPP_FG_PCT',
                'FG3M': 'OPP_FG3M',
                'FG3A': 'OPP_FG3A',
                'FG3_PCT': 'OPP_FG3_PCT',
                'FTM': 'OPP_FTM',
                'FTA': 'OPP_FTA',
                'FT_PCT': 'OPP_FT_PCT',
                'REB': 'OPP_REB',
                'AST': 'OPP_AST',
                'STL': 'OPP_STL',
                'BLK': 'OPP_BLK',
                'TOV': 'OPP_TOV',
                'PF': 'OPP_PF',
                'PLUS_MINUS': 'OPP_PLUS_MINUS'
            }
            
            # Apply renaming
            df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
            
            # Add to our collection
            all_defense.append(df)
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching stats for {team['full_name']}: {e}")
    
    if not all_defense:
        print("No team defense data was fetched")
        return pd.DataFrame()
    
    # Combine all team data
    df_all = pd.concat(all_defense, ignore_index=True)
    
    # Ensure we have the required columns
    required_cols = ['TEAM_ABBREVIATION', 'GAME_DATE', 'OPP_PTS', 'OPP_FG_PCT', 'OPP_FG3_PCT', 'OPP_REB', 'OPP_AST', 'OPP_TOV']
    missing_cols = [col for col in required_cols if col not in df_all.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Available columns:", df_all.columns.tolist())
    
    # Cache the results
    df_all.to_csv(cache_file, index=False)
    print(f"Saved team defense data to {cache_file}")
    
    return df_all
    
    # Print available columns for debugging
    print("\nAvailable columns in team stats:")
    print(df_all.columns.tolist())
    
    # Rename columns to be more descriptive
    # First, let's see what columns we actually have
    column_mapping = {}
    
    # Map available columns to our desired names
    for old, new in [
        ('PTS', 'TEAM_PTS'),
        ('FGA', 'TEAM_FGA'),
        ('FGM', 'TEAM_FGM'),
        ('FG3A', 'TEAM_FG3A'),
        ('FG3M', 'TEAM_FG3M'),
        ('FTA', 'TEAM_FTA'),
        ('FTM', 'TEAM_FTM'),
        ('REB', 'TEAM_REB'),
        ('AST', 'TEAM_AST'),
        ('STL', 'TEAM_STL'),
        ('BLK', 'TEAM_BLK'),
        ('TOV', 'TEAM_TOV'),
        ('FG_PCT', 'TEAM_FG_PCT'),
        ('FG3_PCT', 'TEAM_FG3_PCT'),
        ('FT_PCT', 'TEAM_FT_PCT')
    ]:
        if old in df_all.columns:
            column_mapping[old] = new
    
    df_all = df_all.rename(columns=column_mapping)
    
    # Calculate any missing percentage columns if we have the components
    if 'TEAM_FG_PCT' not in df_all.columns and 'TEAM_FGM' in df_all.columns and 'TEAM_FGA' in df_all.columns:
        df_all['TEAM_FG_PCT'] = df_all['TEAM_FGM'] / df_all['TEAM_FGA']
    if 'TEAM_FG3_PCT' not in df_all.columns and 'TEAM_FG3M' in df_all.columns and 'TEAM_FG3A' in df_all.columns:
        df_all['TEAM_FG3_PCT'] = df_all['TEAM_FG3M'] / df_all['TEAM_FG3A']
    
    # Create opponent stats by joining with the game data
    game_ids = df_all['Game_ID'].unique()
    all_opp_stats = []
    
    for game_id in tqdm(game_ids, desc="Processing game data"):
        game_data = df_all[df_all['Game_ID'] == game_id]
        if len(game_data) == 2:  # Should have exactly 2 rows (home and away)
            # Get the two teams in the game
            team1 = game_data.iloc[0]
            team2 = game_data.iloc[1]
            
            # Get available columns that exist in both team1 and team2
            available_cols = []
            for col in ['TEAM_PTS', 'TEAM_FG_PCT', 'TEAM_FG3_PCT', 'TEAM_REB', 'TEAM_AST', 'TEAM_TOV']:
                if col in team1 and col in team2:
                    available_cols.append(col)
            
            if not available_cols:
                print(f"Warning: No common stat columns found for game {game_id}")
                continue
                
            # Create opponent stats dataframes
            team1_opp_stats = team2[available_cols].copy()
            team1_opp_stats.columns = [col.replace('TEAM_', 'OPP_') for col in available_cols]
            team1_opp_stats['Game_ID'] = game_id
            team1_opp_stats['TEAM_ABBREVIATION'] = team1['TEAM_ABBREVIATION']
            
            team2_opp_stats = team1[available_cols].copy()
            team2_opp_stats.columns = [col.replace('TEAM_', 'OPP_') for col in available_cols]
            team2_opp_stats['Game_ID'] = game_id
            team2_opp_stats['TEAM_ABBREVIATION'] = team2['TEAM_ABBREVIATION']
            
            all_opp_stats.extend([team1_opp_stats, team2_opp_stats])
    
    if not all_opp_stats:
        return pd.DataFrame()
        
    # Combine all opponent stats
    opp_stats_df = pd.DataFrame(all_opp_stats)
    
    # Merge back with original data
    print(f"\n=== Merging Stats ===")
    print(f"Player stats shape before merge: {df_all.shape}")
    print(f"Opponent stats shape before merge: {opp_stats_df.shape}")
    print("Sample opponent stats columns:", list(opp_stats_df.columns)[:10])
    
    # Ensure the merge columns exist
    if 'TEAM_ABBREVIATION' not in df_all.columns:
        print("\n❌ TEAM_ABBREVIATION not found in player stats")
        print("Available columns in player stats:", list(df_all.columns))
        
    if 'TEAM_ABBREVIATION' not in opp_stats_df.columns:
        print("\n❌ TEAM_ABBREVIATION not found in opponent stats")
        print("Available columns in opponent stats:", list(opp_stats_df.columns))
    
    # Perform the merge
    df_all = pd.merge(
        df_all,
        opp_stats_df,
        on=['Game_ID', 'TEAM_ABBREVIATION'],
        how='left'
    )
    
    print(f"\nAfter merge: {df_all.shape[0]} rows, {df_all.shape[1]} columns")
    
    # Check for defensive columns in the merged data
    def_cols = [col for col in df_all.columns if 'OPP_' in col or 'DEF_' in col]
    print(f"Found {len(def_cols)} defensive columns in merged data")
    if def_cols:
        print("Sample defensive columns:", def_cols[:10])
    
    # Add game date for sorting
    if 'GAME_DATE' in df_all.columns:
        df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])
    
    # Save to cache
    df_all.to_csv(cache_file, index=False)
    return df_all

def get_player_game_logs(player_id, season='2023-24', retry_attempt=0, max_retries=2):
    """Get game logs for a player with robust error handling and retry logic
    
    Args:
        player_id: NBA player ID
        season: Season in format 'YYYY-YY'
        retry_attempt: Current retry attempt (0 = first try)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (DataFrame with player logs, bool success)
    """
    import random
    import time
    from datetime import datetime
    
    # Add jittered delay between requests to avoid rate limiting
    base_delay = 0.5 + (retry_attempt * 0.5)  # Increase delay with each retry
    jitter = random.uniform(0, 1)  # Add random jitter
    time.sleep(base_delay + jitter)
    
    try:
        # Get player game logs
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season',
            timeout=30  # Increase timeout to 30 seconds
        )
        
        # Get the data frame with error handling
        try:
            df = gamelog.get_data_frames()[0]
        except (ValueError, IndexError, KeyError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
        
        # Validate the response
        if df.empty:
            raise Exception("Empty response from API")
            
        # Check for required columns
        required_columns = {'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Ensure we have some data
        if len(df) < 1:
            raise Exception("No game logs found for player")
            
        return df, True
        
    except Exception as e:
        error_msg = f"Error getting game logs for player {player_id} (attempt {retry_attempt + 1}): {str(e)}"
        
        # Log the error
        if retry_attempt == 0:  # Only log on first attempt to avoid spam
            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}")
        
        # Implement exponential backoff for retries
        if retry_attempt < max_retries:
            retry_delay = (2 ** retry_attempt) + random.uniform(0, 1)
            time.sleep(retry_delay)
            return get_player_game_logs(player_id, season, retry_attempt + 1, max_retries)
        
        # If we've exhausted retries, log the final failure
        if retry_attempt == max_retries:
            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Final failure for player {player_id}")
            
        return pd.DataFrame(), False
        
    except Exception as e:
        print(f"Unexpected error for player {player_id}: {e}")
        return pd.DataFrame(), False

def calculate_rolling_stats(df, team_defense_df, player_col='PLAYER_NAME', date_col='GAME_DATE', stat_columns=None):
    """Calculate rolling statistics for each player and opponent defense
    
    Args:
        df: DataFrame containing player game logs
        team_defense_df: DataFrame with team defensive stats
        player_col: Name of the column containing player identifiers
        date_col: Name of the column containing game dates
        stat_columns: List of columns to calculate rolling stats for
        
    Returns:
        DataFrame with rolling statistics and EMAs
    """
    if stat_columns is None:
        stat_columns = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
            'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'MIN'
        ]
    
    # Ensure we only use columns that exist in the dataframe
    stat_columns = [col for col in stat_columns if col in df.columns]
    
    # Sort by player and date
    df = df.sort_values([player_col, date_col])
    
    # Handle missing values using expanding mean for each player's historical data
    # This ensures we don't look ahead when filling missing values
    for col in stat_columns:
        df[col] = df.groupby(player_col, group_keys=False)[col].apply(
            lambda x: x.fillna(x.expanding().mean())
        )
    
    # Calculate player rolling stats
    ema_spans = [5, 10, 20]  # EMA spans for different time windows
    
    # Player stats EMAs
    for col in stat_columns:
        for span in ema_spans:
            # Calculate EMA for each player
            df[f'PLAYER_{col}_EMA_{span}'] = df.groupby(player_col, group_keys=False)[col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean().shift(1)
            )
    
    # Process team defense if available
    if not team_defense_df.empty:
        print("\nProcessing team defense data...")
        print(f"Initial team defense shape: {team_defense_df.shape}")
        print("Team defense columns:", team_defense_df.columns.tolist())
        
        try:
            # Process team defense data
            team_defense_df = team_defense_df.copy()
            
            # Ensure we have the required columns
            if 'GAME_DATE' not in team_defense_df.columns:
                print("Error: GAME_DATE column not found in team defense data")
                return df
                
            # Convert date column to datetime
            team_defense_df[date_col] = pd.to_datetime(team_defense_df['GAME_DATE'])
            team_defense_df = team_defense_df.sort_values(['TEAM_ABBREVIATION', date_col])
            
            # Calculate team defense EMAs - using the actual column names from the team defense data
            defense_stats = []
            for prefix in ['OPP_', 'TEAM_']:
                for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV']:
                    col = f'{prefix}{stat}'
                    if col in team_defense_df.columns:
                        defense_stats.append(col)
            
            if not defense_stats:
                print("No defense stats columns found. Available columns:", team_defense_df.columns.tolist())
            else:
                print(f"Calculating EMAs for defense stats: {defense_stats}")
                
                for col in defense_stats:
                    for span in ema_spans:
                        ema_col = f'DEF_{col}_EMA_{span}'
                        team_defense_df[ema_col] = team_defense_df.groupby('TEAM_ABBREVIATION')[col].transform(
                            lambda x: x.ewm(span=span, adjust=False).mean().shift(1)
                        )
                
                # Get all possible defense columns that might have been created
                defense_ema_cols = [col for col in team_defense_df.columns 
                                 if any(f'DEF_{s}_' in col for s in ['OPP_', 'TEAM_'])]
                
                # Prepare columns for merge
                merge_cols = ['TEAM_ABBREVIATION', date_col] + defense_ema_cols
                merge_cols = [col for col in merge_cols if col in team_defense_df.columns]
                
                print(f"Merging {len(merge_cols) - 2} defensive stats columns")
                
                # Ensure we have data to merge
                if len(merge_cols) > 2:  # More than just the merge keys
                    # Ensure date columns are in datetime format
                    df[date_col] = pd.to_datetime(df[date_col])
                    team_defense_df[date_col] = pd.to_datetime(team_defense_df[date_col])
                    
                    # Print merge info for debugging
                    print(f"Merging with defense stats. Player data shape: {df.shape}")
                    print(f"Team defense data shape: {team_defense_df[merge_cols].shape}")
                    
                    # Merge opponent's defensive stats
                    df = pd.merge(
                        df,
                        team_defense_df[merge_cols],
                        left_on=['OPPONENT_TEAM_ABBREVIATION', date_col],
                        right_on=['TEAM_ABBREVIATION', date_col],
                        how='left'
                    )
                    
                    # Print merge results
                    def_cols = [col for col in df.columns if 'DEF_' in col]
                    print(f"After merge: {df.shape}, defensive columns: {len(def_cols)}")
                    if def_cols:
                        missing = df[def_cols].isnull().mean().sort_values(ascending=False)
                        print("Missing defensive stats (%):\n", missing.head())
                    
                    # Drop the extra column from merge if it exists
                    if 'TEAM_ABBREVIATION' in df.columns and 'TEAM_ABBREVIATION' in merge_cols:
                        df = df.drop(columns=['TEAM_ABBREVIATION'])
                    
                    # Check if defensive stats were added
                    def_cols = [col for col in df.columns if 'DEF_' in col]
                    print(f"Successfully added {len(def_cols)} defensive stats columns")
                    if def_cols:
                        print(f"Sample defensive columns: {def_cols[:5]}...")
                else:
                    print("No defensive stats columns available for merging")
                    
        except Exception as e:
            import traceback
            print(f"Error processing defensive stats: {e}")
            print(traceback.format_exc())
    else:
        print("No team defense data available to process")
    
    return df

def main():
    print("=== Starting NBA Player Stats EMA Calculation ===\n")
    start_time = time.time()
    
    # Output file
    output_file = 'nba_player_rolling_stats.csv'
    
    try:
        # Get team defensive stats first
        print("Fetching team defensive stats...")
        team_defense_df = get_team_defensive_stats('2023-24')
        
        # Process player data
        all_player_data = fetch_and_process_player_data('2023-24')
        
        # Combine all player data
        if not all_player_data:
            print("\n❌ No player data was processed")
            return
            
        df_all = pd.concat(all_player_data, ignore_index=True)
        
        # Ensure date columns are in datetime format
        if 'GAME_DATE' in df_all.columns:
            df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])
        
        if 'GAME_DATE' in team_defense_df.columns:
            team_defense_df['GAME_DATE'] = pd.to_datetime(team_defense_df['GAME_DATE'])
        
        # Calculate rolling statistics
        print("\nCalculating rolling statistics...")
        
        # Define stat columns to calculate EMAs for
        stat_columns = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM',
            'TOV', 'MIN', 'PLUS_MINUS', 'FANTASY_POINTS', 'USAGE_PCT'
        ]
        
        # Only keep columns that exist in the dataframe
        stat_columns = [col for col in stat_columns if col in df_all.columns]
        
        # Calculate rolling stats
        df_all = calculate_rolling_stats(
            df_all,
            team_defense_df,
            player_col='PLAYER_NAME',
            date_col='GAME_DATE',
            stat_columns=stat_columns
        )

        # Merge all player data
        all_player_data = pd.concat([df_all], ignore_index=True)

        # Check if defensive stats are present
        def_cols = [col for col in all_player_data.columns if 'DEF_' in col]
        print(f"\nFound {len(def_cols)} defensive stats columns in final output")
        if def_cols:
            print(f"Sample defensive columns: {def_cols[:5]}...")

        # Save to CSV
        all_player_data.to_csv(output_file, index=False)
        print(f"\n Successfully saved {len(all_player_data)} player games to {output_file}")
        print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

        # Print column information for debugging
        print("\n=== Column Information ===")
        print(f"Total columns: {len(all_player_data.columns)}")
        print("\nFirst 20 columns:", list(all_player_data.columns[:20]))
        print("\nLast 20 columns:", list(all_player_data.columns[-20:]))
        print("\nDefensive columns:", [col for col in all_player_data.columns if 'DEF_' in col])

        # Print summary
        print(f"\n Results saved to {output_file}")
        print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")

    except Exception as e:
        print(f"\n An error occurred: {e}")
        import traceback
        traceback.print_exc()  # This will print the full traceback for debugging
        print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")
        
        # Fallback to a predefined list of common player IDs if the API fails
        print("\n⚠️ Using a predefined list of common player IDs as fallback")
        common_player_ids = [
            2544,    # LeBron James
            201939,  # Stephen Curry
            201142,  # Kevin Durant
            203999,  # Nikola Jokić
            203507,  # Giannis Antetokounmpo
            1629029, # Luka Dončić
            203954,  # Joel Embiid
            1628368, # Jayson Tatum
            1629023, # De'Aaron Fox
            203076,  # Anthony Davis
            1626164, # Karl-Anthony Towns
            1627742, # Brandon Ingram
            1628378, # De'Aaron Fox
            1627732, # Domantas Sabonis
            1627759, # Jamal Murray
            203507,  # Giannis Antetokounmpo (duplicate, but kept for emphasis)
            1626174, # Devin Booker
            1630162, # LaMelo Ball
            1630178, # Anthony Edwards
            1630169, # Tyrese Haliburton
        ]
        
        # Create a DataFrame with the fallback players
        fallback_players = pd.DataFrame([{
            'id': pid,
            'full_name': 'Player ' + str(pid),  # Placeholder name
            'is_active': True
        } for pid in common_player_ids])
        
        # Process the fallback players
        all_player_data = []
        for _, player in fallback_players.iterrows():
            try:
                player_id = player['id']
                player_name = player['full_name']
                print(f"\nProcessing fallback player: {player_name} (ID: {player_id})")
                
                # Get player logs with retries
                player_logs, success = get_player_game_logs(
                    player_id,
                    season='2023-24',
                    retry_attempt=0,
                    max_retries=2  # Fewer retries for fallback to be faster
                )
                
                if success and not player_logs.empty:
                    player_logs = process_player_logs(player_logs, player_id, player_name)
                    all_player_data.append(player_logs)
                    print(f"✅ Successfully processed {player_name}")
                else:
                    print(f"⚠️ Could not fetch data for {player_name}")
                    
                time.sleep(1)  # Small delay between API calls
                
            except Exception as e:
                print(f"Error processing fallback player {player_id}: {e}")
        
        if not all_player_data:
            print("\n❌ No player data was processed, even with fallback")
            return
            
        # Continue with the rest of the processing
        df_all = pd.concat(all_player_data, ignore_index=True)
        
        # Ensure date columns are in datetime format
        if 'GAME_DATE' in df_all.columns:
            df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])
        
        if 'GAME_DATE' in team_defense_df.columns:
            team_defense_df['GAME_DATE'] = pd.to_datetime(team_defense_df['GAME_DATE'])
        
        # Continue with the rest of the processing
        print("\nCalculating rolling statistics with fallback players...")
        
        # Define stat columns to calculate EMAs for
        stat_columns = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM',
            'TOV', 'MIN', 'PLUS_MINUS', 'FANTASY_POINTS', 'USAGE_PCT'
        ]
        
        # Only keep columns that exist in the dataframe
        stat_columns = [col for col in stat_columns if col in df_all.columns]
        
        # Calculate rolling stats
        df_all = calculate_rolling_stats(
            df_all,
            team_defense_df,
            player_col='PLAYER_NAME',
            date_col='GAME_DATE',
            stat_columns=stat_columns
        )
        
        # Save results to CSV with a different name to indicate fallback was used
        output_file = 'nba_player_rolling_stats_fallback.csv'
        df_all.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n✅ Fallback results saved to {output_file}")
        print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")
    
    # First pass: Try all players with no retries
    print("\n=== First Pass: Fetching all players (no retries) ===")
    all_player_data = []
    failed_players = []
    
    for i, player in tqdm(players.iterrows(), total=len(players), desc="First pass"):
        player_id = player['PERSON_ID']
        player_name = player['DISPLAY_FIRST_LAST']
        
        # Get player game logs (no retries on first pass)
        player_logs, success = get_player_game_logs(
            player_id, 
            season='2023-24',
            retry_attempt=0,
            max_retries=0
        )
        
        if not success or player_logs.empty:
            failed_players.append((player_id, player_name))
            continue
        
        # Process successful response
        player_logs = process_player_logs(player_logs, player_id, player_name)
        all_player_data.append(player_logs)
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    
    # Second pass: Retry failed players with exponential backoff
    if failed_players:
        print(f"\n=== Second Pass: Retrying {len(failed_players)} failed players ===")
        max_retries = 2  # Maximum of 2 retry attempts
        
        for attempt in range(max_retries):
            retry_players = failed_players.copy()
            failed_players = []
            
            if not retry_players:
                break
                
            print(f"\nRetry attempt {attempt + 1}/{max_retries}")
            
            for player_id, player_name in tqdm(retry_players, desc=f"Retry {attempt + 1}"):
                player_logs, success = get_player_game_logs(
                    player_id,
                    season='2023-24',
                    retry_attempt=attempt + 1,  # Current retry attempt
                    max_retries=max_retries
                )
                
                if not success or player_logs.empty:
                    failed_players.append((player_id, player_name))
                    continue
                    
                # Process successful response
                player_logs = process_player_logs(player_logs, player_id, player_name)
                all_player_data.append(player_logs)
                
                # Slightly longer delay for retries
                time.sleep(0.5)
    
    # Report final status
    print(f"\n=== Data Collection Complete ===")
    print(f"Successfully processed: {len(all_player_data)} players")
    if failed_players:
        print(f"Failed to process {len(failed_players)} players after retries")
        if len(failed_players) < 20:  # Don't spam if there are many failures
            print("Failed player IDs:", ", ".join([str(p[0]) for p in failed_players]))
    
    return all_player_data


def process_player_logs(player_logs, player_id, player_name):
    """Process and clean player game logs"""
    if player_logs.empty:
        return player_logs
        
    # Create a copy to avoid SettingWithCopyWarning
    player_logs = player_logs.copy()
    
    # Add player info
    player_logs['PLAYER_ID'] = player_id
    player_logs['PLAYER_NAME'] = player_name
    
    # Add opponent team abbreviation from MATCHUP
    if 'MATCHUP' in player_logs.columns:
        # Initialize with None
        player_logs['OPPONENT_TEAM_ABBREVIATION'] = None
        
        # Process home games (e.g., 'LAL vs. GSW' -> 'GSW')
        home_mask = player_logs['MATCHUP'].str.contains(' vs\. ', na=False)
        if home_mask.any():
            player_logs.loc[home_mask, 'OPPONENT_TEAM_ABBREVIATION'] = \
                player_logs.loc[home_mask, 'MATCHUP'].str.extract(r'vs\.\s*([A-Z]{2,})')[0]
        
        # Process away games (e.g., 'LAL @ GSW' -> 'GSW')
        away_mask = player_logs['MATCHUP'].str.contains(' @ ', na=False)
        if away_mask.any():
            player_logs.loc[away_mask, 'OPPONENT_TEAM_ABBREVIATION'] = \
                player_logs.loc[away_mask, 'MATCHUP'].str.extract(r'@\s*([A-Z]{2,})')[0]
        
        # For any remaining missing, try to extract any 2-3 letter team code
        missing_mask = player_logs['OPPONENT_TEAM_ABBREVIATION'].isna()
        if missing_mask.any():
            # Extract any 2-3 letter uppercase code that's not the team's own abbreviation
            player_team = player_logs['MATCHUP'].str.extract(r'^([A-Z]{2,})')[0]
            player_logs.loc[missing_mask, 'OPPONENT_TEAM_ABBREVIATION'] = \
                player_logs.loc[missing_mask, 'MATCHUP'].str.extract(r'[^A-Z]([A-Z]{2,3})(?:[^A-Z]|$)')[0]
    
    # Convert date to datetime if it exists
    if 'GAME_DATE' in player_logs.columns:
        player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'FG3A', 'TOV', 'MIN', 'PLUS_MINUS']
    for col in numeric_cols:
        if col in player_logs.columns:
            player_logs[col] = pd.to_numeric(player_logs[col], errors='coerce')
    
    # Calculate additional metrics if columns exist
    if all(col in player_logs.columns for col in ['FGM', 'FGA']):
        player_logs['FG_PCT'] = player_logs['FGM'] / player_logs['FGA']
    if all(col in player_logs.columns for col in ['FG3M', 'FG3A']):
        player_logs['FG3_PCT'] = player_logs['FG3M'] / player_logs['FG3A']
    if all(col in player_logs.columns for col in ['FTM', 'FTA']):
        player_logs['FT_PCT'] = player_logs['FTM'] / player_logs['FTA']
    
    # Calculate fantasy points (standard scoring)
    if all(col in player_logs.columns for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']):
        player_logs['FANTASY_POINTS'] = (
            player_logs['PTS'] + 
            player_logs['REB'] * 1.2 + 
            player_logs['AST'] * 1.5 + 
            player_logs['STL'] * 3 + 
            player_logs['BLK'] * 3 - 
            player_logs['TOV']
        )
    
    return player_logs


def fetch_active_players(max_retries=3):
    """Fetch active NBA players with retry logic"""
    for attempt in range(max_retries + 1):  # +1 because range is 0-based
        try:
            players = commonallplayers.CommonAllPlayers(
                is_only_current_season=1,
                timeout=30
            ).get_data_frames()[0]
            print(f"Found {len(players)} active players")
            return players
        except Exception as e:
            if attempt == max_retries:
                raise Exception(f"Failed to fetch active players after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff


    
    # Process players in batches to manage memory and API rate limits
    batch_size = 20
    total_players = len(players_df)
    processed_count = 0
    failed_players = []
    
    print(f"\nProcessing {total_players} players in batches of {batch_size}...")
    
    for i in range(0, total_players, batch_size):
        batch = players_df.iloc[i:i + batch_size]
        batch_data = []
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_players + batch_size - 1)//batch_size}")
        
        for _, player in tqdm(batch.iterrows(), total=len(batch), desc="Processing players"):
            player_id = player['PERSON_ID']
            player_name = player['DISPLAY_FIRST_LAST']
            
            # Get player game logs with retry logic
            player_logs, success = get_player_game_logs(
                player_id=player_id,
                season=season,
                retry_attempt=0,
                max_retries=2
            )
            
            if not success or player_logs.empty:
                failed_players.append((player_id, player_name))
                continue
            
            # Process the player logs
            try:
                processed_logs = process_player_logs(player_logs, player_id, player_name)
                if not processed_logs.empty:
                    batch_data.append(processed_logs)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing logs for {player_name} (ID: {player_id}): {e}")
                failed_players.append((player_id, player_name))
        
        # Add batch data to main list
        if batch_data:
            all_player_data.extend(batch_data)
        
        # Add delay between batches
        if i + batch_size < total_players:
            time.sleep(2)  # 2 second delay between batches
    
    # Process any remaining failed players with more aggressive retries
    if failed_players:
        print(f"\nRetrying {len(failed_players)} failed players with more aggressive settings...")
        retry_failed = []
        
        for player_id, player_name in tqdm(failed_players, desc="Retrying failed players"):
            player_logs, success = get_player_game_logs(
                player_id=player_id,
                season=season,
                retry_attempt=0,
                max_retries=3  # More retries for failed players
            )
            
            if not success or player_logs.empty:
                retry_failed.append((player_id, player_name))
                continue
            
            try:
                processed_logs = process_player_logs(player_logs, player_id, player_name)
                if not processed_logs.empty:
                    all_player_data.append(processed_logs)
                    processed_count += 1
            except Exception as e:
                print(f"Error processing retry logs for {player_name} (ID: {player_id}): {e}")
                retry_failed.append((player_id, player_name))
        
        failed_players = retry_failed
    
    # Final status report
    print("\n" + "="*50)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Data fetch complete")
    print("="*50)
    
    print(f"✅ Successfully processed: {processed_count}/{total_players} players")
    
    if failed_players:
        print(f"❌ Failed to process {len(failed_players)} players")
        if len(failed_players) <= 20:  # Don't spam if there are many failures
            print("Failed player IDs:")
            for i, (pid, name) in enumerate(failed_players, 1):
                print(f"  {i}. {name} (ID: {pid})")
        else:
            print(f"(Skipping details for {len(failed_players)} failed players)")
    
    return all_player_data

if __name__ == "__main__":
    main()

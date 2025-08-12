import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, teamgamelog, commonallplayers
from nba_api.stats.static import players, teams
import time
from tqdm import tqdm
import os
from datetime import datetime

def get_team_defensive_stats(season='2023-24'):
    """Get team defensive stats for the season"""
    cache_file = f'team_defense_{season.replace("-", "")}.csv'
    
    if os.path.exists(cache_file):
        print(f"Loading team defense data from cache: {cache_file}")
        df = pd.read_csv(cache_file)
        # Ensure date is in datetime format
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df
    
    print("\nFetching team defensive stats...")
    all_teams = teams.get_teams()
    all_defense = []
    
    for team in tqdm(all_teams, desc="Fetching team stats"):
        try:
            gamelog = teamgamelog.TeamGameLog(
                team_id=team['id'],
                season=season,
                season_type_all_star='Regular Season',
                league_id_nullable='00',
                timeout=30
            )
            
            df = gamelog.get_data_frames()[0]
            
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
            
            df = df.rename(columns=rename_dict)
            df['TEAM_ABBREVIATION'] = team['abbreviation']
            all_defense.append(df)
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching stats for {team['full_name']}: {e}")
    
    if not all_defense:
        print("No team defense data was fetched")
        return pd.DataFrame()
    
    df_all = pd.concat(all_defense, ignore_index=True)
    
    # Ensure date is in datetime format
    df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])
    
    # Cache the results
    df_all.to_csv(cache_file, index=False)
    print(f"Saved team defense data to {cache_file}")
    return df_all

def get_player_game_logs(player_id, season='2023-24'):
    """Get game logs for a player with a single attempt"""
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season',
            timeout=30
        )
        return gamelog.get_data_frames()[0], True
    except Exception as e:
        print(f"Error getting game logs for player {player_id}: {e}")
        return pd.DataFrame(), False

def process_player_logs(player_logs, player_id, player_name):
    """Process and clean player game logs"""
    if player_logs.empty:
        return pd.DataFrame()
    
    # Basic cleaning
    df = player_logs.copy()
    
    # Add player info
    df['PLAYER_ID'] = player_id
    df['PLAYER_NAME'] = player_name
    
    # Convert date to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Extract opponent team abbreviation from MATCHUP
    df['OPPONENT_TEAM_ABBREVIATION'] = df['MATCHUP'].str[-3:].str.upper()
    
    return df

def calculate_rolling_stats(df, team_defense_df, player_col='PLAYER_NAME', date_col='GAME_DATE'):
    """Calculate rolling statistics for each player and opponent defense"""
    if df.empty or team_defense_df.empty:
        print("No data to process")
        return df
    
    # Sort by player and date
    df = df.sort_values([player_col, date_col])
    
    # Calculate player EMAs
    ema_spans = [5, 10, 20]
    stat_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'MIN']
    
    print("Calculating player EMAs...")
    for col in stat_columns:
        if col in df.columns:
            for span in ema_spans:
                df[f'PLAYER_{col}_EMA_{span}'] = df.groupby(player_col, group_keys=False)[col].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean().shift(1)
                )
    
    # Merge with team defense stats
    print("Merging with team defense stats...")
    
    # Ensure date columns are in datetime format
    team_defense_df[date_col] = pd.to_datetime(team_defense_df[date_col])
    
    # Calculate team defense EMAs
    defense_stats = ['OPP_PTS', 'OPP_FG_PCT', 'OPP_FG3_PCT', 'OPP_REB', 'OPP_AST', 'OPP_TOV']
    defense_stats = [col for col in defense_stats if col in team_defense_df.columns]
    
    print(f"Calculating defense EMAs for: {defense_stats}")
    
    for col in defense_stats:
        for span in ema_spans:
            team_defense_df[f'DEF_{col}_EMA_{span}'] = team_defense_df.groupby('TEAM_ABBREVIATION')[col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean().shift(1)
            )
    
    # Get all defense EMA columns
    defense_ema_cols = [col for col in team_defense_df.columns if col.startswith('DEF_')]
    merge_cols = ['TEAM_ABBREVIATION', date_col] + defense_ema_cols
    
    # Merge with player data
    df = pd.merge(
        df,
        team_defense_df[merge_cols],
        left_on=['OPPONENT_TEAM_ABBREVIATION', date_col],
        right_on=['TEAM_ABBREVIATION', date_col],
        how='left'
    )
    
    # Drop the extra column from merge
    if 'TEAM_ABBREVIATION' in df.columns:
        df = df.drop(columns=['TEAM_ABBREVIATION'])
    
    return df

def main():
    print("=== Quick NBA Player Stats EMA Calculation ===\n")
    start_time = time.time()
    
    # Output file
    output_file = 'quick_nba_player_rolling_stats.csv'
    
    try:
        # Get team defensive stats first
        print("Fetching team defensive stats...")
        team_defense_df = get_team_defensive_stats('2023-24')
        
        if team_defense_df.empty:
            print("❌ No team defense data available")
            return
        
        # Get active players
        print("\nFetching active players...")
        active_players = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            timeout=30
        ).get_data_frames()[0]
        
        if active_empty := active_players.empty:
            print("❌ No active players found")
            return
        
        print(f"Found {len(active_players)} active players")
        
        # Process only the first 5 players for testing
        sample_players = active_players.head(5)
        all_player_data = []
        
        print("\nProcessing player game logs...")
        for _, player in tqdm(sample_players.iterrows(), total=len(sample_players)):
            player_id = player['PERSON_ID']
            player_name = f"{player['DISPLAY_FIRST_LAST']}"
            
            print(f"\nProcessing {player_name} (ID: {player_id})")
            
            # Get player logs
            logs, success = get_player_game_logs(player_id, '2023-24')
            
            if not success or logs.empty:
                print(f"  ❌ Failed to get logs for {player_name}")
                continue
            
            # Process logs
            processed_logs = process_player_logs(logs, player_id, player_name)
            
            if not processed_logs.empty:
                all_player_data.append(processed_logs)
                print(f"  ✅ Processed {len(processed_logs)} games")
        
        if not all_player_data:
            print("\n❌ No player data was processed")
            return
        
        # Combine all player data
        df_all = pd.concat(all_player_data, ignore_index=True)
        
        # Calculate rolling stats with defensive stats
        print("\nCalculating rolling statistics...")
        df_final = calculate_rolling_stats(df_all, team_defense_df)
        
        # Save to CSV
        df_final.to_csv(output_file, index=False)
        
        # Print summary
        elapsed = (time.time() - start_time) / 60
        print(f"\n✅ Successfully processed {len(df_final)} player games")
        print(f"Output saved to: {output_file}")
        print(f"Time taken: {elapsed:.2f} minutes")
        
        # Print defensive columns that were added
        def_cols = [col for col in df_final.columns if 'DEF_' in col]
        print(f"\nDefensive stats columns added ({len(def_cols)}):")
        print(def_cols[:10])  # Print first 10 defensive columns
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

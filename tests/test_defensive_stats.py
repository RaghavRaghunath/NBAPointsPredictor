import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, teamgamelog, commonallplayers
from nba_api.stats.static import players, teams
import time
import os
from tqdm import tqdm

def get_team_defensive_stats(season='2023-24'):
    """Get team defensive stats for the season"""
    cache_file = f'team_defense_{season.replace("-", "")}.csv'
    
    if os.path.exists(cache_file):
        print(f"Loading team defense data from cache: {cache_file}")
        df = pd.read_csv(cache_file)
        return df
    
    print("\nFetching team defensive stats...")
    all_teams = teams.get_teams()
    all_defense = []
    
    # Only process a few teams for testing
    for team in tqdm(all_teams[:5], desc="Fetching team stats"):
        try:
            gamelog = teamgamelog.TeamGameLog(
                team_id=team['id'],
                season=season,
                season_type_all_star='Regular Season',
                league_id_nullable='00',
                date_from_nullable=None,
                date_to_nullable=None
            )
            
            df = gamelog.get_data_frames()[0]
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
            
            df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
            all_defense.append(df)
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching stats for {team['full_name']}: {e}")
    
    if not all_defense:
        print("No team defense data was fetched")
        return pd.DataFrame()
    
    df_all = pd.concat(all_defense, ignore_index=True)
    df_all.to_csv(cache_file, index=False)
    print(f"Saved team defense data to {cache_file}")
    return df_all

def test_defensive_stats():
    # Get team defensive stats
    team_defense_df = get_team_defensive_stats('2023-24')
    
    if team_defense_df.empty:
        print("No team defense data available")
        return
    
    # Convert team defense dates to datetime
    team_defense_df['GAME_DATE'] = pd.to_datetime(team_defense_df['GAME_DATE'])
    
    print("\nTeam defense columns:", team_defense_df.columns.tolist())
    print("Sample team defense data:")
    print(team_defense_df[['TEAM_ABBREVIATION', 'GAME_DATE', 'OPP_PTS', 'OPP_FG_PCT', 'OPP_REB']].head())
    
    # Get unique team abbreviations from the defense data
    available_teams = team_defense_df['TEAM_ABBREVIATION'].unique()
    print("\nAvailable teams in defense data:", available_teams[:5], "...")  # Show first 5 teams
    
    if len(available_teams) == 0:
        print("No teams found in defense data")
        return
    
    # Create test data using actual teams and dates from the defense data
    sample_defense = team_defense_df.sample(min(10, len(team_defense_df)))
    
    test_data = {
        'PLAYER_NAME': ['Test Player'] * len(sample_defense),
        'GAME_DATE': sample_defense['GAME_DATE'].dt.strftime('%Y-%m-%d').tolist(),
        'OPPONENT_TEAM_ABBREVIATION': sample_defense['TEAM_ABBREVIATION'].tolist(),
        'PTS': np.random.randint(10, 30, size=len(sample_defense)),
        'REB': np.random.randint(3, 15, size=len(sample_defense)),
        'AST': np.random.randint(2, 10, size=len(sample_defense))
    }
    
    print("\nSample test data dates:", test_data['GAME_DATE'][:5])
    print("Sample test teams:", test_data['OPPONENT_TEAM_ABBREVIATION'][:5])
    
    test_df = pd.DataFrame(test_data)
    
    # Ensure both dataframes have datetime columns for GAME_DATE
    test_df['GAME_DATE'] = pd.to_datetime(test_df['GAME_DATE'])
    
    # Merge with team defense data
    merged_df = pd.merge(
        test_df,
        team_defense_df,
        left_on=['OPPONENT_TEAM_ABBREVIATION', 'GAME_DATE'],
        right_on=['TEAM_ABBREVIATION', 'GAME_DATE'],
        how='left'
    )
    
    # Print merge info for debugging
    print(f"\nMerge results: {len(merged_df)} rows (expected {len(test_df)})")
    print(f"Rows with defensive stats: {merged_df['OPP_PTS'].notna().sum()}")
    
    # Drop duplicate column from merge
    if 'TEAM_ABBREVIATION' in merged_df.columns:
        merged_df = merged_df.drop(columns=['TEAM_ABBREVIATION'])
    
    print("\nMerged test data with defensive stats:")
    defensive_cols = [col for col in merged_df.columns if 'OPP_' in col]
    print("Defensive columns found:", defensive_cols)
    
    if not defensive_cols:
        print("No defensive stats were merged. Available columns in team_defense_df:")
        print(team_defense_df.columns.tolist())
    else:
        print("\nSample merged data (first 5 rows):")
        print(merged_df[['PLAYER_NAME', 'GAME_DATE', 'OPPONENT_TEAM_ABBREVIATION', 'PTS'] + defensive_cols[:3]].head())
    
    # Save the test data for inspection
    test_output_file = 'test_defensive_stats.csv'
    merged_df.to_csv(test_output_file, index=False)
    print(f"\nSaved test data to {test_output_file}")

if __name__ == "__main__":
    test_defensive_stats()

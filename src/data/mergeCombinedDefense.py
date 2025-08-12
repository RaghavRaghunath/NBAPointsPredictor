import pandas as pd
import os

def merge_defense_with_player_stats():
    try:
        print("Loading player stats...")
        player_stats_df = pd.read_csv('nba_player_rolling_stats_with_defense.csv')
        
        # Convert column names to uppercase for case-insensitive comparison
        player_stats_df.columns = player_stats_df.columns.str.upper()
        
        print("Loading defense data...")
        defense_files = pd.read_csv('nba_defense_by_position_latest.csv')
        defense_files.columns = defense_files.columns.str.upper()

        # Check for required columns in defense data
        required_def_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'ROLLING_C_AVG', 'ROLLING_F_AVG', 'ROLLING_G_AVG']
        if not all(col in defense_files.columns for col in required_def_cols):
            missing = [col for col in required_def_cols if col not in defense_files.columns]
            print(f"Missing required columns in defense data: {missing}")
            return
            
        # Check for required columns in player stats
        required_player_cols = ['GAME_ID', 'OPPONENT_TEAM_ABBREVIATION']
        if not all(col in player_stats_df.columns for col in required_player_cols):
            missing = [col for col in required_player_cols if col not in player_stats_df.columns]
            print(f"Missing required columns in player stats: {missing}")
            return
    
        print("Processing defense data...")
        # Group by game and team, taking the mean of each position's rolling average
        defense_agg = defense_files.groupby(['GAME_ID', 'TEAM_ABBREVIATION']).agg({
            'ROLLING_C_AVG': 'mean',
            'ROLLING_F_AVG': 'mean',
            'ROLLING_G_AVG': 'mean'
        }).reset_index()

        # Rename columns to indicate these are opponent's defensive stats
        defense_agg = defense_agg.rename(columns={
            'ROLLING_C_AVG': 'OPP_DEF_C_AVG',
            'ROLLING_F_AVG': 'OPP_DEF_F_AVG',
            'ROLLING_G_AVG': 'OPP_DEF_G_AVG'
        })

        print("Merging data...")
        # Merge with player stats on game ID and opponent team
        merged_df = pd.merge(
            player_stats_df,
            defense_agg,
            left_on=['GAME_ID', 'OPPONENT_TEAM_ABBREVIATION'],
            right_on=['GAME_ID', 'TEAM_ABBREVIATION'],
            how='left'
        )

        # Drop the extra TEAM_ABBREVIATION column from the merge if it exists
        if 'TEAM_ABBREVIATION' in merged_df.columns:
            merged_df = merged_df.drop(columns=['TEAM_ABBREVIATION'])

        # Save the merged data
        output_csv = 'combined_defensive_stats.csv'
        merged_df.to_csv(output_csv, index=False)
        
        print(f"Successfully merged data. Output saved to {output_csv}")
        print(f"Original player stats shape: {player_stats_df.shape}")
        print(f"Merged data shape: {merged_df.shape}")
        
        # Check for null values in the merged defensive stats
        if merged_df[['OPP_DEF_C_AVG', 'OPP_DEF_F_AVG', 'OPP_DEF_G_AVG']].isnull().any().any():
            print("\nWarning: Some defensive stats are missing (null values). This might be due to:")
            print("1. Games in player stats not found in defense data")
            print("2. Team abbreviations not matching between datasets")
            
            # Count missing values
            missing = merged_df[['OPP_DEF_C_AVG', 'OPP_DEF_F_AVG', 'OPP_DEF_G_AVG']].isnull().sum()
            print("\nMissing values per column:")
            print(missing)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_defense_with_player_stats()
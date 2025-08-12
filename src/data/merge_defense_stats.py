import pandas as pd
import os

def load_defense_teams(position):
    """Load best and worst defense teams for a given position."""
    best_file = f"nba_{position}_best_defense_stats.csv"
    worst_file = f"nba_{position}_worst_defense_stats.csv"
    
    try:
        best_teams = pd.read_csv(best_file)['Team'].tolist()
        worst_teams = pd.read_csv(worst_file)['Team'].tolist()
        return set(best_teams), set(worst_teams)
    except Exception as e:
        print(f"Error loading defense files for position {position}: {e}")
        return set(), set()

def main():
    # Define positions to process
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    
    # Load the main player stats dataset
    print("Loading main player stats dataset...")
    player_stats_file = 'nba_player_rolling_stats_single_pass_enhanced.csv'
    
    # Read in chunks to handle large file
    chunks = []
    for chunk in pd.read_csv(player_stats_file, chunksize=10000):
        chunks.append(chunk)
    
    df = pd.concat(chunks, axis=0)
    print(f"Loaded {len(df)} rows from {player_stats_file}")
    
    # Process each position
    for position in positions:
        print(f"\nProcessing {position} position...")
        best_teams, worst_teams = load_defense_teams(position)
        
        if not best_teams or not worst_teams:
            print(f"Skipping {position} due to missing data")
            continue
            
        print(f"Found {len(best_teams)} best and {len(worst_teams)} worst defensive teams for {position}")
        
        # Create binary indicators for best/worst defense
        df[f'VS_BEST_DEF_{position}'] = df['OPPONENT_TEAM_ABBREVIATION'].isin(best_teams).astype(int)
        df[f'VS_WORST_DEF_{position}'] = df['OPPONENT_TEAM_ABBREVIATION'].isin(worst_teams).astype(int)
    
    # Save the enhanced dataset
    output_file = 'nba_player_rolling_stats_with_defense.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved enhanced dataset to {output_file}")
    
    # Print some stats
    print("\nDefense indicators added:")
    for position in positions:
        best_col = f'VS_BEST_DEF_{position}'
        worst_col = f'VS_WORST_DEF_{position}'
        if best_col in df.columns:
            print(f"{position}: {df[best_col].sum():,} games vs best defenses, {df[worst_col].sum():,} vs worst")

if __name__ == "__main__":
    main()

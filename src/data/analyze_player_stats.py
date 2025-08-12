import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
print("Loading data...")
df = pd.read_csv('nba_player_rolling_stats.csv')

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Create output directory for reports
output_dir = Path('player_analysis_reports')
output_dir.mkdir(exist_ok=True)

# 1. Investigate Missing Opponent Data
def analyze_missing_opponents(df):
    print("\n=== Missing Opponent Analysis ===")
    missing_opponent = df[df['OPPONENT_TEAM_ABBREVIATION'].isna()]
    total_games = len(df)
    missing_count = len(missing_opponent)
    
    print(f"Total games: {total_games}")
    print(f"Games with missing opponent: {missing_count} ({missing_count/total_games:.1%})")
    
    # Check if missing opponents are from specific dates
    if not missing_opponent.empty:
        print("\nDate range of missing opponent data:")
        print(f"Earliest: {missing_opponent['GAME_DATE'].min().date()}")
        print(f"Latest: {missing_opponent['GAME_DATE'].max().date()}")
        
        # Check if MATCHUP column might contain the opponent info
        print("\nSample of MATCHUP values with missing opponent:")
        print(missing_opponent['MATCHUP'].head())
    
    return missing_opponent

# 2. Data Validation
def validate_data(df):
    print("\n=== Data Validation ===")
    issues = {}
    
    # Check for negative values in stats that should be >= 0
    stats_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGA', 'FGM', 'FTA', 'FTM', 'TOV', 'MIN', 'FG3M', 'FG3A']
    negative_values = (df[stats_columns] < 0).any()
    issues['negative_values'] = negative_values[negative_values == True].index.tolist()
    
    # Check for impossible percentages
    percent_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
    invalid_percentages = ((df[percent_columns] < 0) | (df[percent_columns] > 1)).any()
    issues['invalid_percentages'] = invalid_percentages[invalid_percentages == True].index.tolist()
    
    # Check for impossible stat combinations
    issues['fgm_gt_fga'] = len(df[df['FGM'] > df['FGA']])
    issues['ftm_gt_fta'] = len(df[df['FTM'] > df['FTA']])
    issues['fg3m_gt_fg3a'] = len(df[df['FG3M'] > df['FG3A']])
    issues['fg3m_gt_fgm'] = len(df[df['FG3M'] > df['FGM']])
    
    # Report issues
    print("Data validation results:")
    for issue, value in issues.items():
        if isinstance(value, list):
            if value:
                print(f"- Found {len(value)} columns with {issue}: {', '.join(value)}")
        elif value > 0:
            print(f"- Found {value} rows with {issue}")
    
    return issues

# 3. Generate Player Summaries
def generate_player_summaries(df, top_n=10):
    print("\n=== Generating Player Summaries ===")
    
    # Basic player stats
    player_stats = df.groupby('PLAYER_NAME').agg({
        'PTS': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'REB': 'mean',
        'AST': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'FG_PCT': 'mean',
        'FG3_PCT': 'mean',
        'FT_PCT': 'mean',
        'GAME_DATE': ['min', 'max']
    })
    
    # Flatten multi-index columns
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns.values]
    player_stats = player_stats.rename(columns={
        'PTS_count': 'GAMES_PLAYED',
        'PTS_mean': 'PTS_AVG',
        'REB_mean': 'REB_AVG',
        'AST_mean': 'AST_AVG',
        'STL_mean': 'STL_AVG',
        'BLK_mean': 'BLK_AVG',
        'FG_PCT_mean': 'FG_PCT',
        'FG3_PCT_mean': 'FG3_PCT',
        'FT_PCT_mean': 'FT_PCT',
        'GAME_DATE_min': 'FIRST_GAME',
        'GAME_DATE_max': 'LAST_GAME'
    })
    
    # Calculate fantasy points (standard 9-cat)
    player_stats['FANTASY_PTS'] = (
        player_stats['PTS_AVG'] + 
        player_stats['REB_AVG'] * 1.2 + 
        player_stats['AST_AVG'] * 1.5 +
        player_stats['STL_AVG'] * 2 +
        player_stats['BLK_AVG'] * 2 -
        player_stats['PTS_median'] * 0.5  # Simple TOV estimate
    )
    
    # Sort by fantasy points
    player_stats = player_stats.sort_values('FANTASY_PTS', ascending=False)
    
    # Save full player stats
    player_stats.to_csv(output_dir / 'player_summary_stats.csv')
    
    # Print top performers
    print("\nTop 10 Players by Fantasy Points:")
    print(player_stats[['GAMES_PLAYED', 'PTS_AVG', 'REB_AVG', 'AST_AVG', 'STL_AVG', 'BLK_AVG', 'FANTASY_PTS']].head(10))
    
    return player_stats

# 4. Team Defense Analysis
def analyze_team_defense(df):
    print("\n=== Team Defense Analysis ===")
    
    # Create a working copy
    defense_df = df.copy()
    
    # Extract team and opponent from MATCHUP
    def extract_teams(matchup):
        if pd.isna(matchup):
            return None, None
        parts = matchup.split()
        if len(parts) >= 3 and parts[1] in ['@', 'vs.']:
            return parts[0], parts[2]  # team, opponent
        return None, None
    
    # Add TEAM and OPPONENT columns
    defense_df[['TEAM', 'OPPONENT']] = defense_df['MATCHUP'].apply(
        lambda x: pd.Series(extract_teams(x))
    )
    
    # Group by game and team to get game totals
    game_totals = defense_df.groupby(['Game_ID', 'TEAM', 'OPPONENT']).agg({
        'PTS': 'sum',
        'REB': 'sum',
        'AST': 'sum',
        'FGA': 'sum',
        'FGM': 'sum',
        'GAME_DATE': 'first'
    }).reset_index()
    
    if game_totals.empty:
        print("No game data available for team defense analysis")
        return None
    
    # For each game, calculate points allowed
    points_allowed = []
    for game_id in game_totals['Game_ID'].unique():
        game_data = game_totals[game_totals['Game_ID'] == game_id]
        
        # We need exactly 2 teams per game (home and away)
        if len(game_data) == 2:
            team1 = game_data.iloc[0]
            team2 = game_data.iloc[1]
            
            # Team1's points allowed is team2's points scored
            points_allowed.append({
                'Game_ID': game_id,
                'TEAM': team1['TEAM'],
                'PTS_ALLOWED': team2['PTS'],
                'OPP_FG_PCT': team2['FGM'] / team2['FGA'] if team2['FGA'] > 0 else 0,
                'GAME_DATE': team1['GAME_DATE']
            })
            
            # Team2's points allowed is team1's points scored
            points_allowed.append({
                'Game_ID': game_id,
                'TEAM': team2['TEAM'],
                'PTS_ALLOWED': team1['PTS'],
                'OPP_FG_PCT': team1['FGM'] / team1['FGA'] if team1['FGA'] > 0 else 0,
                'GAME_DATE': team2['GAME_DATE']
            })
    
    if not points_allowed:
        print("Could not calculate points allowed - no valid game pairs found")
        return None
    
    # Create DataFrame from points allowed
    defense_stats = pd.DataFrame(points_allowed)
    
    # Calculate team defense stats
    team_defense = defense_stats.groupby('TEAM').agg({
        'PTS_ALLOWED': ['mean', 'count', 'std'],
        'OPP_FG_PCT': 'mean'
    })
    
    # Flatten multi-index columns
    team_defense.columns = ['_'.join(col).strip() for col in team_defense.columns.values]
    
    # Rename columns for clarity
    team_defense = team_defense.rename(columns={
        'PTS_ALLOWED_mean': 'PTS_ALLOWED',
        'PTS_ALLOWED_count': 'GAMES',
        'PTS_ALLOWED_std': 'PTS_ALLOWED_STD',
        'OPP_FG_PCT_mean': 'OPP_FG_PCT'
    })
    
    # Sort by points allowed (lower is better)
    team_defense = team_defense.sort_values('PTS_ALLOWED')
    
    # Save to CSV
    team_defense.to_csv(output_dir / 'team_defense_stats.csv')
    
    # Print top and bottom defenses
    print("\nTop 5 Defenses (Lowest PPG Allowed):")
    print(team_defense[['GAMES', 'PTS_ALLOWED', 'OPP_FG_PCT']].head())
    
    print("\nBottom 5 Defenses (Highest PPG Allowed):")
    print(team_defense[['GAMES', 'PTS_ALLOWED', 'OPP_FG_PCT']]
          .sort_values('PTS_ALLOWED', ascending=False).head())
    
    return team_defense

# 5. Trend Analysis
def plot_player_trends(df, player_name, stats=['PTS', 'REB', 'AST'], window=10):
    """Plot trend lines for a specific player"""
    player_data = df[df['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
    if len(player_data) < window:
        return  # Not enough data
    
    fig, axes = plt.subplots(len(stats), 1, figsize=(12, 3*len(stats)))
    if len(stats) == 1:
        axes = [axes]  # Make it iterable for single stat case
    
    for ax, stat in zip(axes, stats):
        # Calculate rolling average
        player_data[f'{stat}_ROLLING'] = player_data[stat].rolling(window=window).mean()
        
        # Plot
        ax.plot(player_data['GAME_DATE'], player_data[stat], 'o-', alpha=0.3, label='Game by Game')
        ax.plot(player_data['GAME_DATE'], player_data[f'{stat}_ROLLING'], 'r-', linewidth=2, label=f'{window}-Game Avg')
        
        # Formatting
        ax.set_title(f'{player_name} - {stat} Trend')
        ax.set_ylabel(stat)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{player_name.replace(' ', '_')}_trends.png")
    plt.close()

def analyze_all_players_trends(df, top_n=20, stats=['PTS', 'REB', 'AST'], window=10):
    """Analyze trends for top players"""
    # Get top players by minutes played
    top_players = df.groupby('PLAYER_NAME')['MIN'].sum().nlargest(top_n).index
    
    print(f"\nGenerating trend plots for top {len(top_players)} players...")
    for player in top_players:
        plot_player_trends(df, player, stats=stats, window=window)
    
    # Also generate a combined plot of top 5 players' scoring
    plt.figure(figsize=(12, 6))
    for i, player in enumerate(top_players[:5]):
        player_data = df[df['PLAYER_NAME'] == player].sort_values('GAME_DATE')
        player_data['PTS_MA'] = player_data['PTS'].rolling(window=window).mean()
        plt.plot(player_data['GAME_DATE'], player_data['PTS_MA'], 'o-', label=player, alpha=0.7)
    
    plt.title(f'Top 5 Players - {window}-Game Moving Average Points')
    plt.ylabel('Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_5_players_scoring_trend.png')
    plt.close()

# Run all analyses
def main():
    print("Starting NBA Player Stats Analysis")
    print("=" * 50)
    
    # 1. Missing opponent analysis
    missing_opponent = analyze_missing_opponents(df)
    
    # 2. Data validation
    issues = validate_data(df)
    
    # 3. Player summaries
    player_stats = generate_player_summaries(df)
    
    # 4. Team defense
    team_defense = analyze_team_defense(df)
    
    # 5. Trend analysis
    analyze_all_players_trends(df, top_n=20)
    
    print("\nAnalysis complete! Check the 'player_analysis_reports' directory for output files.")
    print("=" * 50)

if __name__ == "__main__":
    main()

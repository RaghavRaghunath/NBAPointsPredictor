import pandas as pd
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo

# Cache to store player positions
POSITION_CACHE = {}

def get_player_position(player_name, max_retries=3, retry_delay=1):
    """Get player position with retry logic and caching."""
    # Check cache first
    if player_name in POSITION_CACHE:
        return POSITION_CACHE[player_name]
    
    # Default to Guard
    position = 'G'
    
    for attempt in range(max_retries):
        try:
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                break
                
            player_id = player_dict[0]['id']
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info = player_info.get_normalized_dict()
            pos = info['CommonPlayerInfo'][0]['POSITION']
            
            if pd.notna(pos):
                if 'C' in pos:
                    position = 'C'
                elif 'F' in pos:
                    position = 'F'
                break
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error getting position for {player_name}: {str(e)}")
            time.sleep(retry_delay)
    
    # Cache the result
    POSITION_CACHE[player_name] = position
    return position

def main():
    print("Loading data...")
    player_df = pd.read_csv('nba_player_rolling_stats_with_defense.csv')
    defense_df = pd.read_csv('merged_defense_stats.csv')
    
    print("Adding positions...")
    # Get unique players first to minimize API calls
    unique_players = player_df['PLAYER_NAME'].unique()
    position_map = {player: get_player_position(player) for player in unique_players}
    
    # Map positions to the main dataframe
    player_df['POSITION'] = player_df['PLAYER_NAME'].map(position_map)
    
    print("Merging data...")
    # Convert IDs to string
    player_df['Game_ID'] = player_df['Game_ID'].astype(str)
    defense_df['GAME_ID'] = defense_df['GAME_ID'].astype(str)
    
    # Perform the merge
    final_df = pd.merge(
        player_df,
        defense_df,
        how='left',
        left_on=['Game_ID', 'OPPONENT_TEAM_ABBREVIATION', 'POSITION'],
        right_on=['GAME_ID', 'TEAM_ABBREVIATION', 'POSITION_GROUP'],
        suffixes=('', '_def'),
        indicator=True
    )
    
    # Check merge results
    print("\nMerge results:")
    print(final_df['_merge'].value_counts())
    
    # Save the result
    output_file = 'merged_player_defense.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved merged data to {output_file}")


if __name__ == "__main__":
    player_df = pd.read_csv('nba_player_rolling_stats_with_defense.csv')
    unique_players = player_df['PLAYER_NAME'].nunique()
    print(f"Unique players: {unique_players}")
    estimated_time = (unique_players * 1.25) / 60  # 1.25 seconds per player
    print(f"Estimated time: ~{estimated_time:.1f} minutes")
    main()
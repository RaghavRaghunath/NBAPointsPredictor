import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, teamgamelogs, commonteamroster
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
from geopy.distance import geodesic
import pytz

class NBADataEnhancer:
    def __init__(self, data_path='nba_player_rolling_stats_single_pass.csv'):
        """Initialize with path to the existing data."""
        self.data_path = data_path
        self.df = pd.read_csv(data_path, parse_dates=['GAME_DATE'])
        self.teams = teams.get_teams()
        self.players = players.get_players()
        self.team_id_to_abbr = {team['id']: team['abbreviation'] for team in self.teams}
        self.team_abbr_to_id = {team['abbreviation']: team['id'] for team in self.teams}
        self.team_cities = self._get_team_cities()
        
    def _get_team_cities(self):
        """Get team city information for distance calculation."""
        return {
            'ATL': (33.7573, -84.3963), 'BOS': (42.3662, -71.0621),
            'BKN': (40.6826, -73.9754), 'CHA': (35.2251, -80.8392),
            'CHI': (41.8807, -87.6742), 'CLE': (41.4966, -81.6882),
            'DAL': (32.7903, -96.8099), 'DEN': (39.7487, -105.0077),
            'DET': (42.3411, -83.0550), 'GSW': (37.7680, -122.3875),
            'HOU': (29.7508, -95.3621), 'IND': (39.7640, -86.1555),
            'LAC': (34.0430, -118.2673), 'LAL': (34.0430, -118.2673),
            'MEM': (35.1380, -90.0504), 'MIA': (25.7814, -80.1866),
            'MIL': (43.0439, -87.9172), 'MIN': (44.9795, -93.2761),
            'NOP': (29.9490, -90.0821), 'NYK': (40.7505, -73.9934),
            'OKC': (35.4634, -97.5151), 'ORL': (28.5392, -81.3836),
            'PHI': (39.9012, -75.1720), 'PHX': (33.4457, -112.0712),
            'POR': (45.5316, -122.6668), 'SAC': (38.5800, -121.4996),
            'SAS': (29.4270, -98.4375), 'TOR': (43.6435, -79.3791),
            'UTA': (40.7683, -111.9011), 'WAS': (38.8981, -77.0209)
        }

    def calculate_distance(self, team1, team2, date):
        """Calculate distance between two teams' arenas in miles."""
        if team1 not in self.team_cities or team2 not in self.team_cities:
            return np.nan
        return geodesic(self.team_cities[team1], self.team_cities[team2]).miles

    def get_team_game_dates(self, team_id, season):
        """Get all game dates for a team in a season."""
        try:
            gamelog = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
            df = gamelog.get_data_frames()[0]
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            return df[['Game_ID', 'GAME_DATE']].sort_values('GAME_DATE')
        except Exception as e:
            print(f"Error getting games for team {team_id}: {e}")
            return pd.DataFrame()

    def add_game_context(self):
        """Add game context features like home/away, days rest, etc."""
        print("Adding game context features...")
        
        # Extract home/away
        self.df['IS_HOME'] = self.df['MATCHUP'].str.contains(' vs. ').astype(int)
        
        # Sort by player and date
        self.df = self.df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        # Initialize new columns
        self.df['DAYS_REST'] = np.nan
        self.df['IS_BACK_TO_BACK'] = 0
        self.df['TRAVEL_DISTANCE'] = np.nan
        
        # Process each player's games
        for player_id in tqdm(self.df['PLAYER_ID'].unique(), desc="Processing players"):
            player_games = self.df[self.df['PLAYER_ID'] == player_id].copy()
            
            # Calculate days rest and back-to-back
            player_games['DAYS_REST'] = player_games['GAME_DATE'].diff().dt.days - 1
            player_games['IS_BACK_TO_BACK'] = (player_games['DAYS_REST'] == 0).astype(int)
            
            # Calculate travel distance
            for i in range(1, len(player_games)):
                if player_games['DAYS_REST'].iloc[i] > 0:  # Only calculate if not back-to-back
                    team = player_games['OPPONENT_TEAM_ABBREVIATION'].iloc[i-1]
                    next_team = player_games['OPPONENT_TEAM_ABBREVIATION'].iloc[i]
                    distance = self.calculate_distance(team, next_team, player_games['GAME_DATE'].iloc[i])
                    player_games.iloc[i, player_games.columns.get_loc('TRAVEL_DISTANCE')] = distance
            
            # Update the main dataframe
            self.df.update(player_games[['DAYS_REST', 'IS_BACK_TO_BACK', 'TRAVEL_DISTANCE']])
        
        # Fill any remaining NaNs
        self.df['DAYS_REST'] = self.df['DAYS_REST'].fillna(7)  # Default to 7 days if no previous game
        self.df['TRAVEL_DISTANCE'] = self.df['TRAVEL_DISTANCE'].fillna(0)
        
        return self.df

    def add_advanced_metrics(self):
        """Add advanced player metrics like usage rate, TS%, etc."""
        print("Adding advanced metrics...")
        
        # True Shooting Percentage (TS%)
        self.df['TS_PCT'] = self.df['PTS'] / (2 * (self.df['FGA'] + 0.44 * self.df['FTA'] + 0.001))
        
        # Calculate team totals for each game
        team_game_totals = self.df.groupby(['Game_ID', 'Player_ID']).agg({
            'FGA': 'sum',
            'FTA': 'sum',
            'TOV': 'sum',
            'MIN': lambda x: x.sum() / 5  # Approximate team minutes (5 players on court)
        }).reset_index()
        
        # Rename columns for merging
        team_game_totals = team_game_totals.rename(columns={
            'FGA': 'TEAM_FGA',
            'FTA': 'TEAM_FTA',
            'TOV': 'TEAM_TOV',
            'MIN': 'TEAM_MIN'
        })
        
        # Get team ID for each game from the original data
        team_mapping = self.df[['Game_ID', 'Player_ID', 'OPPONENT_TEAM_ABBREVIATION']].drop_duplicates()
        team_game_totals = pd.merge(
            team_game_totals,
            team_mapping,
            on=['Game_ID', 'Player_ID'],
            how='left'
        )
        
        # Merge team totals back to the main dataframe
        self.df = pd.merge(
            self.df, 
            team_game_totals, 
            on=['Game_ID', 'Player_ID', 'OPPONENT_TEAM_ABBREVIATION'],
            how='left'
        )
        
        # Calculate Usage Rate (USG%)
        # USG% = 100 * ((FGA + 0.44 * FTA + TOV) * (Team Minutes / 5)) / (MIN * (Team FGA + 0.44 * Team FTA + Team TOV))
        numerator = (self.df['FGA'] + 0.44 * self.df['FTA'] + self.df['TOV']) * (self.df['TEAM_MIN'] / 5)
        denominator = self.df['MIN'] * (self.df['TEAM_FGA'] + 0.44 * self.df['TEAM_FTA'] + self.df['TEAM_TOV'] + 0.001)
        self.df['USG_PCT'] = 100 * (numerator / denominator).clip(0, 1)  # Clip to 0-100% range
        
        # Clean up intermediate columns
        self.df = self.df.drop(['TEAM_FGA', 'TEAM_FTA', 'TEAM_TOV', 'TEAM_MIN'], axis=1)
        
        # Player Efficiency Rating (PER) components
        # PER is complex to calculate, but we can include some components
        self.df['PTS_PER_MIN'] = self.df['PTS'] / self.df['MIN'].replace(0, 1)
        self.df['AST_PER_MIN'] = self.df['AST'] / self.df['MIN'].replace(0, 1)
        self.df['REB_PER_MIN'] = self.df['REB'] / self.df['MIN'].replace(0, 1)
        
        return self.df

    def add_team_metrics(self):
        """Add team-level metrics like pace, offensive/defensive ratings."""
        print("Adding team metrics...")
        
        # This is a simplified version - in practice, you'd want to get these from the NBA API
        # For now, we'll add placeholders that can be filled with actual data
        
        # Team pace (possessions per 48 minutes)
        self.df['TEAM_PACE'] = 100.0  # Placeholder
        
        # Team offensive/defensive ratings
        self.df['TEAM_OFF_RTG'] = 110.0  # Placeholder
        self.df['TEAM_DEF_RTG'] = 110.0  # Placeholder
        
        return self.df

    def add_injury_data(self):
        """Add injury information."""
        print("Adding injury data...")
        
        # This would typically come from an injury API or scraping
        # For now, we'll add a placeholder column
        self.df['IS_INJURED'] = 0
        
        return self.df

    def add_playoff_push(self):
        """Add indicators for playoff push games."""
        print("Adding playoff push indicators...")
        
        # Mark last 10 games of regular season as playoff push
        season_ends = self.df.groupby('SEASON_ID')['GAME_DATE'].transform('max')
        self.df['IS_PLAYOFF_PUSH'] = (
            (season_ends - self.df['GAME_DATE']) <= pd.Timedelta(days=21)
        ).astype(int)
        
        return self.df

    def enhance_data(self, save_path=None):
        """Run all enhancement steps and save the result."""
        print("Starting data enhancement...")
        
        # Apply all enhancement steps
        self.add_game_context()
        self.add_advanced_metrics()
        self.add_team_metrics()
        self.add_injury_data()
        self.add_playoff_push()
        
        # Save the enhanced data
        if save_path is None:
            base, ext = os.path.splitext(self.data_path)
            save_path = f"{base}_enhanced{ext}"
        
        self.df.to_csv(save_path, index=False)
        print(f"Enhanced data saved to {save_path}")
        return self.df

if __name__ == "__main__":
    enhancer = NBADataEnhancer()
    enhanced_df = enhancer.enhance_data()

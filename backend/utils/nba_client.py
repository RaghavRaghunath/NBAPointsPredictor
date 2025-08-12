"""NBA API client for fetching current player and team data."""
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from nba_api.stats.endpoints import commonplayerinfo, commonallplayers
from nba_api.stats.static import teams
import pandas as pd
from requests_cache import install_cache

# Set up logging
logger = logging.getLogger(__name__)

# Set up cache to avoid hitting NBA API too frequently
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "nba_api_cache"

# Install cache with 1-hour expiration
install_cache(
    str(CACHE_FILE),
    backend="sqlite",
    expire_after=timedelta(hours=1),
    ignored_parameters=["headers", "timeout"]
)

class NBAClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NBAClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.teams = self._load_teams()
        self.players = self._load_players()
        self._team_rosters = {}
    
    def _load_teams(self) -> Dict[str, Dict]:
        """Load and index NBA teams."""
        teams_list = teams.get_teams()
        return {str(team['id']): team for team in teams_list}
    
    def _load_players(self) -> Dict[str, Dict]:
        """Load active NBA players."""
        try:
            players = commonallplayers.CommonAllPlayers(
                is_only_current_season=1
            ).get_data_frames()[0]
            
            # Convert to dictionary with player ID as key
            players_dict = {}
            for _, row in players.iterrows():
                player_id = str(row['PERSON_ID'])
                # Safely get each field with a default value if None
                players_dict[player_id] = {
                    'id': player_id,
                    'name': row.get('DISPLAY_FIRST_LAST', ''),
                    'team_id': str(row.get('TEAM_ID', '')),
                    'team_abbreviation': row.get('TEAM_ABBREVIATION', ''),
                    'jersey': row.get('JERSEY', ''),
                    'position': row.get('POSITION', ''),
                    'height': row.get('HEIGHT', ''),
                    'weight': row.get('WEIGHT', ''),
                    'season_exp': row.get('SEASON_EXP', 0),
                    'from_year': row.get('FROM_YEAR', ''),
                    'to_year': row.get('TO_YEAR', '')
                }
            return players_dict
        except Exception as e:
            logger.error(f"Error loading players: {e}")
            return {}
    
    def get_team_roster(self, team_id: str) -> List[Dict]:
        """Get current roster for a team."""
        if team_id in self._team_rosters:
            return self._team_rosters[team_id]
            
        try:
            # Get all players and filter by team
            roster = [
                player for player in self.players.values() 
                if player.get('team_id') == team_id
            ]
            self._team_rosters[team_id] = roster
            return roster
        except Exception as e:
            logger.error(f"Error getting roster for team {team_id}: {e}")
            return []
    
    def search_players(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for players by name, with fuzzy matching."""
        if not query or len(query) < 2:
            return []
            
        query = query.lower().strip()
        results = []
        
        # Handle empty player list
        if not self.players:
            logger.warning("No players loaded in the NBA client")
            return []
            
        # Split search terms
        terms = [t for t in query.split() if t]
        
        for player_id, player in self.players.items():
            try:
                name = player.get('name', '').lower()
                if not name:  # Skip players without a name
                    continue
                    
                team_abbr = str(player.get('team_abbreviation', '')).lower()
                position = str(player.get('position', '')).lower()
                
                # Calculate match score
                score = 0
                
                # Exact match
                if name == query:
                    score += 100
                
                # Check each search term
                for term in terms:
                    # Name matches
                    if term in name:
                        # Full first or last name match
                        if any(term == part for part in name.split()):
                            score += 10
                        else:
                            score += 5
                    
                    # Team abbreviation match (only if team_abbr is not empty)
                    if team_abbr and term in team_abbr:
                        score += 8
                        
                    # Position match (e.g., 'pg' for 'PG')
                    if position and term in position.lower():
                        score += 6
                
                if score > 0:
                    # Ensure we have all required fields with defaults
                    results.append((score, {
                        'id': player_id,
                        'name': player.get('name', 'Unknown Player'),
                        'team_abbreviation': team_abbr.upper() if team_abbr else '',
                        'position': player.get('position', ''),
                        'jersey': player.get('jersey', '')
                    }))
            except Exception as e:
                logger.warning(f"Error processing player {player_id}: {e}")
                continue
        
        # Sort by score (descending) and take top results
        results.sort(key=lambda x: -x[0])
        return [player for _, player in results[:limit]]
    
    def get_player_info(self, player_id: str) -> Optional[Dict]:
        """Get detailed info for a specific player."""
        try:
            player = self.players.get(str(player_id))
            if not player:
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
                df = player_info.get_data_frames()[0]
                if df.empty:
                    return None
                
                player = {
                    'id': str(player_id),
                    'name': df['DISPLAY_FIRST_LAST'].iloc[0],
                    'team_id': str(df['TEAM_ID'].iloc[0]),
                    'team_abbreviation': df['TEAM_ABBREVIATION'].iloc[0],
                    'position': df['POSITION'].iloc[0],
                    'jersey': df['JERSEY'].iloc[0],
                    'height': df['HEIGHT'].iloc[0],
                    'weight': df['WEIGHT'].iloc[0],
                }
                self.players[player_id] = player
            
            return player
        except Exception as e:
            logger.error(f"Error getting player info for {player_id}: {e}")
            return None

# Singleton instance
nba_client = NBAClient()

def get_nba_client() -> NBAClient:
    """Get the NBA client instance."""
    return nba_client

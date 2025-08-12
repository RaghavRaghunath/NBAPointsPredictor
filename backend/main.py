from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.nba_client import get_nba_client

# Make sure model package is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.model_predictor import get_predictor  # adjust if your path differs

app = FastAPI(title="NBA Predictor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# File used for search suggestions
# Go up one level from backend/ to reach project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CSV = os.path.join(project_root, "data", "raw", "merged_player_defense.csv")
print(f"Loading player data from: {DEFAULT_CSV}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir(os.path.dirname(DEFAULT_CSV))}")
_players_cache: Optional[pd.DataFrame] = None

def load_players_df() -> pd.DataFrame:
    global _players_cache
    if _players_cache is not None:
        return _players_cache
    if not os.path.exists(DEFAULT_CSV):
        logger.warning(f"Player CSV not found at {DEFAULT_CSV}; search will return empty.")
        _players_cache = pd.DataFrame(columns=["PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION"])
        return _players_cache
    try:
        # Load necessary columns including GAME_DATE to find most recent team
        usecols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION", "GAME_DATE"]
        df = pd.read_csv(DEFAULT_CSV, usecols=usecols, parse_dates=["GAME_DATE"])
        
        # Sort by GAME_DATE in descending order and keep the first occurrence of each player
        df = df.sort_values("GAME_DATE", ascending=False)
        df = df.drop_duplicates(subset=["PLAYER_NAME"], keep="first")
        
        # Drop the GAME_DATE column as it's no longer needed
        df = df.drop(columns=["GAME_DATE"])
        
        _players_cache = df
        logger.info(f"Loaded player data with {len(df)} unique players")
    except Exception as e:
        logger.error(f"Failed loading {DEFAULT_CSV}: {e}")
        _players_cache = pd.DataFrame(columns=["PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION"])
    return _players_cache

class PlayerPredictionRequest(BaseModel):
    player_name: str
    stat: str = "PTS"
    line: float
    game_date: Optional[str] = None
    is_home: bool = True
    # optional context (accepted but not required by the predictor)
    days_rest: Optional[int] = None
    opponent: Optional[str] = None
    last_n_games: Optional[int] = None

class ParlayRequest(BaseModel):
    legs: List[dict]

# Predictor singleton
predictor = get_predictor()

def get_prediction(player_name: str, stat: str, line: float, is_home: bool = True) -> dict:
    try:
        payload = {
            "player_name": player_name,
            "stat": stat,
            "line": float(line),
            "is_home": 1.0 if is_home else 0.0,
        }
        result = predictor.predict(payload)
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "Prediction failed"))
        yhat = float(result.get("prediction", line))
        conf = float(result.get("confidence", 0.65))
        return {
            "player_name": player_name,
            "stat": stat,
            "line": line,
            "prediction": round(yhat, 2),
            "confidence": round(conf, 3),
            "over_probability": round(conf if yhat > line else 1 - conf, 3),
            "success": True,
        }
    except Exception as e:
        logger.exception(f"Prediction error for {player_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/players/search")
async def search_players(
    q: str | None = Query(None, min_length=2, max_length=50),
    query: str | None = Query(None, min_length=2, max_length=50),
    limit: int = 8,
):
    """Search for NBA players with flexible matching using the NBA API.
    
    Supports:
    - Partial name matches (e.g., 'jalen j' will match 'Jalen Johnson')
    - Case-insensitive search
    - Team abbreviations (e.g., 'lebron lal' will match LeBron James on the Lakers)
    - Position search (e.g., 'pg' will match point guards)
    """
    try:
        nba = get_nba_client()
        term = (q or query or "").strip()
        
        if not term or len(term) < 2:
            return []
            
        # Use the NBA API client to search for players
        players = nba.search_players(term, limit=limit)
        
        # Format the response
        results = []
        for player in players:
            results.append({
                "id": player['id'],
                "name": player['name'],
                "team": player.get('team_abbreviation', ''),
                "position": player.get('position', '')
            })
            
        return results
        
    except Exception as e:
        logger.error(f"search_players failed: {e}", exc_info=True)
        # Fallback to local search if NBA API fails
        try:
            return await _fallback_search_players(term or "", limit)
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Search failed: {str(e)}. Fallback also failed: {str(fallback_error)}"
            )

async def _fallback_search_players(term: str, limit: int) -> List[Dict]:
    """Fallback search using local data if NBA API is unavailable."""
    df = load_players_df()
    if len(term) < 2:
        return []
        
    # Split search terms by spaces and filter out empty strings
    search_terms = [t for t in term.lower().split() if t]
    
    def search_score(row):
        """Calculate a search score for each player based on search terms."""
        name = str(row["PLAYER_NAME"]).lower()
        team = str(row.get("TEAM_ABBREVIATION", "")).lower()
        position = str(row.get("POSITION", "")).lower()
        
        score = 0
        
        # Exact match gets highest priority
        if name == term.lower():
            score += 100
            
        # Check each search term against player properties
        for t in search_terms:
            # Name matches
            if t in name:
                # Full first or last name match gets higher score
                if any(part == t for part in name.split()):
                    score += 10
                else:
                    score += 5
            
            # Team abbreviation match (e.g., 'lal' for 'LAL')
            if t in team:
                score += 8
                
            # Position match (e.g., 'pg' for 'PG')
            if t in position:
                score += 6
                
        return score
        
    # Calculate scores and filter out non-matches
    df['score'] = df.apply(search_score, axis=1)
    matches = df[df['score'] > 0].sort_values('score', ascending=False).head(limit)
    
    # Convert to response format
    results = []
    for _, r in matches.iterrows():
        results.append({
            "id": r["PLAYER_NAME"],  # Using name as ID for backward compatibility
            "name": r["PLAYER_NAME"],
            "team": r.get("TEAM_ABBREVIATION", "") or "",
            "position": r.get("POSITION", "") or "",
        })
        
    return results

@app.post("/api/predict/player")
async def predict_player(req: PlayerPredictionRequest):
    logger.info(f"Predict: {req.player_name} {req.stat} {req.line}")
    return get_prediction(req.player_name, req.stat, req.line, req.is_home)

@app.post("/api/parlay/calculate")
async def calculate_parlay(request: ParlayRequest):
    try:
        if not request.legs:
            raise HTTPException(status_code=400, detail="No legs provided")
        combined_prob = 1.0
        legs_out = []
        for leg in request.legs:
            pred = get_prediction(
                player_name=leg.get("player_name", ""),
                stat=leg.get("stat", "PTS"),
                line=float(leg.get("line", 0.0)),
                is_home=bool(leg.get("is_home", True)),
            )
            legs_out.append(pred)
            combined_prob *= pred["over_probability"]
        fair_odds = f"+{int((1/combined_prob - 1)*100) if combined_prob > 0 else 0}"
        return {
            "success": True,
            "probability": round(combined_prob, 4),
            "fair_odds": fair_odds,
            "recommendation": "Bet" if combined_prob > 0.5 else "Pass",
            "legs": legs_out,
        }
    except Exception as e:
        logger.exception(f"Parlay calc error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import pandas as pd
import time
import os
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats, commonteamroster
from nba_api.stats.static import players, teams
from io import StringIO

def fetch_teamrankings_stat(url):
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table")
    
    if table is None:
        print(f"❌ No table found at URL: {url}")
        return pd.DataFrame()

    try:
        table_html = str(table)
        df = pd.read_html(StringIO(table_html))[0]
    except Exception as e:
        print(f"❌ Error reading HTML table from {url}: {e}")
        return pd.DataFrame()

    # Handle multi-level headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # Rename first column to "Team" safely
    df.columns = [col if i != 0 else "Team" for i, col in enumerate(df.columns)]
    
    if "Team" not in df.columns:
        print(f"❌ 'Team' column not found after parsing {url}")
        return pd.DataFrame()

    # Ensure column is string before applying str operations
    df["Team"] = df["Team"].astype(str).str.replace(r"\d+", "", regex=True).str.strip()
    return df


def scrape_all_defensive_stats():
    urls = {
        "opp_ppg": "https://www.teamrankings.com/nba/stat/opponent-points-per-game",
        "opp_margin": "https://www.teamrankings.com/nba/stat/opponent-average-scoring-margin",
        "def_eff": "https://www.teamrankings.com/nba/stat/defensive-efficiency",
        "opp_floor_pct": "https://www.teamrankings.com/nba/stat/opponent-floor-percentage",
        "opp_paint_pts": "https://www.teamrankings.com/nba/stat/opponent-points-in-paint-per-game",
        "opp_fastbreak_pts": "https://www.teamrankings.com/nba/stat/opponent-fastbreak-points-per-game",
        "opp_fastbreak_eff": "https://www.teamrankings.com/nba/stat/opponent-fastbreak-efficiency",
        "opp_2pt_pts": "https://www.teamrankings.com/nba/stat/opponent-points-from-2-pointers",
        "opp_3pt_pts": "https://www.teamrankings.com/nba/stat/opponent-points-from-3-pointers",
        "opp_pct_2pt": "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-2-pointers",
        "opp_pct_3pt": "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-3-pointers",
        "opp_pct_ft": "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-free-throws",
        "opp_fg_pct": "https://www.teamrankings.com/nba/stat/opponent-shooting-pct",
        "opp_efg_pct": "https://www.teamrankings.com/nba/stat/opponent-effective-field-goal-pct",
        "opp_3p_pct": "https://www.teamrankings.com/nba/stat/opponent-three-point-pct",
        "opp_2p_pct": "https://www.teamrankings.com/nba/stat/opponent-two-point-pct",
        "opp_fgm": "https://www.teamrankings.com/nba/stat/opponent-field-goals-made-per-game",
        "opp_fga": "https://www.teamrankings.com/nba/stat/opponent-field-goals-attempted-per-game",
        "opp_3pm": "https://www.teamrankings.com/nba/stat/opponent-three-pointers-made-per-game",
    }
    dfs = []
    for name, url in urls.items():
        df = fetch_teamrankings_stat(url)
        df = df[["Team", df.columns[1]]].rename(columns={df.columns[1]: name})
        dfs.append(df)
    combined = dfs[0]
    for df in dfs[1:]:
        combined = pd.merge(combined, df, on="Team", how="outer")
    return combined

def get_players_averaging_7_ppg(season="2024-25"):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    return stats[stats["PTS"] >= 7]["PLAYER_NAME"].unique().tolist()

def get_player_id(player_name):
    result = players.find_players_by_full_name(player_name)
    return result[0]["id"] if result else None

def get_player_team(player_name):
    for t in teams.get_teams():
        try:
            roster = commonteamroster.CommonTeamRoster(t["id"]).get_data_frames()[0]
            if player_name in roster["PLAYER"].values:
                return t["full_name"]
        except:
            continue
    return None

def get_player_game_data(player_name, defense_df, season="2024-25"):
    pid = get_player_id(player_name)
    if not pid:
        return pd.DataFrame()
    try:
        df = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
    except:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("GAME_DATE")
    rolling_cols = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
                    "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "MIN"]
    for col in rolling_cols:
        df[f"avg_{col.lower()}_last_5"] = df[col].rolling(5).mean().shift(1)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
    df["opponent"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
    df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 5)
    df["player_name"] = player_name
    df["actual_pts"] = df["PTS"]

    team = get_player_team(player_name)
    if team and team in defense_df["Team"].values:
        row = defense_df[defense_df["Team"] == team].iloc[0]
        for col in defense_df.columns[1:]:
            df[col] = row[col]

    return df.dropna()

def build_dataset(output_csv="nba_defensive_augmented_dataset.csv", resume_from=0):
    defense_df = scrape_all_defensive_stats()
    player_list = get_players_averaging_7_ppg()

    all_data = []
    for i, name in enumerate(player_list[resume_from:], start=resume_from):
        print(f"🔄 ({i+1}/{len(player_list)}): {name}")
        df = get_player_game_data(name, defense_df)
        if not df.empty:
            all_data.append(df)
        time.sleep(0.5)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"✅ Saved to {output_csv}")
    else:
        print("⚠️ No data collected.")

if __name__ == "__main__":
    build_dataset()

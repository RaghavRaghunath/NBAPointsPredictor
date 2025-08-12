import pandas as pd
import time
import os
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats
from nba_api.stats.static import players

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_teamrankings_stat(url):
    print(f"🌐 Fetching: {url}")
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.content, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No table found at {url}")
    df = pd.read_html(str(table))[0]

    stat_name = url.split("/")[-1].replace("opponent-", "").replace("-", "_")

    if "Team" not in df.columns:
        df.columns.values[0] = "Team"

    df["Team"] = df["Team"].astype(str).str.replace(r"\\d+", "", regex=True).str.strip()
    df = df.rename(columns={col: f"{stat_name}_{col}" for col in df.columns if col != "Team"})
    print(f"✅ Fetched: {url}")
    return df

def scrape_all_defensive_stats():
    urls = [
        "https://www.teamrankings.com/nba/stat/opponent-points-per-game",
        "https://www.teamrankings.com/nba/stat/opponent-average-scoring-margin",
        "https://www.teamrankings.com/nba/stat/defensive-efficiency",
        "https://www.teamrankings.com/nba/stat/opponent-floor-percentage",
        "https://www.teamrankings.com/nba/stat/opponent-points-in-paint-per-game",
        "https://www.teamrankings.com/nba/stat/opponent-fastbreak-points-per-game",
        "https://www.teamrankings.com/nba/stat/opponent-fastbreak-efficiency",
        "https://www.teamrankings.com/nba/stat/opponent-points-from-2-pointers",
        "https://www.teamrankings.com/nba/stat/opponent-points-from-3-pointers",
        "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-2-pointers",
        "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-3-pointers",
        "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-free-throws",
        "https://www.teamrankings.com/nba/stat/opponent-shooting-pct",
        "https://www.teamrankings.com/nba/stat/opponent-effective-field-goal-pct",
        "https://www.teamrankings.com/nba/stat/opponent-three-point-pct",
        "https://www.teamrankings.com/nba/stat/opponent-two-point-pct",
        "https://www.teamrankings.com/nba/stat/opponent-field-goals-made-per-game",
        "https://www.teamrankings.com/nba/stat/opponent-field-goals-attempted-per-game",
        "https://www.teamrankings.com/nba/stat/opponent-three-pointers-made-per-game",
    ]
    combined = None
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_teamrankings_stat, url) for url in urls]
        for future in as_completed(futures):
            try:
                df = future.result()
                if combined is None:
                    combined = df
                else:
                    combined = pd.merge(combined, df, on="Team", how="outer")
            except Exception as e:
                print(f"❌ Failed to fetch stat: {e}")

    if combined is not None:
        print("✅ All defensive stats scraped and merged.")
    else:
        print("⚠️ No defensive data scraped.")

    return combined

def get_player_id(player_name):
    result = players.find_players_by_full_name(player_name)
    return result[0]["id"] if result else None

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

    # Merge defensive stats by opponent
    for team in defense_df["Team"]:
        if team[:3] in df["opponent"].values:
            sub_df = df[df["opponent"] == team[:3]]
            for col in defense_df.columns[1:]:
                df.loc[sub_df.index, col] = defense_df.loc[defense_df["Team"] == team, col].values[0]

    return df.dropna()

def get_players_averaging_7_ppg(season="2024-25"):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    return stats[stats["PTS"] >= 7]["PLAYER_NAME"].unique().tolist()

def build_dataset(output_csv="nba_full_dataset.csv", resume_from=0):
    defense_df = scrape_all_defensive_stats()
    player_list = get_players_averaging_7_ppg()

    all_data = []
    for i, name in enumerate(player_list[resume_from:], start=resume_from):
        print(f"🔄 ({i+1}/{len(player_list)}): {name}")
        df = get_player_game_data(name, defense_df)
        if not df.empty:
            print(f"✅ Added {len(df)} rows for {name}")
            all_data.append(df)
        time.sleep(0.2)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"✅ Final dataset saved to {output_csv}")
    else:
        print("⚠️ No player data was collected.")

if __name__ == "__main__":
    build_dataset()

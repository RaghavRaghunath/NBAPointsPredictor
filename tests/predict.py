import pandas as pd
import time
import os
from nba_api.stats.endpoints import (
    playergamelog,
    commonteamroster,
    leaguedashteamstats,
    leaguedashplayerstats
)
from nba_api.stats.static import players, teams

import requests
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import playergamelog
from requests.exceptions import RequestException
import time

def fetch_with_retry(player_id, season="2023-24", max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            return playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        except RequestException as e:
            print(f"⚠️ Retry {attempt + 1}/{max_retries} for player {player_id} due to: {e}")
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"❌ Unexpected error for player {player_id}: {e}")
            return pd.DataFrame()
    print(f"❌ Failed after {max_retries} retries for player {player_id}")
    return pd.DataFrame()


# --------------------------
# Step 1: Get Players ≥ 7 PPG
# --------------------------
def get_players_averaging_7_ppg(season="2023-24"):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    return stats[stats["PTS"] >= 7]["PLAYER_NAME"].unique().tolist()

# --------------------------
# Player Utilities
# --------------------------
def get_player_id(player_name):
    player_list = players.find_players_by_full_name(player_name)
    return player_list[0]["id"] if player_list else None

def get_player_team_abbr(player_name):
    all_teams = teams.get_teams()
    for team in all_teams:
        try:
            roster = commonteamroster.CommonTeamRoster(team["id"]).get_data_frames()[0]
            if player_name in roster["PLAYER"].values:
                return team["abbreviation"]
        except:
            continue
    return None

from concurrent.futures import ThreadPoolExecutor, as_completed

def get_top_teammates_ppg(player_name, team_abbr, season="2023-24"):
    team = [t for t in teams.get_teams() if t["abbreviation"] == team_abbr]
    if not team:
        return {}

    try:
        roster = commonteamroster.CommonTeamRoster(team[0]["id"]).get_data_frames()[0]
    except:
        print(f"❌ Could not retrieve roster for team {team_abbr}")
        return {}

    teammates = roster[roster["PLAYER"] != player_name]["PLAYER"].tolist()
    ppg_list = []

    def fetch_teammate_stats(teammate_name):
        try:
            pid = get_player_id(teammate_name)
            if not pid:
                return None
            print(f"⏳ Fetching stats for {teammate_name}...")
            df = fetch_with_retry(pid, season=season)
            if df.empty:
                return None
            avg_ppg = df["PTS"].astype(float).mean()
            return (teammate_name, avg_ppg)
        except Exception as e:
            print(f"⚠️ Failed to fetch stats for {teammate_name}: {e}")
            return None


    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_name = {executor.submit(fetch_teammate_stats, name): name for name in teammates}
        for future in as_completed(future_to_name):
            result = future.result()
            if result:
                ppg_list.append(result)

    top_teammates = sorted(ppg_list, key=lambda x: x[1], reverse=True)[:2]
    result = {}
    for i, (name, ppg) in enumerate(top_teammates):
        result[f"teammate_{i+1}_ppg"] = round(ppg, 2)
        result[f"teammate_{i+1}_name"] = name
    return result


# --------------------------
# Player Game Data
# --------------------------
def get_player_game_data_predictive(player_name, season="2023-24"):
    pid = get_player_id(player_name)
    if not pid:
        return pd.DataFrame()

    try:
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season)
        df = gamelog.get_data_frames()[0]

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["is_home"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
        df["opponent"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
        df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 5)

        df["avg_minutes_last_5"] = df["MIN"].rolling(5).mean().shift(1)
        df["avg_fga_last_5"] = df["FGA"].rolling(5).mean().shift(1)
        df["avg_rebounds_last_5"] = df["REB"].rolling(5).mean().shift(1)
        df["avg_points_last_5"] = df["PTS"].rolling(5).mean().shift(1)
        df["avg_assists_last_5"] = df["AST"].rolling(5).mean().shift(1)
        df["trend_last_3_pts"] = df["PTS"].diff().rolling(3).mean().shift(1)

        df["player_name"] = player_name
        df["actual_pts"] = df["PTS"]

        team_abbr = get_player_team_abbr(player_name)
        teammates_info = get_top_teammates_ppg(player_name, team_abbr, season=season)
        for k, v in teammates_info.items():
            df[k] = v

        return df[[
            "GAME_DATE", "player_name", "opponent", "is_home", "days_rest",
            "avg_minutes_last_5", "avg_fga_last_5", "avg_rebounds_last_5",
            "avg_points_last_5", "avg_assists_last_5", "trend_last_3_pts",
            "teammate_1_name", "teammate_1_ppg", "teammate_2_name", "teammate_2_ppg",
            "actual_pts"
        ]].dropna()

    except Exception as e:
        print(f"❌ Error for {player_name}: {e}")
        return pd.DataFrame()

# --------------------------
# Build Dataset
# --------------------------
def build_predictive_dataset(output_csv="nba_predictive_dataset_with_teammates.csv", resume_from=0):
    player_list = get_players_averaging_7_ppg()
    all_data = []

    for i, name in enumerate(player_list[resume_from:], start=resume_from):
        print(f"🔄 ({i + 1}/{len(player_list)}): {name}")
        df = get_player_game_data_predictive(name)
        if not df.empty:
            all_data.append(df)
        time.sleep(0.6)

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved to {output_csv}")
        return full_df
    else:
        print("⚠️ No data collected.")
        return pd.DataFrame()

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    build_predictive_dataset()

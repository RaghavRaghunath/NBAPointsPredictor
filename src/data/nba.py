import pandas as pd
import time
import os
from nba_api.stats.endpoints import playergamelog, commonteamroster, leaguedashteamstats
from nba_api.stats.static import players, teams

# --------------------------
# Scrape opponent position defense stats
# --------------------------
def scrape_points_allowed_by_position():
    url = "https://www.fantasypros.com/daily-fantasy/nba/fanduel-defense-vs-position.php"
    tables = pd.read_html(url)
    df = tables[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    df = df.rename(columns={"Team": "team_name"})

    team_abbr = {
        "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BRK", "Charlotte": "CHO",
        "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
        "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
        "LA Clippers": "LAC", "LA Lakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
        "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
        "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHO",
        "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
        "Utah": "UTA", "Washington": "WAS"
    }

    df["team"] = df["team_name"].map(team_abbr)

    result = {}
    for _, row in df.iterrows():
        result[row["team"]] = {
            "PG": row.get("PG", None),
            "SG": row.get("SG", None),
            "SF": row.get("SF", None),
            "PF": row.get("PF", None),
            "C": row.get("C", None)
        }

    return result

# --------------------------
# Get defensive ratings & pace from NBA API
# --------------------------
def get_defensive_stats():
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season="2023-24",
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]

    team_abbr_map = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK", "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA", "Washington Wizards": "WAS"
    }

    stats["TEAM_ABBREVIATION"] = stats["TEAM_NAME"].map(team_abbr_map)

    # Print columns for verification
    print("✅ Advanced stats columns:", stats.columns.tolist())

    return stats[["TEAM_ABBREVIATION", "DEF_RATING", "PACE"]]


def get_avg_pts_vs_opponent_last3(gamelog_df, current_game_date, opponent):
    past_games = gamelog_df[
        (gamelog_df["opponent"] == opponent) &
        (gamelog_df["GAME_DATE"] < current_game_date)
    ]
    last3 = past_games.sort_values("GAME_DATE").tail(3)
    return last3["PTS"].mean() if not last3.empty else None


# --------------------------
# Game data & enrichment
# --------------------------
def get_player_id(player_name):
    p = players.find_players_by_full_name(player_name)
    return p[0]["id"] if p else None

def get_player_game_data(player_name, season="2023-24"):
    pid = get_player_id(player_name)
    if not pid:
        return pd.DataFrame()

    try:
        log = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
        log["is_home"] = log["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
        log["opponent"] = log["MATCHUP"].apply(lambda x: x.split()[-1])
        log["avg_points_last_5"] = log["PTS"].rolling(5).mean().shift(1)
        log["avg_assists_last_5"] = log["AST"].rolling(5).mean().shift(1)
        log["trend_last_3_pts"] = log["PTS"].diff().rolling(3).mean().shift(1)
        log["days_rest"] = log["GAME_DATE"].diff().dt.days.fillna(2).clip(0, 5)
        log["player_name"] = player_name
        return log
    except:
        return pd.DataFrame()

def enrich(df, def_stats, full_player_gamelog):
    if df.empty:
        return df

    df["avg_pts_vs_same_opponent_last_3"] = df.apply(
    lambda row: get_avg_pts_vs_opponent_last3(full_player_gamelog, row["GAME_DATE"], row["opponent"]),
    axis=1
    )
    opp_team = df["opponent"].iloc[0]
    team_row = def_stats[def_stats["TEAM_ABBREVIATION"] == opp_team]
    if not team_row.empty:
        df["opponent_def_rating"] = team_row["DEF_RATING"].values[0]
        df["opponent_pace"] = team_row["PACE"].values[0]

    return df

# --------------------------
# Main function
# --------------------------
def build_dataset(resume_from=0, output_csv="nba_player_augmented_dataset.csv"):
    fantasypros_pos = scrape_points_allowed_by_position()
    def_stats = get_defensive_stats()
    all_players = players.get_active_players()

    existing_players = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_players = set(existing_df["player_name"].unique())
        except Exception as e:
            print(f"⚠️ Couldn't read existing CSV: {e}")

    for i, p in enumerate(all_players[resume_from:], start=resume_from):
        name = p["full_name"]
        if name in existing_players:
            print(f"⏩ Skipping {name} (already in dataset)")
            continue

        print(f"🔄 Processing ({i + 1}): {name}")
        df = get_player_game_data(name)
        if df.empty:
            continue

        # TODO: Replace this with actual position detection logic
        df["position"] = "SG"

        try:
            df["opp_points_allowed_to_position"] = df.apply(
                lambda row: fantasypros_pos.get(row["opponent"], {}).get(row["position"], None),
                axis=1
            )
            df = enrich(df, def_stats, full_player_gamelog=player_df)
        except Exception as e:
            print(f"⚠️ Error enriching {name}: {e}")
            continue

        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode="a", index=False, header=False)
        else:
            df.to_csv(output_csv, index=False)

        print(f"✅ Added {name} to {output_csv}")
        time.sleep(0.6)

def build_dataset_v2(resume_from=0, output_path="full_nba_player_dataset_v2.csv"):
    fantasy_df = scrape_points_allowed_by_position()
    def_stats = get_defensive_stats()
    all_players = players.get_active_players()
    player_dfs = []

    for i, p in enumerate(all_players[resume_from:], start=resume_from):
        name = p["full_name"]
        print(f"\n🔄 Processing ({i + 1}/{len(all_players)}): {name}")

        player_df = get_player_game_data(name)
        if player_df.empty:
            continue

        try:
            # Merge FantasyPros stats (defense vs position)
            # player_df = player_df.merge(
            #     fantasy_df,
            #     left_on="opponent",
            #     right_on="opponent_abbr",
            #     how="left"
            # )

            # Merge NBA team defense stats
            player_df = player_df.merge(
                def_stats,
                left_on="opponent",
                right_on="TEAM_ABBREVIATION",
                how="left"
            )

            # Enrich with teammate info, historical trends, and new features
            player_df = enrich(player_df, def_stats, full_player_gamelog=player_df)

        except Exception as e:
            print(f"⚠️ Skipping {name} due to error: {e}")
            continue

        # Save output: write header only if the file doesn't exist
        if os.path.exists(output_path):
            player_df.to_csv(output_path, mode="a", index=False, header=False)
        else:
            player_df.to_csv(output_path, index=False)

        print(f"✅ Added {name} to {output_path}")
        time.sleep(0.6)  # avoid rate limits


if __name__ == "__main__":
    build_dataset_v2()

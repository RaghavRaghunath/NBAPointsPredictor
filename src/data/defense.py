import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_teamrankings_stat(url):
    try:
        print(f"🌐 Fetching: {url}")
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        table = soup.find("table")
        if table is None:
            print(f"⚠️ No table found at {url}")
            return pd.DataFrame()

        df = pd.read_html(str(table))[0]

        if "Team" not in df.columns:
            df.columns.values[0] = "Team"

        df["Team"] = df["Team"].astype(str).str.replace(r"\\d+", "", regex=True).str.strip()

        stat_name = url.split("/")[-1].replace("opponent-", "").replace("-", "_")
        df = df.rename(columns={col: f"{stat_name}_{col}" for col in df.columns if col != "Team"})

        print(f"✅ Acquired: {url}")
        return df

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return pd.DataFrame()

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
        "https://www.teamrankings.com/nba/stat/opponent-three-pointers-made-per-game"
    ]

    combined = None
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_teamrankings_stat, url): url for url in urls}
        for future in as_completed(futures):
            df = future.result()
            if df.empty or "Team" not in df.columns:
                print(f"⚠️ Skipping merge for {futures[future]}")
                continue

            if combined is None:
                combined = df
            else:
                combined = pd.merge(combined, df, on="Team", how="outer")

    if combined is None:
        print("❌ No valid data collected from any URLs.")
        return pd.DataFrame()

    print("\n🎉 All defensive stats successfully scraped and merged!")
    return combined

if __name__ == "__main__":
    defensive_stats_df = scrape_all_defensive_stats()
    if not defensive_stats_df.empty:
        defensive_stats_df.to_csv("scraped_defensive_stats.csv", index=False)
        print("\n✅ Defensive stats saved to scraped_defensive_stats.csv")
    else:
        print("⚠️ No data was saved.")

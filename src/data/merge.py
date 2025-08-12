
import pandas as pd

# Load datasets
player_df = pd.read_csv("/Users/raghavraghunath/SportsParlayMLIdea/nba_full_dataset.csv")
defense_df = pd.read_csv("/Users/raghavraghunath/SportsParlayMLIdea/scraped_defensive_stats.csv")

# Mapping from team abbreviation to city/team name (uppercased to match formats)
team_map = {
    "ATL": "ATLANTA", "BOS": "BOSTON", "BRK": "BROOKLYN", "CHO": "CHARLOTTE",
    "CHI": "CHICAGO", "CLE": "CLEVELAND", "DAL": "DALLAS", "DEN": "DENVER",
    "DET": "DETROIT", "GSW": "GOLDEN STATE", "HOU": "HOUSTON", "IND": "INDIANA",
    "LAC": "LA CLIPPERS", "LAL": "LOS ANGELES LAKERS", "MEM": "MEMPHIS", "MIA": "MIAMI",
    "MIL": "MILWAUKEE", "MIN": "MINNESOTA", "NOP": "NEW ORLEANS", "NYK": "NEW YORK",
    "OKC": "OKLAHOMA CITY", "ORL": "ORLANDO", "PHI": "PHILADELPHIA", "PHO": "PHOENIX",
    "POR": "PORTLAND", "SAC": "SACRAMENTO", "SAS": "SAN ANTONIO", "TOR": "TORONTO",
    "UTA": "UTAH", "WAS": "WASHINGTON"
}

# Prepare player dataframe
player_df["opponent_full_name"] = player_df["opponent"].map(team_map)
player_df["opponent_full_name"] = player_df["opponent_full_name"].str.upper()
defense_df["Team"] = defense_df["Team"].str.upper()

# Merge
merged_df = pd.merge(player_df, defense_df, left_on="opponent_full_name", right_on="Team", how="left")

# Save to new file
output_path = "/Users/raghavraghunath/SportsParlayMLIdea/nba_merged_with_defense_final.csv"
merged_df.to_csv(output_path, index=False)

print(f"✅ Merged dataset with defensive stats saved to: {output_path}")


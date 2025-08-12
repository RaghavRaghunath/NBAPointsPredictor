import pandas as pd
import os
import re
from glob import glob

def is_final_range_file(filename):
    """Check if the filename is a final range file (e.g., nba_defense_by_position_0_to_1797.csv)"""
    pattern = r'nba_defense_by_position_\d+_to_\d+\.csv$'
    return bool(re.match(pattern, filename))

def merge_defense_files():
    # Get all defense files
    all_files = sorted(glob('nba_defense_by_position_*.csv'))
    
    # Filter to only include final range files (exclude intermediate saves and latest)
    defense_files = [f for f in all_files if is_final_range_file(f) and 'latest' not in f.lower()]
    
    # Extract ranges and find the most comprehensive files
    file_ranges = {}
    for f in defense_files:
        try:
            # Extract start and end numbers from filename
            start_end = re.findall(r'\d+', f)
            if len(start_end) >= 2:
                start, end = map(int, start_end[:2])
                # Keep the file with the widest range for each starting point
                if start not in file_ranges or (end - start) > (file_ranges[start][1] - start):
                    file_ranges[start] = (end, f)
        except Exception as e:
            print(f"  Warning: Could not process filename {f}: {e}")
    
    # Get the final list of files to process
    final_files = sorted([v[1] for v in file_ranges.values()])
    
    if not final_files:
        print("No valid defense files found!")
        if all_files:
            print("Available files that were skipped:", all_files)
        return
    
    print(f"Found {len(final_files)} final range files to merge:")
    for f in final_files:
        print(f"  - {f}")
    
    # Initialize merged dataframe
    merged_df = pd.DataFrame()
    
    # Process each file
    for file in final_files:
        print(f"\nProcessing {file}...")
        try:
            # Read the file
            df = pd.read_csv(file)
            
            # Ensure required columns exist
            required_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'POSITION_GROUP', 'PTS']
            if not all(col in df.columns for col in required_cols):
                print(f"  Skipping {file} - missing required columns")
                print(f"  Available columns: {df.columns.tolist()}")
                continue
                
            # If we have rolling columns, keep them
            rolling_cols = [col for col in df.columns if 'ROLLING_' in col or '_DIFF' in col]
            
            # Keep only the columns we need
            cols_to_keep = required_cols + rolling_cols
            df = df[cols_to_keep]
            
            # Convert GAME_ID to string to avoid type mismatches
            df['GAME_ID'] = df['GAME_ID'].astype(str)
            
            # Add to merged dataframe
            if merged_df.empty:
                merged_df = df
                print(f"  Added initial {len(df)} records")
            else:
                # Get the count before merge
                before_count = len(merged_df)
                
                # Remove any existing rows that match the current file's games
                merged_df = merged_df[~merged_df['GAME_ID'].isin(df['GAME_ID'])]
                
                # Add the new data
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                
                print(f"  Added {len(df)} records, replaced {before_count - len(merged_df) + len(df)} total")
            
        except Exception as e:
            print(f"  Error processing {file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Sort by game ID and team
    if not merged_df.empty:
        merged_df = merged_df.sort_values(['GAME_ID', 'TEAM_ABBREVIATION', 'POSITION_GROUP'])
        print(f"\nFinal merged dataset contains {len(merged_df)} records")
        
        # Save the merged file
        output_file = 'merged_defense_stats.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"\nMerged data saved to {output_file} ({len(merged_df)} records)")
        
        # Also save as the latest file
        latest_file = 'nba_defense_by_position_latest.csv'
        merged_df.to_csv(latest_file, index=False)
        print(f"Latest defense file updated: {latest_file}")
        
        # Print summary
        print("\nSummary by position:")
        print(merged_df['POSITION_GROUP'].value_counts())
        
        # Check for rolling columns
        rolling_cols = [col for col in merged_df.columns if 'ROLLING_' in col or '_DIFF' in col]
        if rolling_cols:
            print("\nRolling stats columns found:", rolling_cols)
        else:
            print("\nNo rolling stats columns found. You may need to calculate them.")
    else:
        print("No valid data to merge!")

if __name__ == "__main__":
    merge_defense_files()

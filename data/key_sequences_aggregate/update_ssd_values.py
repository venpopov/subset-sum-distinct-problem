#!/usr/bin/env python3
"""
Script to find pairs of CSV files where one is an extended version of another,
and update the SSD column values in the longer file with known values from the shorter file.
"""

import os
import csv
import re
from pathlib import Path

def extract_seed_pattern(filename):
    """Extract the seed sequence from filename (everything between 'seed_' and '_n')."""
    match = re.match(r'seed_(.+)_n\d+\.csv', filename)
    return match.group(1) if match else None

def get_n_value(filename):
    """Extract the n value from filename."""
    match = re.search(r'_n(\d+)\.csv', filename)
    return int(match.group(1)) if match else None

def read_csv_data(filepath):
    """Read CSV data and return header and rows."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

def write_csv_data(filepath, header, rows):
    """Write CSV data to file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def rows_match_except_ssd(row1, row2):
    """Check if two rows match in all columns except the first (SSD) column."""
    if len(row1) != len(row2):
        return False
    return row1[1:] == row2[1:]

def find_and_update_pairs():
    """Find file pairs and update SSD values in extended files."""
    # Get all CSV files with their metadata
    csv_files = []
    for filename in os.listdir('.'):
        if filename.endswith('.csv'):
            seed_pattern = extract_seed_pattern(filename)
            if seed_pattern:
                n_value = get_n_value(filename)
                csv_files.append((filename, seed_pattern, n_value))
    
    # Sort by n value to check shorter files first
    csv_files.sort(key=lambda x: x[2])
    
    # Find pairs where one file's seed pattern is a prefix of another
    pairs_found = []
    pairs_updated = []
    
    # Check each pair of files
    for i in range(len(csv_files)):
        for j in range(i + 1, len(csv_files)):
            shorter_file, shorter_seed, shorter_n = csv_files[i]
            longer_file, longer_seed, longer_n = csv_files[j]
            
            # Check if the shorter file's seed is a prefix of the longer file's seed
            if longer_seed.startswith(shorter_seed + '_') or longer_seed == shorter_seed:
                # Read both files
                try:
                    shorter_header, shorter_rows = read_csv_data(shorter_file)
                    longer_header, longer_rows = read_csv_data(longer_file)
                except:
                    continue
                
                # Verify headers match
                if shorter_header != longer_header:
                    continue
                
                # Verify that the first len(shorter_rows) rows match (except SSD column)
                if len(longer_rows) < len(shorter_rows):
                    continue
                
                all_match = True
                for k in range(len(shorter_rows)):
                    if not rows_match_except_ssd(shorter_rows[k], longer_rows[k]):
                        all_match = False
                        break
                
                if all_match:
                    pairs_found.append((shorter_file, longer_file))
                    print(f"Found pair: {shorter_file} -> {longer_file}")
                    
                    # Check if longer file has -1 values that need updating
                    has_unknown = any(row[0] == '-1' for row in longer_rows[:len(shorter_rows)])
                    
                    if has_unknown:
                        # Update the longer file with SSD values from shorter file
                        for k in range(len(shorter_rows)):
                            longer_rows[k][0] = shorter_rows[k][0]
                        
                        # Write the updated file
                        write_csv_data(longer_file, longer_header, longer_rows)
                        pairs_updated.append((shorter_file, longer_file))
                        print(f"  ✓ Updated {len(shorter_rows)} rows in {longer_file}")
    
    return pairs_found, pairs_updated

if __name__ == '__main__':
    print("Searching for file pairs and updating SSD values...\n")
    pairs_found, pairs_updated = find_and_update_pairs()
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total pairs found: {len(pairs_found)}")
    print(f"  Pairs updated: {len(pairs_updated)}")
    print(f"{'='*60}")

#!/usr/bin/env python3
"""
Parse IPUMS CPS Data - Version 2
================================

Analyzes the file structure to determine variable positions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"
cps_path = Path("/Users/amalkova/Downloads/IPUMS CPS Data 00003.dat")

print("=" * 80)
print("IPUMS CPS DATA STRUCTURE ANALYSIS")
print("=" * 80)

# Read first few lines to analyze
with open(cps_path, 'r') as f:
    lines = [f.readline() for _ in range(10)]

print(f"\nLine length: {len(lines[0])}")
print("\nFirst 5 records:")
for i, line in enumerate(lines[:5]):
    print(f"{i}: {line.strip()}")

# Analyze the structure
print("\n" + "-" * 80)
print("Position Analysis")
print("-" * 80)

line = lines[0].strip()
print(f"\nFull line ({len(line)} chars):")
print(line)

# Try to identify patterns
# Positions 1-4: likely year
print(f"\nPos 1-4: {line[0:4]} (likely year)")
print(f"Pos 5-10: {line[4:10]}")
print(f"Pos 11-20: {line[10:20]}")

# Look at multiple lines to find varying columns
print("\n" + "-" * 80)
print("Comparing across records")
print("-" * 80)

# Check which positions vary
for pos in range(0, min(100, len(line)), 5):
    values = [l[pos:pos+5].strip() for l in lines if len(l) > pos+5]
    unique = set(values)
    if len(unique) > 1:
        print(f"Pos {pos+1}-{pos+5}: {unique}")

# Try a simple parsing approach based on common IPUMS CPS structure
# Standard IPUMS CPS ASEC format often has:
# - Year in first 4 positions
# - Various household/person IDs
# - Demographics starting around position 15-30

print("\n" + "-" * 80)
print("Attempting to parse with IPUMS standard positions")
print("-" * 80)

# Common IPUMS CPS positions (may need adjustment)
# This is a simplified version - actual positions depend on the specific extract
try:
    # Try reading with minimal variables to see what we get
    # Year is clearly 1-4
    # Let's try to find age and sex by looking at reasonable values

    # Read a larger sample
    sample_data = []
    with open(cps_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            sample_data.append(line.strip())

    print(f"Read {len(sample_data)} records")

    # Analyze each position to find age (should be 0-99) and sex (should be 1 or 2)
    line_len = len(sample_data[0])

    # Find positions with values that look like ages (0-99)
    print("\nSearching for age-like values (0-99)...")
    age_candidates = []
    for pos in range(4, min(50, line_len-2)):
        values = []
        for line in sample_data[:1000]:
            try:
                val = int(line[pos:pos+2])
                if 0 <= val <= 99:
                    values.append(val)
            except:
                pass
        if len(values) > 900:  # Most should be valid
            avg = np.mean(values)
            if 20 < avg < 60:  # Reasonable average age
                age_candidates.append((pos, avg, len(values)))

    print(f"Age candidates: {age_candidates[:5]}")

    # Find positions with values that look like sex (1 or 2)
    print("\nSearching for sex-like values (1 or 2)...")
    sex_candidates = []
    for pos in range(4, min(50, line_len-1)):
        values = []
        for line in sample_data[:1000]:
            try:
                val = int(line[pos:pos+1])
                if val in [1, 2]:
                    values.append(val)
            except:
                pass
        if len(values) > 900:
            pct_female = sum(1 for v in values if v == 2) / len(values)
            if 0.4 < pct_female < 0.6:  # Should be roughly 50/50
                sex_candidates.append((pos, pct_female, len(values)))

    print(f"Sex candidates: {sex_candidates[:5]}")

    # Try to find income (larger numbers)
    print("\nSearching for income-like values...")
    income_candidates = []
    for pos in range(50, min(150, line_len-8)):
        values = []
        for line in sample_data[:1000]:
            try:
                val = int(line[pos:pos+8])
                if 0 < val < 9999999:
                    values.append(val)
            except:
                pass
        if len(values) > 500:
            avg = np.mean(values)
            if 10000 < avg < 200000:  # Reasonable income range
                income_candidates.append((pos, avg, len(values)))

    print(f"Income candidates: {income_candidates[:5]}")

    # If we found candidates, try to use them
    if age_candidates and sex_candidates:
        age_pos = age_candidates[0][0]
        sex_pos = sex_candidates[0][0]

        print(f"\n" + "-" * 80)
        print(f"Using age position: {age_pos+1}-{age_pos+2}")
        print(f"Using sex position: {sex_pos+1}")
        print("-" * 80)

        # Parse full file with identified positions
        print("\nParsing full file...")

        data = []
        with open(cps_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0:
                    print(f"  Processed {i:,} records...")
                line = line.strip()
                try:
                    year = int(line[0:4])
                    age = int(line[age_pos:age_pos+2])
                    sex = int(line[sex_pos:sex_pos+1])

                    # Try to get income if we found a candidate
                    income = None
                    if income_candidates:
                        inc_pos = income_candidates[0][0]
                        try:
                            income = int(line[inc_pos:inc_pos+8])
                            if income < 0 or income > 9999990:
                                income = None
                        except:
                            pass

                    data.append({
                        'year': year,
                        'age': age,
                        'sex': sex,
                        'income': income
                    })
                except:
                    pass

        df = pd.DataFrame(data)
        print(f"\nParsed {len(df):,} records")
        print(f"Year range: {df['year'].min()}-{df['year'].max()}")
        print(f"Age range: {df['age'].min()}-{df['age'].max()}")
        print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")

        # Filter to analysis sample
        df['birth_year'] = df['year'] - df['age']
        sample = df[
            (df['sex'] == 2) &
            (df['birth_year'] >= 1957) &
            (df['birth_year'] <= 1964) &
            (df['age'] >= 35) &
            (df['age'] <= 50)
        ].copy()

        print(f"\nWomen born 1957-1964, ages 35-50: {len(sample):,}")

        if len(sample) > 0:
            print(f"Year range: {sample['year'].min()}-{sample['year'].max()}")

            # Save for further analysis
            sample.to_csv(OUTPUT_DIR / "cps_ipums_sample.csv", index=False)
            print(f"Saved: cps_ipums_sample.csv")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

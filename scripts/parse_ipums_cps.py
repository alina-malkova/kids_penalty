#!/usr/bin/env python3
"""
Parse IPUMS CPS Data to Fill Age 35-50 Gap
==========================================

Uses the large IPUMS CPS fixed-width file with variable positions from cps_00001.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

print("=" * 80)
print("IPUMS CPS DATA PROCESSING")
print("Filling the Age 35-50 Gap for 1957-1964 Birth Cohort")
print("=" * 80)

# ============================================================================
# DEFINE VARIABLE POSITIONS FROM cps_00001.txt
# ============================================================================

# Column positions from the Stata do-file (1-indexed, convert to 0-indexed)
# Format: (start-1, end) for pandas read_fwf
colspecs = [
    (0, 4),       # year: 1-4
    (219, 221),   # age: 220-221
    (221, 222),   # sex: 222-222
    (222, 225),   # race: 223-225
    (252, 253),   # nchild: 253-253
    (396, 399),   # educ: 397-399
    (497, 506),   # inctot: 498-506
    (506, 514),   # incwage: 507-514
    (204, 215),   # asecwt: 205-215
    (303, 306),   # hispan: 304-306
]

names = ['year', 'age', 'sex', 'race', 'nchild', 'educ', 'inctot', 'incwage', 'asecwt', 'hispan']

# ============================================================================
# READ IPUMS CPS DATA
# ============================================================================

cps_path = Path("/Users/amalkova/Downloads/IPUMS CPS Data 00003.dat")

if not cps_path.exists():
    print(f"ERROR: File not found: {cps_path}")
    exit(1)

print(f"\nReading {cps_path}...")
print("This may take a few minutes for the 1.9 GB file...")

# Read in chunks to handle large file
chunk_size = 500000
chunks = []

try:
    # First, check the file structure with a small sample
    print("\nChecking file structure with sample...")
    sample = pd.read_fwf(cps_path, colspecs=colspecs, names=names, nrows=1000)
    print(f"Sample loaded: {len(sample)} rows")
    print(f"Year range in sample: {sample['year'].min()} - {sample['year'].max()}")
    print(f"Age range in sample: {sample['age'].min()} - {sample['age'].max()}")
    print(f"\nSample data:")
    print(sample.head(10))

    # Check if the parsing looks correct
    if sample['year'].min() < 1960 or sample['year'].max() > 2030:
        print("\nWARNING: Year values look incorrect. Checking raw data...")
        with open(cps_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"Line {i}: {line[:100]}...")
                else:
                    break

    # If year looks correct, proceed with full read
    if 1980 <= sample['year'].min() <= 2025:
        print(f"\nProceeding with full data read...")

        # Read full file
        cps = pd.read_fwf(cps_path, colspecs=colspecs, names=names)
        print(f"Loaded {len(cps):,} observations")

        # ============================================================================
        # FILTER TO ANALYSIS SAMPLE
        # ============================================================================

        print("\n" + "-" * 80)
        print("Filtering to Analysis Sample")
        print("-" * 80)

        # Calculate birth year
        cps['birth_year'] = cps['year'] - cps['age']

        # Filter to women (sex=2), birth cohort 1957-1964, ages 35-50
        sample_mask = (
            (cps['sex'] == 2) &
            (cps['birth_year'] >= 1957) &
            (cps['birth_year'] <= 1964) &
            (cps['age'] >= 35) &
            (cps['age'] <= 50)
        )

        cps_sample = cps[sample_mask].copy()
        print(f"Women born 1957-1964, ages 35-50: {len(cps_sample):,}")

        if len(cps_sample) > 0:
            print(f"Year range: {cps_sample['year'].min()}-{cps_sample['year'].max()}")
            print(f"Age range: {cps_sample['age'].min()}-{cps_sample['age'].max()}")

            # ============================================================================
            # HARMONIZE VARIABLES
            # ============================================================================

            print("\n" + "-" * 80)
            print("Harmonizing Variables")
            print("-" * 80)

            cps_harm = pd.DataFrame()
            cps_harm['year'] = cps_sample['year'].values
            cps_harm['age'] = cps_sample['age'].values
            cps_harm['birth_year'] = cps_sample['birth_year'].values
            cps_harm['female'] = 1

            # Motherhood (nchild > 0)
            cps_harm['nchild'] = cps_sample['nchild'].values
            cps_harm['mother'] = (cps_harm['nchild'] > 0).astype(int)

            # Income
            cps_harm['income'] = pd.to_numeric(cps_sample['inctot'], errors='coerce')
            # Handle negative values (missing codes)
            cps_harm.loc[cps_harm['income'] < 0, 'income'] = np.nan
            # Handle top codes (9999999 etc)
            cps_harm.loc[cps_harm['income'] > 9999990, 'income'] = np.nan

            # Education (harmonized)
            educ = cps_sample['educ'].values
            cps_harm['education'] = np.where(educ >= 111, 4,  # BA+
                                    np.where(educ >= 91, 3,   # Some college
                                    np.where(educ >= 73, 2,   # HS
                                    1)))  # Less than HS

            # Race (harmonized: 1=White, 2=Black, 3=Hispanic/Other)
            race = cps_sample['race'].values
            hispan = cps_sample['hispan'].values
            cps_harm['race'] = np.where(hispan > 0, 3,        # Hispanic
                               np.where(race == 200, 2,       # Black
                               np.where(race == 100, 1,       # White
                               3)))  # Other

            # Weight
            cps_harm['weight'] = pd.to_numeric(cps_sample['asecwt'], errors='coerce') / 10000

            # Source
            cps_harm['source'] = 'CPS'

            print(f"Harmonized data: {len(cps_harm):,} observations")

            # ============================================================================
            # MOTHERHOOD PENALTY ANALYSIS
            # ============================================================================

            print("\n" + "-" * 80)
            print("MOTHERHOOD PENALTY ANALYSIS (Ages 35-50)")
            print("-" * 80)

            # Filter to positive income
            analysis = cps_harm[
                (cps_harm['income'].notna()) &
                (cps_harm['income'] > 0)
            ].copy()

            print(f"Analysis sample with positive income: {len(analysis):,}")

            mothers = analysis[analysis['mother'] == 1]
            childless = analysis[analysis['mother'] == 0]

            print(f"\nMotherhood status:")
            print(f"  Mothers: {len(mothers):,} ({len(mothers)/len(analysis)*100:.1f}%)")
            print(f"  Childless: {len(childless):,} ({len(childless)/len(analysis)*100:.1f}%)")

            if len(childless) > 0:
                print(f"\nOVERALL:")
                print(f"  Mothers: mean=${mothers['income'].mean():,.0f}, median=${mothers['income'].median():,.0f}")
                print(f"  Childless: mean=${childless['income'].mean():,.0f}, median=${childless['income'].median():,.0f}")

                penalty = (childless['income'].mean() - mothers['income'].mean()) / childless['income'].mean() * 100
                print(f"  Motherhood Penalty: {penalty:+.1f}%")

                # By 5-year age bins
                print("\nBy Age Group:")
                results = []
                for age_min in [35, 40, 45]:
                    age_max = age_min + 5
                    subset = analysis[(analysis['age'] >= age_min) & (analysis['age'] < age_max)]
                    m = subset[subset['mother'] == 1]
                    c = subset[subset['mother'] == 0]

                    if len(m) >= 50 and len(c) >= 10:
                        m_inc = m['income'].mean()
                        c_inc = c['income'].mean()
                        pen = (c_inc - m_inc) / c_inc * 100

                        results.append({
                            'age_group': f"{age_min}-{age_max}",
                            'n_mothers': len(m),
                            'n_childless': len(c),
                            'mothers_income': m_inc,
                            'childless_income': c_inc,
                            'penalty_pct': pen,
                            'source': 'CPS'
                        })

                        print(f"  {age_min}-{age_max}: Mothers=${m_inc:,.0f} (n={len(m)}), Childless=${c_inc:,.0f} (n={len(c)}), Penalty={pen:+.1f}%")
                    else:
                        print(f"  {age_min}-{age_max}: Insufficient data (m={len(m)}, c={len(c)})")

                # Save results
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(OUTPUT_DIR / "cps_penalty_ages_35_50.csv", index=False)
                    print(f"\nSaved: cps_penalty_ages_35_50.csv")

                # Save harmonized CPS data
                cps_harm.to_csv(OUTPUT_DIR / "cps_harmonized_35_50.csv", index=False)
                print(f"Saved: cps_harmonized_35_50.csv ({len(cps_harm):,} rows)")

                # ============================================================================
                # COMBINE WITH NLSY79 + HRS
                # ============================================================================

                print("\n" + "-" * 80)
                print("COMBINING WITH NLSY79 + HRS DATA")
                print("-" * 80)

                combined_path = OUTPUT_DIR / "combined_lifecycle_panel.csv"
                if combined_path.exists():
                    existing = pd.read_csv(combined_path)
                    print(f"Loaded existing combined panel: {len(existing):,} observations")

                    # Prepare CPS for merge
                    cps_merge = analysis[['year', 'age', 'female', 'mother', 'income']].copy()
                    cps_merge['source'] = 'CPS'
                    cps_merge['id'] = range(2000000, 2000000 + len(cps_merge))

                    # Add missing columns
                    for col in existing.columns:
                        if col not in cps_merge.columns:
                            cps_merge[col] = np.nan

                    cps_merge = cps_merge[existing.columns]

                    # Combine
                    full = pd.concat([existing, cps_merge], ignore_index=True)
                    print(f"Combined panel: {len(full):,} observations")

                    # Save
                    full.to_csv(OUTPUT_DIR / "combined_lifecycle_with_cps.csv", index=False)
                    print(f"Saved: combined_lifecycle_with_cps.csv")

                    # ============================================================================
                    # FULL LIFECYCLE ANALYSIS
                    # ============================================================================

                    print("\n" + "-" * 80)
                    print("FULL LIFECYCLE MOTHERHOOD PENALTY")
                    print("-" * 80)

                    women = full[full['female'] == 1].copy()
                    women = women[women['income'].notna() & (women['income'] > 0)]
                    women = women[women['mother'].notna()]

                    print(f"\n{'Age':<10} {'Penalty':>10} {'N Mothers':>12} {'N Childless':>12} {'Source':>10}")
                    print("-" * 60)

                    final_results = []
                    for age_min in [20, 25, 30, 35, 40, 45, 50, 55, 60]:
                        age_max = age_min + 5
                        subset = women[(women['age'] >= age_min) & (women['age'] < age_max)]
                        m = subset[subset['mother'] == 1]
                        c = subset[subset['mother'] == 0]

                        if len(m) >= 50 and len(c) >= 10:
                            m_inc = m['income'].mean()
                            c_inc = c['income'].mean()
                            pen = (c_inc - m_inc) / c_inc * 100
                            src = subset['source'].mode().iloc[0] if len(subset) > 0 else 'mixed'

                            final_results.append({
                                'age_group': f"{age_min}-{age_max}",
                                'penalty_pct': pen,
                                'n_mothers': len(m),
                                'n_childless': len(c),
                                'source': src
                            })

                            print(f"{age_min}-{age_max:<5} {pen:>+9.1f}% {len(m):>12,} {len(c):>12,} {src:>10}")
                        else:
                            print(f"{age_min}-{age_max:<5} {'(gap)':>10} {len(m):>12,} {len(c):>12,} {'---':>10}")

                    if final_results:
                        final_df = pd.DataFrame(final_results)
                        final_df.to_csv(OUTPUT_DIR / "lifecycle_penalty_complete.csv", index=False)
                        print(f"\nSaved: lifecycle_penalty_complete.csv")

    else:
        print(f"\nERROR: Year parsing seems incorrect (got {sample['year'].min()}-{sample['year'].max()})")
        print("The IPUMS file may have a different format than cps_00001.txt")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)

#!/usr/bin/env python3
"""
Parse IPUMS CPS ASEC Data - Extract 00003
=========================================

Variable positions from codebook:
- YEAR: 1-4
- MONTH: 10-11
- HHINCOME: 49-56
- PERNUM: 57-58
- ASECWT: 102-112
- AGE: 113-114
- SEX: 115
- RACE: 116-118
- NCHILD: 119
- HISPAN: 120-122
- EDUC: 123-125
- INCTOT: 126-134
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

print("=" * 80)
print("IPUMS CPS ASEC DATA PROCESSING")
print("Extract 00003 - Years 1990-2025")
print("=" * 80)

# File path
cps_path = Path("/Users/amalkova/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology/KIDS Penalty/IPUMS CPS Data 00003.dat.gz")

if not cps_path.exists():
    # Try alternative location
    cps_path = Path("/Users/amalkova/Downloads/IPUMS CPS Data 00003.dat.gz")

print(f"\nReading: {cps_path}")

# Variable positions (0-indexed for Python)
# From codebook: columns are 1-indexed, so subtract 1 for start
colspecs = [
    (0, 4),       # YEAR: 1-4
    (9, 11),      # MONTH: 10-11
    (48, 56),     # HHINCOME: 49-56
    (56, 58),     # PERNUM: 57-58
    (101, 112),   # ASECWT: 102-112
    (112, 114),   # AGE: 113-114
    (114, 115),   # SEX: 115
    (115, 118),   # RACE: 116-118
    (118, 119),   # NCHILD: 119
    (119, 122),   # HISPAN: 120-122
    (122, 125),   # EDUC: 123-125
    (125, 134),   # INCTOT: 126-134
]

names = ['year', 'month', 'hhincome', 'pernum', 'asecwt', 'age', 'sex', 'race', 'nchild', 'hispan', 'educ', 'inctot']

# Read the gzipped file
print("\nLoading data (this may take a minute)...")

try:
    with gzip.open(cps_path, 'rt') as f:
        cps = pd.read_fwf(f, colspecs=colspecs, names=names)

    print(f"Loaded {len(cps):,} observations")
    print(f"\nYear range: {cps['year'].min()} - {cps['year'].max()}")
    print(f"Age range: {cps['age'].min()} - {cps['age'].max()}")

    # Check data quality
    print("\nData quality check:")
    print(f"  Sex distribution: {cps['sex'].value_counts().to_dict()}")
    print(f"  Sample years: {sorted(cps['year'].unique())[:10]}...")

    # ============================================================================
    # FILTER TO ANALYSIS SAMPLE
    # ============================================================================

    print("\n" + "-" * 80)
    print("FILTERING TO 1957-1964 BIRTH COHORT, AGES 35-50")
    print("-" * 80)

    # Calculate birth year
    cps['birth_year'] = cps['year'] - cps['age']

    # Filter to:
    # - Women (sex=2)
    # - Birth cohort 1957-1964
    # - Ages 35-50
    # - ASEC months (March = 3) for income data

    sample = cps[
        (cps['sex'] == 2) &
        (cps['birth_year'] >= 1957) &
        (cps['birth_year'] <= 1964) &
        (cps['age'] >= 35) &
        (cps['age'] <= 50) &
        (cps['month'] == 3)  # ASEC is March supplement
    ].copy()

    print(f"Women born 1957-1964, ages 35-50 (March ASEC): {len(sample):,}")

    if len(sample) > 0:
        print(f"Year range: {sample['year'].min()}-{sample['year'].max()}")
        print(f"Age range: {sample['age'].min()}-{sample['age'].max()}")

        # ============================================================================
        # CREATE HARMONIZED VARIABLES
        # ============================================================================

        print("\n" + "-" * 80)
        print("HARMONIZING VARIABLES")
        print("-" * 80)

        cps_harm = pd.DataFrame()
        cps_harm['year'] = sample['year'].values
        cps_harm['age'] = sample['age'].values
        cps_harm['birth_year'] = sample['birth_year'].values
        cps_harm['female'] = 1

        # Motherhood (nchild > 0 = children in household)
        cps_harm['nchild'] = sample['nchild'].values
        cps_harm['mother'] = (cps_harm['nchild'] > 0).astype(int)

        # Income (INCTOT = total personal income)
        cps_harm['income'] = pd.to_numeric(sample['inctot'], errors='coerce')
        # Handle missing/invalid values
        cps_harm.loc[cps_harm['income'] == 99999999, 'income'] = np.nan
        cps_harm.loc[cps_harm['income'] < 0, 'income'] = np.nan

        # Education (harmonized: 1=<HS, 2=HS, 3=Some college, 4=BA+)
        educ = sample['educ'].values
        cps_harm['education'] = np.where(educ >= 111, 4,      # BA+
                                np.where(educ >= 80, 3,       # Some college/Associate
                                np.where(educ >= 73, 2,       # HS
                                1)))  # Less than HS

        # Race (harmonized: 1=White, 2=Black, 3=Hispanic/Other)
        race = sample['race'].values
        hispan = sample['hispan'].values
        cps_harm['race'] = np.where(hispan > 0, 3,            # Hispanic
                           np.where(race == 200, 2,           # Black
                           np.where(race == 100, 1,           # White
                           3)))  # Other

        # Weight (ASECWT needs to be divided by 10000 per IPUMS)
        cps_harm['weight'] = pd.to_numeric(sample['asecwt'], errors='coerce') / 10000

        # Source
        cps_harm['source'] = 'CPS'

        print(f"Harmonized data: {len(cps_harm):,} observations")

        # Motherhood status
        n_mothers = (cps_harm['mother'] == 1).sum()
        n_childless = (cps_harm['mother'] == 0).sum()
        print(f"\nMotherhood status:")
        print(f"  Mothers: {n_mothers:,} ({n_mothers/len(cps_harm)*100:.1f}%)")
        print(f"  Childless: {n_childless:,} ({n_childless/len(cps_harm)*100:.1f}%)")

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

        print(f"Sample with positive income: {len(analysis):,}")

        mothers = analysis[analysis['mother'] == 1]
        childless = analysis[analysis['mother'] == 0]

        print(f"\n{'='*60}")
        print("OVERALL (Ages 35-50):")
        print(f"  Mothers (n={len(mothers):,}): mean=${mothers['income'].mean():,.0f}, median=${mothers['income'].median():,.0f}")
        print(f"  Childless (n={len(childless):,}): mean=${childless['income'].mean():,.0f}, median=${childless['income'].median():,.0f}")

        if len(childless) > 0:
            penalty = (childless['income'].mean() - mothers['income'].mean()) / childless['income'].mean() * 100
            print(f"  Motherhood Penalty: {penalty:+.1f}%")
        print(f"{'='*60}")

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

                print(f"  {age_min}-{age_max}: Mothers=${m_inc:,.0f} (n={len(m):,}), Childless=${c_inc:,.0f} (n={len(c):,}), Penalty={pen:+.1f}%")
            else:
                print(f"  {age_min}-{age_max}: Insufficient data (mothers={len(m)}, childless={len(c)})")

        # ============================================================================
        # SAVE RESULTS
        # ============================================================================

        print("\n" + "-" * 80)
        print("SAVING RESULTS")
        print("-" * 80)

        # Save CPS results
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(OUTPUT_DIR / "cps_penalty_ages_35_50.csv", index=False)
            print(f"Saved: cps_penalty_ages_35_50.csv")

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
            cps_merge['id'] = range(3000000, 3000000 + len(cps_merge))

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

            print("\n" + "=" * 80)
            print("FULL LIFECYCLE MOTHERHOOD PENALTY (NLSY79 + CPS + HRS)")
            print("=" * 80)

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
                        'mothers_income': m_inc,
                        'childless_income': c_inc,
                        'source': src
                    })

                    print(f"{age_min}-{age_max:<5} {pen:>+9.1f}% {len(m):>12,} {len(c):>12,} {src:>10}")
                else:
                    print(f"{age_min}-{age_max:<5} {'(gap)':>10} {len(m):>12,} {len(c):>12,} {'---':>10}")

            if final_results:
                final_df = pd.DataFrame(final_results)
                final_df.to_csv(OUTPUT_DIR / "lifecycle_penalty_complete.csv", index=False)
                print(f"\nSaved: lifecycle_penalty_complete.csv")

                print("\n" + "=" * 80)
                print("KEY FINDING: AGE 35-50 GAP NOW FILLED WITH CPS DATA")
                print("=" * 80)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)

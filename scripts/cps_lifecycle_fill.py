#!/usr/bin/env python3
"""
CPS Lifecycle Gap Analysis
==========================

Uses Current Population Survey (CPS) Annual Social and Economic Supplement (ASEC)
to fill the age 35-50 gap in the NLSY79 + HRS lifecycle analysis.

For the 1957-1964 birth cohort:
- Age 35-50 corresponds to years 1992-2014
- NLSY79 income ends ~1993 (age ~33)
- HRS starts at age 50+ (~2007 for this cohort)

CPS ASEC provides annual income data that can fill this gap.

Data Source: IPUMS CPS (https://cps.ipums.org/)
Required Extract: cps_00001.dat with the following variables:
- year, age, sex, race, hispan
- nchild (number of own children in household)
- educ (education)
- inctot (total personal income)
- incwage (wage income)
- asecwt (ASEC weight)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import struct
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"
CPS_DIR = Path("/Users/amalkova/Downloads")

print("=" * 80)
print("CPS LIFECYCLE GAP ANALYSIS")
print("Filling the Age 35-50 Gap for 1957-1964 Birth Cohort")
print("=" * 80)

# ============================================================================
# CHECK FOR CPS DATA
# ============================================================================

cps_dat_path = CPS_DIR / "cps_00001.dat"
cps_do_path = CPS_DIR / "cps_00001.txt"

if not cps_dat_path.exists():
    print("\n" + "!" * 80)
    print("CPS DATA FILE NOT FOUND")
    print("!" * 80)
    print(f"""
The CPS data file was not found at: {cps_dat_path}

To download CPS data from IPUMS:

1. Go to https://cps.ipums.org/cps/
2. Create an account or log in
3. Select data -> Create extract
4. Choose samples: ASEC (March supplement) for years 1992-2014
5. Select variables:
   - YEAR, AGE, SEX, RACE, HISPAN (demographics)
   - NCHILD, ELDCH, YNGCH (fertility - children in household)
   - EDUC (education)
   - INCTOT (total personal income)
   - INCWAGE (wage income)
   - ASECWT (person weight for ASEC)
6. Submit extract and download the .dat file
7. Rename to cps_00001.dat and place in {CPS_DIR}

The extraction script (cps_00001.txt) is already present and will be used to
define the variable positions.

Alternatively, we can estimate the gap using the existing NLSY79 and HRS data
with interpolation, but this is less precise.
""")

    # Try alternative: create synthetic estimates from existing data
    print("\n" + "-" * 80)
    print("ALTERNATIVE: Synthetic Gap Estimation from NLSY79 + HRS")
    print("-" * 80)

    # Load existing combined data
    combined_path = OUTPUT_DIR / "combined_lifecycle_panel.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        print(f"Loaded combined panel: {len(combined):,} person-years")

        # Analyze available data
        women = combined[combined['female'] == 1].copy()
        print(f"Women: {len(women):,} person-years")

        # Check age distribution
        print("\nAge distribution by source:")
        age_dist = women.groupby(['source', pd.cut(women['age'], bins=[15,25,35,45,55,65,75])]).size()
        print(age_dist)

        # Calculate penalty by 5-year age groups
        print("\n" + "-" * 80)
        print("Motherhood Penalty by Age Group (existing data)")
        print("-" * 80)

        age_bins = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65)]
        results = []

        for (age_min, age_max) in age_bins:
            subset = women[(women['age'] >= age_min) & (women['age'] < age_max)].copy()
            subset = subset[subset['income'].notna() & (subset['income'] > 0)]
            subset = subset[subset['mother'].notna()]

            mothers = subset[subset['mother'] == 1]
            childless = subset[subset['mother'] == 0]

            if len(mothers) >= 100 and len(childless) >= 20:
                m_inc = mothers['income'].mean()
                c_inc = childless['income'].mean()
                penalty = (c_inc - m_inc) / c_inc * 100

                results.append({
                    'age_group': f"{age_min}-{age_max}",
                    'n_mothers': len(mothers),
                    'n_childless': len(childless),
                    'mothers_income': m_inc,
                    'childless_income': c_inc,
                    'penalty_pct': penalty,
                    'source': subset['source'].mode().iloc[0] if len(subset) > 0 else 'mixed'
                })

                print(f"\nAge {age_min}-{age_max}:")
                print(f"  Mothers (n={len(mothers):,}): ${m_inc:,.0f}")
                print(f"  Childless (n={len(childless):,}): ${c_inc:,.0f}")
                print(f"  Penalty: {penalty:+.1f}%")
            else:
                print(f"\nAge {age_min}-{age_max}: Insufficient data (mothers={len(mothers)}, childless={len(childless)})")
                results.append({
                    'age_group': f"{age_min}-{age_max}",
                    'n_mothers': len(mothers),
                    'n_childless': len(childless),
                    'mothers_income': np.nan,
                    'childless_income': np.nan,
                    'penalty_pct': np.nan,
                    'source': 'insufficient'
                })

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "lifecycle_penalty_5yr_bins.csv", index=False)
        print(f"\nSaved: lifecycle_penalty_5yr_bins.csv")

        # Interpolation for missing ages
        print("\n" + "-" * 80)
        print("Interpolated Estimates for Missing Ages")
        print("-" * 80)

        valid_results = results_df[results_df['penalty_pct'].notna()].copy()
        if len(valid_results) >= 2:
            # Simple linear interpolation
            valid_results['age_mid'] = valid_results['age_group'].apply(
                lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
            )

            from scipy import interpolate

            # Check if we have enough points for interpolation
            if len(valid_results) >= 2:
                f = interpolate.interp1d(
                    valid_results['age_mid'].values,
                    valid_results['penalty_pct'].values,
                    kind='linear',
                    fill_value='extrapolate'
                )

                # Estimate for gap ages
                gap_ages = [37.5, 42.5, 47.5]  # midpoints of 35-40, 40-45, 45-50
                print("\nInterpolated/extrapolated estimates:")
                for age in gap_ages:
                    est_penalty = float(f(age))
                    print(f"  Age {age:.0f}: {est_penalty:+.1f}% penalty (estimated)")

    else:
        print(f"Combined panel not found at {combined_path}")
        print("Run combined_lifecycle_analysis.py first.")

else:
    # ============================================================================
    # PROCESS CPS DATA
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 1: Processing CPS Data")
    print("-" * 80)

    # Parse the Stata do-file to get variable positions
    # Key variables from cps_00001.txt:
    # year: 1-4
    # age: 220-221
    # sex: 222-222
    # race: 223-225
    # hispan: 304-306
    # nchild: 253-253
    # educ: 397-399
    # inctot: 498-506
    # incwage: 507-514
    # asecwt: 205-215

    print("Reading CPS fixed-width data...")

    # Define column positions (0-indexed)
    colspecs = [
        (0, 4),      # year
        (219, 221),  # age
        (221, 222),  # sex
        (222, 225),  # race
        (303, 306),  # hispan
        (252, 253),  # nchild
        (396, 399),  # educ
        (497, 506),  # inctot
        (506, 514),  # incwage
        (204, 215),  # asecwt
    ]

    names = ['year', 'age', 'sex', 'race', 'hispan', 'nchild', 'educ', 'inctot', 'incwage', 'asecwt']

    # Read fixed-width file
    try:
        cps = pd.read_fwf(cps_dat_path, colspecs=colspecs, names=names)
        print(f"Loaded {len(cps):,} observations")
    except Exception as e:
        print(f"Error reading CPS data: {e}")
        print("The column positions may need adjustment based on the actual extract.")
        raise

    # ============================================================================
    # FILTER TO RELEVANT SAMPLE
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 2: Filtering to 1957-1964 Birth Cohort Women")
    print("-" * 80)

    # Calculate birth year
    cps['birth_year'] = cps['year'] - cps['age']

    # Filter to:
    # - Women (sex == 2)
    # - Birth cohort 1957-1964
    # - Ages 35-50
    cps_sample = cps[
        (cps['sex'] == 2) &
        (cps['birth_year'] >= 1957) &
        (cps['birth_year'] <= 1964) &
        (cps['age'] >= 35) &
        (cps['age'] <= 50)
    ].copy()

    print(f"Filtered sample: {len(cps_sample):,} woman-years")
    print(f"Years covered: {cps_sample['year'].min()}-{cps_sample['year'].max()}")
    print(f"Ages covered: {cps_sample['age'].min()}-{cps_sample['age'].max()}")

    # ============================================================================
    # CREATE HARMONIZED VARIABLES
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 3: Harmonizing Variables")
    print("-" * 80)

    cps_harmonized = pd.DataFrame()
    cps_harmonized['id'] = range(len(cps_sample))  # Synthetic IDs
    cps_harmonized['year'] = cps_sample['year'].values
    cps_harmonized['age'] = cps_sample['age'].values
    cps_harmonized['birth_year'] = cps_sample['birth_year'].values
    cps_harmonized['female'] = 1

    # Motherhood: nchild > 0 (own children in household)
    # NOTE: This undercounts mothers whose children have left home
    cps_harmonized['nchild'] = cps_sample['nchild'].values
    cps_harmonized['mother'] = (cps_harmonized['nchild'] > 0).astype(int)

    # Income (total personal income)
    cps_harmonized['income'] = cps_sample['inctot'].values
    cps_harmonized.loc[cps_harmonized['income'] < 0, 'income'] = np.nan

    # Education
    # EDUC codes: 0-72 = less than HS, 73 = HS, 81-90 = some college, 91+ = BA+
    educ = cps_sample['educ'].values
    cps_harmonized['education'] = np.where(educ >= 91, 4,  # College+
                                  np.where(educ >= 81, 3,   # Some college
                                  np.where(educ >= 73, 2,   # HS
                                  1)))  # Less than HS

    # Race (harmonized: 1=White, 2=Black, 3=Hispanic/Other)
    race = cps_sample['race'].values
    hispan = cps_sample['hispan'].values
    cps_harmonized['race'] = np.where(hispan > 0, 3,  # Hispanic
                             np.where(race == 200, 2,  # Black
                             np.where(race == 100, 1,  # White
                             3)))  # Other

    # Weight
    cps_harmonized['weight'] = cps_sample['asecwt'].values / 10000  # Adjust per do-file

    # Source
    cps_harmonized['source'] = 'CPS'

    print(f"Harmonized dataset: {len(cps_harmonized):,} observations")

    # ============================================================================
    # MOTHERHOOD PENALTY ANALYSIS
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 4: Motherhood Penalty Analysis (Ages 35-50)")
    print("-" * 80)

    # Filter to positive income
    analysis_sample = cps_harmonized[
        (cps_harmonized['income'].notna()) &
        (cps_harmonized['income'] > 0)
    ].copy()

    print(f"Analysis sample: {len(analysis_sample):,} observations with positive income")

    # Overall penalty
    mothers = analysis_sample[analysis_sample['mother'] == 1]
    childless = analysis_sample[analysis_sample['mother'] == 0]

    print(f"\nOverall (Ages 35-50):")
    print(f"  Mothers (n={len(mothers):,}): mean income = ${mothers['income'].mean():,.0f}")
    print(f"  Childless (n={len(childless):,}): mean income = ${childless['income'].mean():,.0f}")

    penalty = (childless['income'].mean() - mothers['income'].mean()) / childless['income'].mean() * 100
    print(f"  Motherhood Penalty: {penalty:+.1f}%")

    # By 5-year age bins
    print("\nBy Age Group:")
    results = []
    for age_min in [35, 40, 45]:
        age_max = age_min + 5
        subset = analysis_sample[(analysis_sample['age'] >= age_min) & (analysis_sample['age'] < age_max)]
        m = subset[subset['mother'] == 1]
        c = subset[subset['mother'] == 0]

        if len(m) >= 50 and len(c) >= 20:
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

            print(f"  Age {age_min}-{age_max}: Mothers=${m_inc:,.0f}, Childless=${c_inc:,.0f}, Penalty={pen:+.1f}%")
        else:
            print(f"  Age {age_min}-{age_max}: Insufficient data (mothers={len(m)}, childless={len(c)})")

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 5: Saving Results")
    print("-" * 80)

    # Save CPS harmonized data
    cps_harmonized.to_csv(OUTPUT_DIR / "cps_lifecycle_harmonized.csv", index=False)
    print("Saved: cps_lifecycle_harmonized.csv")

    # Save age-group results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "cps_penalty_by_age.csv", index=False)
        print("Saved: cps_penalty_by_age.csv")

    # ============================================================================
    # COMBINE WITH NLSY79 + HRS
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 6: Combining with NLSY79 + HRS Data")
    print("-" * 80)

    combined_path = OUTPUT_DIR / "combined_lifecycle_panel.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        print(f"Loaded existing combined panel: {len(existing):,} observations")

        # Add CPS data
        cps_for_merge = cps_harmonized[['id', 'year', 'age', 'female', 'mother', 'income',
                                        'education', 'race', 'weight', 'source']].copy()

        # Ensure consistent columns
        for col in existing.columns:
            if col not in cps_for_merge.columns:
                cps_for_merge[col] = np.nan

        cps_for_merge = cps_for_merge[existing.columns]

        # Combine
        full_combined = pd.concat([existing, cps_for_merge], ignore_index=True)
        print(f"Combined panel: {len(full_combined):,} observations")

        # Save
        full_combined.to_csv(OUTPUT_DIR / "combined_lifecycle_panel_with_cps.csv", index=False)
        print("Saved: combined_lifecycle_panel_with_cps.csv")

        # Calculate full lifecycle penalty
        print("\n" + "-" * 80)
        print("FULL LIFECYCLE MOTHERHOOD PENALTY")
        print("-" * 80)

        women = full_combined[full_combined['female'] == 1].copy()
        women = women[women['income'].notna() & (women['income'] > 0)]
        women = women[women['mother'].notna()]

        for age_min in [20, 25, 30, 35, 40, 45, 50, 55, 60]:
            age_max = age_min + 5
            subset = women[(women['age'] >= age_min) & (women['age'] < age_max)]
            m = subset[subset['mother'] == 1]
            c = subset[subset['mother'] == 0]

            if len(m) >= 50 and len(c) >= 15:
                m_inc = m['income'].mean()
                c_inc = c['income'].mean()
                pen = (c_inc - m_inc) / c_inc * 100
                source = subset['source'].mode().iloc[0] if len(subset) > 0 else 'mixed'
                print(f"Age {age_min}-{age_max}: Penalty = {pen:+.1f}% (n_m={len(m)}, n_c={len(c)}, source={source})")
            else:
                print(f"Age {age_min}-{age_max}: Insufficient data")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

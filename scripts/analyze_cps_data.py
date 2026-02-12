#!/usr/bin/env python3
"""
Analyze CPS Data to Fill Age 35-50 Gap
======================================

Uses existing CPS.csv file with CPS ASEC data to fill the lifecycle gap.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

print("=" * 80)
print("CPS LIFECYCLE GAP ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD CPS DATA
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading CPS Data")
print("-" * 80)

# Try CPS.csv first
cps_csv_path = Path("/Users/amalkova/Downloads/CPS.csv")

if cps_csv_path.exists():
    print(f"Loading {cps_csv_path}...")
    cps = pd.read_csv(cps_csv_path, low_memory=False)
    print(f"Loaded {len(cps):,} observations with {len(cps.columns)} variables")

    # Check key variables
    print("\nKey variables available:")
    key_vars = ['HRYEAR4', 'PRTAGE', 'PESEX', 'PRCHLD', 'PRNMCHLD', 'PTOTVAL', 'PEARNVAL',
                'PEEDUCA', 'PTDTRACE', 'PEHSPNON', 'PWSSWGT', 'PWSUPWGT', 'MARSUPWT']

    for var in key_vars:
        if var in cps.columns:
            print(f"  {var}: Found")
        else:
            # Check for similar names
            similar = [c for c in cps.columns if var[:4].upper() in c.upper()]
            if similar:
                print(f"  {var}: NOT FOUND (similar: {similar[:3]})")
            else:
                print(f"  {var}: NOT FOUND")

    # Print all column names to find the right ones
    print("\nAll column names containing key terms:")
    for term in ['YEAR', 'AGE', 'SEX', 'CHILD', 'EARN', 'INC', 'EDUC', 'RACE', 'WGT']:
        matches = [c for c in cps.columns if term in c.upper()]
        if matches:
            print(f"  {term}: {matches[:5]}")
else:
    print(f"CPS.csv not found at {cps_csv_path}")
    cps = None

# ============================================================================
# IDENTIFY VARIABLE NAMES
# ============================================================================

if cps is not None:
    print("\n" + "-" * 80)
    print("STEP 2: Identifying Variables")
    print("-" * 80)

    # Based on CPS Basic Monthly/ASEC variable names
    # HRYEAR4 = 4-digit year
    # PRTAGE = age
    # PESEX = sex (1=male, 2=female)
    # PRCHLD = number of children
    # PRNMCHLD = number of minor children
    # PTDTRACE = race
    # PEHSPNON = Hispanic origin
    # PEEDUCA = education

    # Check what year variable exists
    year_vars = [c for c in cps.columns if 'YEAR' in c.upper()]
    print(f"Year variables: {year_vars}")

    if 'HRYEAR4' in cps.columns:
        year_col = 'HRYEAR4'
    elif year_vars:
        year_col = year_vars[0]
    else:
        year_col = None

    if year_col:
        print(f"Using year column: {year_col}")
        print(f"Year range: {cps[year_col].min()} - {cps[year_col].max()}")
        print(f"Year distribution:\n{cps[year_col].value_counts().sort_index()}")

# ============================================================================
# PROCESS CPS DATA
# ============================================================================

if cps is not None and 'HRYEAR4' in cps.columns:
    print("\n" + "-" * 80)
    print("STEP 3: Processing CPS Data")
    print("-" * 80)

    # Variable mapping based on standard CPS names
    # Check which income variables exist
    income_vars = [c for c in cps.columns if any(x in c.upper() for x in ['EARN', 'INC', 'WAGE'])]
    print(f"Income-related variables: {income_vars[:10]}")

    # Create harmonized dataset
    cps_harm = pd.DataFrame()

    # Year
    cps_harm['year'] = cps['HRYEAR4']

    # Age
    if 'PRTAGE' in cps.columns:
        cps_harm['age'] = cps['PRTAGE']
    elif 'A_AGE' in cps.columns:
        cps_harm['age'] = cps['A_AGE']

    # Sex
    if 'PESEX' in cps.columns:
        cps_harm['female'] = (cps['PESEX'] == 2).astype(int)
    elif 'A_SEX' in cps.columns:
        cps_harm['female'] = (cps['A_SEX'] == 2).astype(int)

    # Children - check multiple variables
    child_vars = [c for c in cps.columns if 'CHILD' in c.upper() or 'CHLD' in c.upper()]
    print(f"Child variables: {child_vars}")

    if 'PRCHLD' in cps.columns:
        cps_harm['nchild'] = cps['PRCHLD']
    elif 'PRNMCHLD' in cps.columns:
        cps_harm['nchild'] = cps['PRNMCHLD']
    elif 'FOWNU18' in cps.columns:
        cps_harm['nchild'] = cps['FOWNU18']  # Own children under 18
    elif child_vars:
        # Use first available
        for cv in child_vars:
            if cps[cv].dtype in ['int64', 'float64']:
                cps_harm['nchild'] = cps[cv]
                print(f"Using {cv} for children")
                break

    # Income - look for total or earnings
    if 'PTOTVAL' in cps.columns:
        cps_harm['income'] = pd.to_numeric(cps['PTOTVAL'], errors='coerce')
    elif 'PEARNVAL' in cps.columns:
        cps_harm['income'] = pd.to_numeric(cps['PEARNVAL'], errors='coerce')
    elif 'WSAL_VAL' in cps.columns:
        cps_harm['income'] = pd.to_numeric(cps['WSAL_VAL'], errors='coerce')

    # Education
    if 'PEEDUCA' in cps.columns:
        cps_harm['education_raw'] = cps['PEEDUCA']
    elif 'A_HGA' in cps.columns:
        cps_harm['education_raw'] = cps['A_HGA']

    # Race
    if 'PTDTRACE' in cps.columns:
        cps_harm['race_raw'] = cps['PTDTRACE']
    elif 'PRDTRACE' in cps.columns:
        cps_harm['race_raw'] = cps['PRDTRACE']

    # Hispanic
    if 'PEHSPNON' in cps.columns:
        cps_harm['hispanic'] = cps['PEHSPNON']
    elif 'PRDTHSP' in cps.columns:
        cps_harm['hispanic'] = cps['PRDTHSP']

    # Weight
    weight_vars = [c for c in cps.columns if 'WGT' in c.upper() or 'WEIGHT' in c.upper()]
    print(f"Weight variables: {weight_vars[:5]}")
    if 'MARSUPWT' in cps.columns:
        cps_harm['weight'] = pd.to_numeric(cps['MARSUPWT'], errors='coerce')
    elif 'PWSSWGT' in cps.columns:
        cps_harm['weight'] = pd.to_numeric(cps['PWSSWGT'], errors='coerce')
    elif weight_vars:
        cps_harm['weight'] = pd.to_numeric(cps[weight_vars[0]], errors='coerce')

    print(f"\nHarmonized CPS data: {len(cps_harm):,} observations")
    print(f"Variables: {list(cps_harm.columns)}")

    # ============================================================================
    # FILTER TO ANALYSIS SAMPLE
    # ============================================================================

    print("\n" + "-" * 80)
    print("STEP 4: Filtering to Analysis Sample")
    print("-" * 80)

    # Calculate birth year
    cps_harm['birth_year'] = cps_harm['year'] - cps_harm['age']

    # Filter to:
    # - Women
    # - Birth cohort 1957-1964
    # - Ages 35-50

    if 'female' in cps_harm.columns and 'age' in cps_harm.columns:
        sample = cps_harm[
            (cps_harm['female'] == 1) &
            (cps_harm['birth_year'] >= 1957) &
            (cps_harm['birth_year'] <= 1964) &
            (cps_harm['age'] >= 35) &
            (cps_harm['age'] <= 50)
        ].copy()

        print(f"Women born 1957-1964, ages 35-50: {len(sample):,}")

        if len(sample) > 0:
            print(f"Year range: {sample['year'].min()}-{sample['year'].max()}")
            print(f"Age range: {sample['age'].min()}-{sample['age'].max()}")

            # Define motherhood
            if 'nchild' in sample.columns:
                sample['mother'] = (sample['nchild'] > 0).astype(int)
                n_mothers = sample['mother'].sum()
                n_childless = (sample['mother'] == 0).sum()
                print(f"\nMotherhood status:")
                print(f"  Mothers: {n_mothers:,}")
                print(f"  Childless: {n_childless:,}")
                print(f"  % Mothers: {n_mothers/(n_mothers+n_childless)*100:.1f}%")

            # ============================================================================
            # MOTHERHOOD PENALTY ANALYSIS
            # ============================================================================

            if 'mother' in sample.columns and 'income' in sample.columns:
                print("\n" + "-" * 80)
                print("STEP 5: Motherhood Penalty Analysis")
                print("-" * 80)

                # Filter to positive income
                analysis = sample[
                    (sample['income'].notna()) &
                    (sample['income'] > 0) &
                    (sample['mother'].notna())
                ].copy()

                print(f"Analysis sample with positive income: {len(analysis):,}")

                if len(analysis) > 100:
                    mothers = analysis[analysis['mother'] == 1]
                    childless = analysis[analysis['mother'] == 0]

                    print(f"\nOVERALL (Ages 35-50):")
                    print(f"  Mothers (n={len(mothers):,}): mean=${mothers['income'].mean():,.0f}, median=${mothers['income'].median():,.0f}")
                    print(f"  Childless (n={len(childless):,}): mean=${childless['income'].mean():,.0f}, median=${childless['income'].median():,.0f}")

                    if len(childless) > 0:
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

                        if len(m) >= 20 and len(c) >= 5:
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

                            print(f"  Age {age_min}-{age_max}: Mothers=${m_inc:,.0f} (n={len(m)}), Childless=${c_inc:,.0f} (n={len(c)}), Penalty={pen:+.1f}%")
                        else:
                            print(f"  Age {age_min}-{age_max}: Insufficient data (mothers={len(m)}, childless={len(c)})")

                    # Save results
                    if results:
                        results_df = pd.DataFrame(results)
                        results_df.to_csv(OUTPUT_DIR / "cps_penalty_by_age.csv", index=False)
                        print(f"\nSaved: cps_penalty_by_age.csv")

                    # Save harmonized CPS data
                    sample.to_csv(OUTPUT_DIR / "cps_harmonized.csv", index=False)
                    print(f"Saved: cps_harmonized.csv")

                    # ============================================================================
                    # COMBINE WITH EXISTING DATA
                    # ============================================================================

                    print("\n" + "-" * 80)
                    print("STEP 6: Combining with NLSY79 + HRS")
                    print("-" * 80)

                    combined_path = OUTPUT_DIR / "combined_lifecycle_panel.csv"
                    if combined_path.exists():
                        existing = pd.read_csv(combined_path)
                        print(f"Loaded existing combined panel: {len(existing):,} observations")

                        # Prepare CPS data for merge
                        cps_merge = analysis[['year', 'age', 'female', 'mother', 'income']].copy()
                        cps_merge['source'] = 'CPS'
                        cps_merge['id'] = range(1000000, 1000000 + len(cps_merge))

                        # Add missing columns
                        for col in existing.columns:
                            if col not in cps_merge.columns:
                                cps_merge[col] = np.nan

                        # Reorder columns
                        cps_merge = cps_merge[existing.columns]

                        # Combine
                        full_combined = pd.concat([existing, cps_merge], ignore_index=True)
                        print(f"Combined panel: {len(full_combined):,} observations")

                        # Save
                        full_combined.to_csv(OUTPUT_DIR / "combined_lifecycle_panel_with_cps.csv", index=False)
                        print("Saved: combined_lifecycle_panel_with_cps.csv")

                        # ============================================================================
                        # FULL LIFECYCLE ANALYSIS
                        # ============================================================================

                        print("\n" + "-" * 80)
                        print("FULL LIFECYCLE MOTHERHOOD PENALTY (with CPS)")
                        print("-" * 80)

                        women = full_combined[full_combined['female'] == 1].copy()
                        women = women[women['income'].notna() & (women['income'] > 0)]
                        women = women[women['mother'].notna()]

                        print(f"\n{'Age':<10} {'Penalty':>10} {'N Mothers':>12} {'N Childless':>14} {'Source':>10}")
                        print("-" * 60)

                        lifecycle_results = []
                        for age_min in [20, 25, 30, 35, 40, 45, 50, 55, 60]:
                            age_max = age_min + 5
                            subset = women[(women['age'] >= age_min) & (women['age'] < age_max)]
                            m = subset[subset['mother'] == 1]
                            c = subset[subset['mother'] == 0]

                            if len(m) >= 50 and len(c) >= 10:
                                m_inc = m['income'].mean()
                                c_inc = c['income'].mean()
                                pen = (c_inc - m_inc) / c_inc * 100
                                source = subset['source'].mode().iloc[0] if len(subset) > 0 else 'mixed'

                                lifecycle_results.append({
                                    'age_group': f"{age_min}-{age_max}",
                                    'n_mothers': len(m),
                                    'n_childless': len(c),
                                    'mothers_income': m_inc,
                                    'childless_income': c_inc,
                                    'penalty_pct': pen,
                                    'source': source
                                })

                                print(f"{age_min}-{age_max:<5} {pen:>+9.1f}% {len(m):>12,} {len(c):>14,} {source:>10}")
                            else:
                                print(f"{age_min}-{age_max:<5} {'(gap)':>10} {len(m):>12,} {len(c):>14,} {'---':>10}")

                        # Save lifecycle results
                        if lifecycle_results:
                            lifecycle_df = pd.DataFrame(lifecycle_results)
                            lifecycle_df.to_csv(OUTPUT_DIR / "lifecycle_penalty_with_cps.csv", index=False)
                            print(f"\nSaved: lifecycle_penalty_with_cps.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

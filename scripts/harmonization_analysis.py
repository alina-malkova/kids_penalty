#!/usr/bin/env python3
"""
NLSY79 + RAND HRS Data Harmonization and Analysis
==================================================
Kids Penalty Project: Heterogeneous Effects of Children on Women's Income

This script:
1. Loads and harmonizes NLSY79 (new extract with fertility/income)
2. Loads and harmonizes RAND HRS 2022
3. Creates synthetic cohort comparison (birth cohort 1957-1964)
4. Generates descriptive statistics and analysis outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data paths
NLSY_PATH = Path("/Users/amalkova/Downloads/NLSY2_Data/nlsy2.csv")
HRS_PATH = Path("/Users/amalkova/Downloads/RAND_HRS_2022/randhrs1992_2022v1.dta")

print("=" * 80)
print("KIDS PENALTY PROJECT: DATA HARMONIZATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PROCESS NLSY79
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading NLSY79 (New Extract)")
print("-" * 80)

nlsy = pd.read_csv(NLSY_PATH)
print(f"Loaded {len(nlsy):,} respondents with {len(nlsy.columns)} variables")

# NLSY Variable Mapping (from codebook)
# R0000100 = CASEID
# R0214700 = SAMPLE_RACE (1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic)
# R0214800 = SAMPLE_SEX (1=Male, 2=Female)
# R0217900 = TNFI_TRUNC 1979
# R0218001 = NUMCH79

# Get birth year from first NLSY survey (respondents were 14-22 in 1979)
# Need to calculate from sample info or use derived variable

# Create harmonized NLSY dataframe
nlsy_harm = pd.DataFrame()
nlsy_harm['id'] = nlsy['R0000100']
nlsy_harm['source'] = 'NLSY79'

# Demographics
nlsy_harm['race_orig'] = nlsy['R0214700']  # 1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic
nlsy_harm['female'] = (nlsy['R0214800'] == 2).astype(int)

# Recode race to match HRS (1=White, 2=Black, 3=Other/Hispanic)
nlsy_harm['race'] = np.where(nlsy_harm['race_orig'] == 3, 1,  # Non-Black/Non-Hispanic -> White
                    np.where(nlsy_harm['race_orig'] == 2, 2,   # Black -> Black
                    3))  # Hispanic -> Other

# Sample ID for identifying birth cohort
nlsy_harm['sample_id'] = nlsy['R0173600']

# Income variables by year (TNFI_TRUNC)
income_vars = {
    1979: 'R0217900',
    1980: 'R0406010',
    1981: 'R0618410',
    1982: 'R0898600',
    1983: 'R1144500',
    1984: 'R1519700',
    1985: 'R1890400',
    1986: 'R2257500',
    1987: 'R2444700',
    1988: 'R2870200',
    1990: 'R3400700',
    1991: 'R3656100',
    1992: 'R4006600',
    1993: 'R4417700',
}

# Children count by year (NUMCH##)
children_vars = {
    1979: 'R0218001',
    1980: 'R0407601',
    1981: 'R0647101',
    1982: 'R0898838',
    1983: 'R1146830',
    1984: 'R1522037',
    1985: 'R1892737',
    1986: 'R2259837',
    1987: 'R2448037',
    1988: 'R2877600',
    1990: 'R3407700',
    1991: 'R3659047',
    1992: 'R4009447',
}

# Add income and children for each year
for year, var in income_vars.items():
    if var in nlsy.columns:
        col_name = f'income_{year}'
        nlsy_harm[col_name] = pd.to_numeric(nlsy[var], errors='coerce')
        # Replace negative values (missing codes) with NaN
        nlsy_harm.loc[nlsy_harm[col_name] < 0, col_name] = np.nan

for year, var in children_vars.items():
    if var in nlsy.columns:
        col_name = f'children_{year}'
        nlsy_harm[col_name] = pd.to_numeric(nlsy[var], errors='coerce')
        nlsy_harm.loc[nlsy_harm[col_name] < 0, col_name] = np.nan

# Has children ever (use 1979 baseline question R0013300 = Q9-72)
if 'R0013300' in nlsy.columns:
    nlsy_harm['ever_had_children_1979'] = (nlsy['R0013300'] == 1).astype(int)
    nlsy_harm.loc[nlsy['R0013300'] < 0, 'ever_had_children_1979'] = np.nan

# Calculate latest children count
child_cols = [c for c in nlsy_harm.columns if c.startswith('children_')]
if child_cols:
    nlsy_harm['children_latest'] = nlsy_harm[child_cols].ffill(axis=1).iloc[:, -1]
    nlsy_harm['has_children'] = (nlsy_harm['children_latest'] > 0).astype(int)

# Calculate latest income
income_cols = [c for c in nlsy_harm.columns if c.startswith('income_')]
if income_cols:
    nlsy_harm['income_latest'] = nlsy_harm[income_cols].ffill(axis=1).iloc[:, -1]

print(f"NLSY79 harmonized: {len(nlsy_harm):,} observations")
print(f"Variables: {len(nlsy_harm.columns)}")

# ============================================================================
# STEP 2: LOAD AND PROCESS HRS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Loading RAND HRS 2022")
print("-" * 80)

# HRS variables to load
hrs_vars = [
    'hhidpn',      # ID
    'rabyear',     # Birth year
    'ragender',    # Gender
    'raracem',     # Race
    'raeduc',      # Education
]

# Add wave-specific variables (waves 1-16 = 1992-2022)
for w in range(1, 17):
    hrs_vars.extend([
        f'h{w}child',   # Number of children in household
        f'h{w}itot',    # Household total income
    ])

print(f"Loading HRS with {len(hrs_vars)} variables...")

try:
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
except Exception as e:
    print(f"Loading subset of variables... ({e})")
    # Try loading with fewer variables if memory issues
    hrs_vars_basic = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc',
                      'h1child', 'h16child', 'h1itot', 'h16itot']
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars_basic, convert_categoricals=False)

print(f"Loaded {len(hrs):,} respondents")

# Filter to 1957-1964 birth cohort (matches NLSY79)
hrs_cohort = hrs[(hrs['rabyear'] >= 1957) & (hrs['rabyear'] <= 1964)].copy()
print(f"HRS 1957-1964 cohort: {len(hrs_cohort):,} respondents")

# Create harmonized HRS dataframe
hrs_harm = pd.DataFrame()
hrs_harm['id'] = hrs_cohort['hhidpn']
hrs_harm['source'] = 'HRS'
hrs_harm['birth_year'] = hrs_cohort['rabyear']

# Gender
hrs_harm['female'] = (hrs_cohort['ragender'] == 2).astype(int)

# Race (HRS: 1=White, 2=Black, 3=Other)
hrs_harm['race'] = hrs_cohort['raracem']

# Education
hrs_harm['education'] = hrs_cohort['raeduc']

# Children (use latest wave available)
for w in range(16, 0, -1):
    col = f'h{w}child'
    if col in hrs_cohort.columns:
        valid = hrs_cohort[col].notna()
        if valid.sum() > 1000:
            hrs_harm['children_latest'] = hrs_cohort[col]
            hrs_harm['children_wave'] = w
            break

hrs_harm['has_children'] = (hrs_harm['children_latest'] > 0).astype(int)

# Income (use latest wave - household total income)
for w in range(16, 0, -1):
    col = f'h{w}itot'
    if col in hrs_cohort.columns:
        valid = hrs_cohort[col].notna()
        if valid.sum() > 1000:
            hrs_harm['income_latest'] = hrs_cohort[col]
            hrs_harm['income_wave'] = w
            break

print(f"HRS harmonized: {len(hrs_harm):,} observations")

# ============================================================================
# STEP 3: SUMMARY STATISTICS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Summary Statistics")
print("-" * 80)

def print_summary(df, name):
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)

    n = len(df)
    print(f"Total observations: {n:,}")

    if 'female' in df.columns:
        n_female = df['female'].sum()
        pct_female = n_female / n * 100
        print(f"\nGender:")
        print(f"  Female: {n_female:,} ({pct_female:.1f}%)")
        print(f"  Male: {n - n_female:,} ({100-pct_female:.1f}%)")

    if 'race' in df.columns:
        print(f"\nRace:")
        race_map = {1: 'White', 2: 'Black', 3: 'Hispanic/Other'}
        for code, label in race_map.items():
            n_race = (df['race'] == code).sum()
            print(f"  {label}: {n_race:,} ({n_race/n*100:.1f}%)")

    if 'has_children' in df.columns:
        n_children = df['has_children'].sum()
        print(f"\nChildren:")
        print(f"  Has children: {n_children:,} ({n_children/n*100:.1f}%)")
        print(f"  Childless: {n - n_children:,} ({(n-n_children)/n*100:.1f}%)")

    if 'children_latest' in df.columns:
        valid = df['children_latest'].dropna()
        if len(valid) > 0:
            print(f"  Mean children: {valid.mean():.2f}")
            print(f"  Median children: {valid.median():.0f}")

    if 'income_latest' in df.columns:
        valid = df['income_latest'].dropna()
        valid = valid[valid > 0]
        if len(valid) > 0:
            print(f"\nIncome:")
            print(f"  Mean: ${valid.mean():,.0f}")
            print(f"  Median: ${valid.median():,.0f}")
            print(f"  25th percentile: ${valid.quantile(0.25):,.0f}")
            print(f"  75th percentile: ${valid.quantile(0.75):,.0f}")

# NLSY Summary (females only for motherhood analysis)
nlsy_female = nlsy_harm[nlsy_harm['female'] == 1].copy()
print_summary(nlsy_female, "NLSY79 - FEMALES")

# HRS Summary (females only, 1957-1964 cohort)
hrs_female = hrs_harm[hrs_harm['female'] == 1].copy()
print_summary(hrs_female, "HRS - FEMALES (Born 1957-1964)")

# ============================================================================
# STEP 4: MOTHERHOOD PENALTY ANALYSIS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Motherhood Penalty Analysis")
print("-" * 80)

def analyze_motherhood_penalty(df, name):
    """Calculate income gap between mothers and childless women."""

    print(f"\n{name}")
    print("-" * 40)

    # Filter to females with valid income
    women = df[(df['female'] == 1) & (df['income_latest'].notna()) & (df['income_latest'] > 0)].copy()

    if len(women) < 100:
        print("Insufficient data for analysis")
        return None

    mothers = women[women['has_children'] == 1]
    childless = women[women['has_children'] == 0]

    if len(mothers) < 50 or len(childless) < 50:
        print("Insufficient data for comparison")
        return None

    mean_mothers = mothers['income_latest'].mean()
    mean_childless = childless['income_latest'].mean()

    penalty = mean_childless - mean_mothers
    penalty_pct = (penalty / mean_childless) * 100

    print(f"Sample sizes:")
    print(f"  Mothers: {len(mothers):,}")
    print(f"  Childless: {len(childless):,}")
    print(f"\nMean Income:")
    print(f"  Mothers: ${mean_mothers:,.0f}")
    print(f"  Childless: ${mean_childless:,.0f}")
    print(f"\nMotherhood Penalty:")
    print(f"  Absolute: ${penalty:,.0f}")
    print(f"  Relative: {penalty_pct:.1f}%")

    # By race
    print(f"\nBy Race:")
    race_map = {1: 'White', 2: 'Black', 3: 'Hispanic/Other'}
    for code, label in race_map.items():
        race_women = women[women['race'] == code]
        race_mothers = race_women[race_women['has_children'] == 1]
        race_childless = race_women[race_women['has_children'] == 0]

        if len(race_mothers) >= 30 and len(race_childless) >= 30:
            m_inc = race_mothers['income_latest'].mean()
            c_inc = race_childless['income_latest'].mean()
            r_penalty = (c_inc - m_inc) / c_inc * 100
            print(f"  {label}: {r_penalty:.1f}% (n={len(race_women):,})")

    return {
        'n_mothers': len(mothers),
        'n_childless': len(childless),
        'mean_mothers': mean_mothers,
        'mean_childless': mean_childless,
        'penalty_absolute': penalty,
        'penalty_pct': penalty_pct
    }

nlsy_results = analyze_motherhood_penalty(nlsy_harm, "NLSY79 (Early/Mid Career)")
hrs_results = analyze_motherhood_penalty(hrs_harm, "HRS (Late Career, Ages 58-67)")

# ============================================================================
# STEP 5: LIFECYCLE COMPARISON
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Lifecycle Comparison")
print("-" * 80)

if nlsy_results and hrs_results:
    print(f"""
LIFECYCLE EVOLUTION OF MOTHERHOOD PENALTY
=========================================

                          NLSY79              HRS
                        (Early/Mid)      (Late Career)

Sample Size:
  Mothers                {nlsy_results['n_mothers']:,}            {hrs_results['n_mothers']:,}
  Childless              {nlsy_results['n_childless']:,}            {hrs_results['n_childless']:,}

Mean Income:
  Mothers            ${nlsy_results['mean_mothers']:>12,.0f}    ${hrs_results['mean_mothers']:>12,.0f}
  Childless          ${nlsy_results['mean_childless']:>12,.0f}    ${hrs_results['mean_childless']:>12,.0f}

Motherhood Penalty:
  Absolute           ${nlsy_results['penalty_absolute']:>12,.0f}    ${hrs_results['penalty_absolute']:>12,.0f}
  Relative                 {nlsy_results['penalty_pct']:>6.1f}%          {hrs_results['penalty_pct']:>6.1f}%

Key Finding:
""")

    if hrs_results['penalty_pct'] < nlsy_results['penalty_pct']:
        print("  The motherhood penalty DECREASES over the lifecycle.")
        print(f"  Gap narrows from {nlsy_results['penalty_pct']:.1f}% to {hrs_results['penalty_pct']:.1f}%")
    else:
        print("  The motherhood penalty PERSISTS into late career.")
        print(f"  Gap remains at {hrs_results['penalty_pct']:.1f}% (vs {nlsy_results['penalty_pct']:.1f}% in early career)")

# ============================================================================
# STEP 6: SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Saving Outputs")
print("-" * 80)

# Save harmonized datasets
nlsy_harm.to_csv(OUTPUT_DIR / "nlsy79_harmonized_v2.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'nlsy79_harmonized_v2.csv'}")

hrs_harm.to_csv(OUTPUT_DIR / "hrs_harmonized_v2.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'hrs_harmonized_v2.csv'}")

# Save summary results
results_df = pd.DataFrame([
    {'Dataset': 'NLSY79', 'Stage': 'Early/Mid Career', **nlsy_results} if nlsy_results else {},
    {'Dataset': 'HRS', 'Stage': 'Late Career', **hrs_results} if hrs_results else {}
])
results_df.to_csv(OUTPUT_DIR / "motherhood_penalty_results.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'motherhood_penalty_results.csv'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

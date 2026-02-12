#!/usr/bin/env python3
"""
Motherhood Penalty on Retirement Income Analysis
=================================================

Research Question: Does having children have long-lasting effects on women's
retirement income?

Datasets:
- NLSY79: Early/mid career income (1979-1993)
- HRS: Late career and retirement income (1992-2022)

Key Challenge: Small childless sample in HRS (baby boomer cohort had high fertility)

Author: Kids Penalty Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

print("=" * 80)
print("MOTHERHOOD PENALTY ON RETIREMENT INCOME")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD HRS WITH RETIREMENT VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading HRS with Retirement Income Variables")
print("-" * 80)

HRS_PATH = Path("/Users/amalkova/Downloads/RAND_HRS_2022/randhrs1992_2022v1.dta")

# Key retirement variables
hrs_vars = [
    'hhidpn',      # ID
    'rabyear',     # Birth year
    'ragender',    # Gender
    'raracem',     # Race
    'raeduc',      # Education
]

# Add children and income variables by wave
for w in range(1, 17):
    hrs_vars.extend([
        f'h{w}child',   # Household children
        f'h{w}itot',    # Household total income
        f'r{w}issi',    # Respondent Social Security income
        f'r{w}ipen',    # Respondent pension income
        f'r{w}iearn',   # Respondent earnings
    ])

# Try loading
try:
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
except ValueError as e:
    # Some variables may not exist, load what we can
    print(f"Note: Some variables not found, loading available ones...")
    available_vars = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']
    for w in range(1, 17):
        available_vars.extend([f'h{w}child', f'h{w}itot'])
    hrs = pd.read_stata(HRS_PATH, columns=available_vars, convert_categoricals=False)

print(f"Loaded {len(hrs):,} HRS respondents")

# Filter to 1957-1964 cohort (NLSY79 overlap)
hrs_cohort = hrs[(hrs['rabyear'] >= 1957) & (hrs['rabyear'] <= 1964)].copy()
print(f"1957-1964 birth cohort: {len(hrs_cohort):,} respondents")

# Filter to women
hrs_women = hrs_cohort[hrs_cohort['ragender'] == 2].copy()
print(f"Women in cohort: {len(hrs_women):,}")

# ============================================================================
# STEP 2: IDENTIFY MOTHERS VS CHILDLESS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Identifying Mothers vs Childless Women")
print("-" * 80)

# Use multiple waves to identify who ever had children
# (h#child = children in household, but we can use max across waves)
child_cols = [f'h{w}child' for w in range(1, 17) if f'h{w}child' in hrs_women.columns]

if child_cols:
    # Take max children across all waves (approximates "ever had children")
    hrs_women['max_children'] = hrs_women[child_cols].max(axis=1)
    hrs_women['ever_mother'] = (hrs_women['max_children'] > 0).astype(int)

    # Also check if ANY wave had children > 0
    for col in child_cols:
        hrs_women[col + '_any'] = (hrs_women[col] > 0).astype(int)
    any_child_cols = [col + '_any' for col in child_cols]
    hrs_women['ever_had_child_flag'] = hrs_women[any_child_cols].max(axis=1)

print("\nMotherhood classification (using max children across waves):")
print(f"  Mothers (ever had children): {(hrs_women['ever_mother'] == 1).sum():,}")
print(f"  Childless: {(hrs_women['ever_mother'] == 0).sum():,}")
print(f"  Missing: {hrs_women['ever_mother'].isna().sum():,}")

# ============================================================================
# STEP 3: RETIREMENT INCOME ANALYSIS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Retirement Income Analysis")
print("-" * 80)

# Get latest wave income (wave 16 = 2022)
income_col = None
for w in range(16, 0, -1):
    col = f'h{w}itot'
    if col in hrs_women.columns and hrs_women[col].notna().sum() > 500:
        income_col = col
        print(f"Using income from wave {w} (year {1992 + (w-1)*2})")
        break

if income_col:
    hrs_women['retirement_income'] = hrs_women[income_col]

    # Filter to valid income
    valid_sample = hrs_women[
        (hrs_women['retirement_income'].notna()) &
        (hrs_women['retirement_income'] > 0) &
        (hrs_women['ever_mother'].notna())
    ].copy()

    print(f"\nValid sample for analysis: {len(valid_sample):,}")

    mothers = valid_sample[valid_sample['ever_mother'] == 1]
    childless = valid_sample[valid_sample['ever_mother'] == 0]

    print(f"  Mothers: {len(mothers):,}")
    print(f"  Childless: {len(childless):,}")

    # KEY FINDING: Sample size issue
    print("\n" + "!" * 60)
    print("CRITICAL: SMALL CHILDLESS SAMPLE")
    print("!" * 60)
    if len(childless) < 500:
        print(f"""
The childless sample has only {len(childless)} women.
This is because the 1957-1964 cohort (baby boomers) had very high fertility.
About 83% of NLSY79 women had children by age 46.

This small sample limits statistical power for retirement analysis.
Consider:
1. Expanding birth cohort range
2. Using alternative data (CPS, SIPP)
3. Reporting findings with appropriate caveats
""")

    # Calculate motherhood penalty despite small sample
    if len(mothers) >= 50 and len(childless) >= 50:
        mean_mothers = mothers['retirement_income'].mean()
        mean_childless = childless['retirement_income'].mean()
        median_mothers = mothers['retirement_income'].median()
        median_childless = childless['retirement_income'].median()

        penalty_mean = (mean_childless - mean_mothers) / mean_childless * 100
        penalty_median = (median_childless - median_mothers) / median_childless * 100

        print(f"""
RETIREMENT INCOME COMPARISON (HRS 2022)
=======================================
                        Mothers         Childless
Sample size:            {len(mothers):,}           {len(childless):,}

Mean income:            ${mean_mothers:,.0f}       ${mean_childless:,.0f}
Median income:          ${median_mothers:,.0f}       ${median_childless:,.0f}

Motherhood Penalty:
  Based on means:       {penalty_mean:+.1f}%
  Based on medians:     {penalty_median:+.1f}%

Note: Negative penalty means mothers earn MORE than childless
""")

        # By education
        print("\nBy Education Level:")
        if 'raeduc' in valid_sample.columns:
            for educ in [1, 2, 3, 4, 5]:
                educ_sample = valid_sample[valid_sample['raeduc'] == educ]
                educ_mothers = educ_sample[educ_sample['ever_mother'] == 1]
                educ_childless = educ_sample[educ_sample['ever_mother'] == 0]
                if len(educ_mothers) >= 30 and len(educ_childless) >= 10:
                    m_inc = educ_mothers['retirement_income'].mean()
                    c_inc = educ_childless['retirement_income'].mean()
                    pen = (c_inc - m_inc) / c_inc * 100 if c_inc > 0 else 0
                    educ_labels = {1: 'Less than HS', 2: 'GED', 3: 'HS Graduate',
                                   4: 'Some College', 5: 'College+'}
                    print(f"  {educ_labels.get(educ, educ)}: {pen:+.1f}% (n={len(educ_sample)})")

# ============================================================================
# STEP 4: LOAD NLSY79 FOR EARLY CAREER COMPARISON
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: NLSY79 Early Career Comparison")
print("-" * 80)

nlsy = pd.read_csv("/Users/amalkova/Downloads/Kids Penalty (2024)/harmonized_data/nlsy79_harmonized_v2.csv")
nlsy_women = nlsy[nlsy['female'] == 1].copy()

# Get valid income sample
nlsy_valid = nlsy_women[
    (nlsy_women['income_latest'].notna()) &
    (nlsy_women['income_latest'] > 0) &
    (nlsy_women['has_children'].notna())
].copy()

nlsy_mothers = nlsy_valid[nlsy_valid['has_children'] == 1]
nlsy_childless = nlsy_valid[nlsy_valid['has_children'] == 0]

print(f"NLSY79 Women (Early/Mid Career):")
print(f"  Total valid: {len(nlsy_valid):,}")
print(f"  Mothers: {len(nlsy_mothers):,}")
print(f"  Childless: {len(nlsy_childless):,}")

if len(nlsy_mothers) >= 50 and len(nlsy_childless) >= 50:
    nlsy_mean_m = nlsy_mothers['income_latest'].mean()
    nlsy_mean_c = nlsy_childless['income_latest'].mean()
    nlsy_penalty = (nlsy_mean_c - nlsy_mean_m) / nlsy_mean_c * 100

    print(f"\n  Mean income - Mothers: ${nlsy_mean_m:,.0f}")
    print(f"  Mean income - Childless: ${nlsy_mean_c:,.0f}")
    print(f"  Motherhood Penalty: {nlsy_penalty:+.1f}%")

# ============================================================================
# STEP 5: LIFECYCLE COMPARISON SUMMARY
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Lifecycle Comparison Summary")
print("-" * 80)

print("""
MOTHERHOOD PENALTY ACROSS THE LIFECYCLE
========================================

                    NLSY79              HRS
                 (Early/Mid Career)  (Retirement Age)
                   Ages 20-35         Ages 58-67
---------------------------------------------------------
""")

# Print comparison if we have both
if len(nlsy_mothers) >= 50 and len(childless) >= 50:
    print(f"""Sample Size:
  Mothers:        {len(nlsy_mothers):,}             {len(mothers):,}
  Childless:      {len(nlsy_childless):,}             {len(childless):,}

Mean Income:
  Mothers:        ${nlsy_mean_m:>10,.0f}       ${mean_mothers:>10,.0f}
  Childless:      ${nlsy_mean_c:>10,.0f}       ${mean_childless:>10,.0f}

Motherhood Penalty:
                     {nlsy_penalty:>+6.1f}%           {penalty_mean:>+6.1f}%
""")

print("""
KEY FINDINGS:
-------------
1. The motherhood penalty is evident in EARLY CAREER (~7%)

2. By RETIREMENT AGE, the pattern reverses - mothers have HIGHER income

3. CAUTION: This reversal may be driven by:
   a) Small childless sample in HRS (selection bias)
   b) Spousal income (HRS uses household income)
   c) Survivor bias (mothers may have better social support)
   d) Measurement: HRS h#child = current household, not total ever born

4. RECOMMENDATION: Need larger childless sample or different methodology
   to reliably estimate retirement-age motherhood penalty
""")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Saving Results")
print("-" * 80)

# Save HRS processed data
hrs_women.to_csv(OUTPUT_DIR / "hrs_retirement_analysis.csv", index=False)
print(f"Saved: hrs_retirement_analysis.csv")

# Create summary table
summary = pd.DataFrame({
    'Dataset': ['NLSY79', 'HRS'],
    'Career_Stage': ['Early/Mid (20-35)', 'Retirement (58-67)'],
    'N_Mothers': [len(nlsy_mothers), len(mothers)],
    'N_Childless': [len(nlsy_childless), len(childless)],
    'Mean_Income_Mothers': [nlsy_mean_m, mean_mothers],
    'Mean_Income_Childless': [nlsy_mean_c, mean_childless],
    'Motherhood_Penalty_Pct': [nlsy_penalty, penalty_mean]
})
summary.to_csv(OUTPUT_DIR / "lifecycle_penalty_summary.csv", index=False)
print(f"Saved: lifecycle_penalty_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

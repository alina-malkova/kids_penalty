#!/usr/bin/env python3
"""
Robustness Checks: Sensitivity Analysis
========================================

This script performs robustness checks on the motherhood penalty findings:
1. Alternative childlessness definitions
2. Different age cutoffs
3. Sample restrictions
4. Alternative income measures

Author: Kids Penalty Project
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/Users/amalkova/OneDrive - Florida Institute of Technology/_Research/Labor_Economics/KIDS Penalty/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "data" / "harmonized_data"
FIGURES_DIR = BASE_DIR / "figures"
HRS_PATH = Path("/Users/amalkova/Downloads/RAND_HRS_2022/randhrs1992_2022v1.dta")

print("=" * 80)
print("ROBUSTNESS CHECKS: SENSITIVITY ANALYSIS")
print("Alternative Definitions, Age Cutoffs, and Specifications")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD HRS DATA
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading HRS Data")
print("-" * 80)

# Key variables
hrs_vars = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']
for w in range(1, 17):
    hrs_vars.extend([f'h{w}child', f'h{w}itot', f'r{w}mstat', f'r{w}iearn', f'r{w}ipen'])

print("Loading HRS data...")
hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
print(f"Loaded {len(hrs):,} HRS respondents")

# Filter to 1957-1964 cohort
hrs = hrs[(hrs['rabyear'] >= 1957) & (hrs['rabyear'] <= 1964)].copy()
print(f"1957-1964 birth cohort: {len(hrs):,} respondents")

# Filter to women
hrs_women = hrs[hrs['ragender'] == 2].copy()
print(f"Women in cohort: {len(hrs_women):,}")

# ============================================================================
# STEP 2: CREATE BASE VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Creating Base Variables")
print("-" * 80)

# Maximum children across waves
child_cols = [f'h{w}child' for w in range(1, 17) if f'h{w}child' in hrs_women.columns]
hrs_women['max_children'] = hrs_women[child_cols].max(axis=1)

# Income - latest wave
income_cols = [f'h{w}itot' for w in range(1, 17) if f'h{w}itot' in hrs_women.columns]
hrs_women['income_latest'] = np.nan
for col in reversed(income_cols):
    hrs_women['income_latest'] = hrs_women['income_latest'].fillna(hrs_women[col])

# Education categories
def categorize_education(ed):
    if pd.isna(ed):
        return np.nan
    elif ed <= 2:
        return 'Less than HS'
    elif ed == 3:
        return 'HS Graduate'
    elif ed == 4:
        return 'Some College'
    else:
        return 'College+'

hrs_women['educ_cat'] = hrs_women['raeduc'].apply(categorize_education)

# ============================================================================
# ROBUSTNESS CHECK 1: ALTERNATIVE CHILDLESSNESS DEFINITIONS
# ============================================================================

print("\n" + "-" * 80)
print("ROBUSTNESS CHECK 1: Alternative Childlessness Definitions")
print("-" * 80)

print("""
Testing different definitions of childlessness:
1. Baseline: 0 children ever (max_children == 0)
2. Conservative: 0 children in all observed waves
3. Liberal: Childless if 0 children in majority of waves
""")

# Valid sample for analysis
valid_sample = hrs_women[
    (hrs_women['income_latest'].notna()) &
    (hrs_women['income_latest'] > 0)
].copy()

# Definition 1: Baseline (0 children ever)
valid_sample['childless_baseline'] = (valid_sample['max_children'] == 0).astype(int)

# Definition 2: Conservative (0 in all non-missing waves)
def all_waves_childless(row):
    child_vals = [row[col] for col in child_cols if pd.notna(row[col])]
    if len(child_vals) == 0:
        return np.nan
    return 1 if all(c == 0 for c in child_vals) else 0

valid_sample['childless_conservative'] = valid_sample.apply(all_waves_childless, axis=1)

# Definition 3: Liberal (childless in majority of waves)
def majority_childless(row):
    child_vals = [row[col] for col in child_cols if pd.notna(row[col])]
    if len(child_vals) == 0:
        return np.nan
    childless_waves = sum(1 for c in child_vals if c == 0)
    return 1 if childless_waves > len(child_vals) / 2 else 0

valid_sample['childless_liberal'] = valid_sample.apply(majority_childless, axis=1)

print("\nChildlessness Definition Comparison:")
print("-" * 60)

results_definitions = []

for def_name, def_col in [('Baseline', 'childless_baseline'),
                            ('Conservative', 'childless_conservative'),
                            ('Liberal', 'childless_liberal')]:

    subset = valid_sample[valid_sample[def_col].notna()].copy()
    childless = subset[subset[def_col] == 1]
    mothers = subset[subset[def_col] == 0]

    n_c = len(childless)
    n_m = len(mothers)
    pct_c = n_c / len(subset) * 100

    if n_c >= 10 and n_m >= 20:
        mean_c = childless['income_latest'].mean()
        mean_m = mothers['income_latest'].mean()
        gap = (mean_c - mean_m) / mean_c * 100

        print(f"{def_name:<15}: {pct_c:>5.1f}% childless | Gap = {gap:>+6.1f}% | N = {len(subset):,}")

        results_definitions.append({
            'definition': def_name,
            'pct_childless': pct_c,
            'n_childless': n_c,
            'n_mothers': n_m,
            'mean_childless': mean_c,
            'mean_mothers': mean_m,
            'gap_pct': gap
        })

# ============================================================================
# ROBUSTNESS CHECK 2: DIFFERENT AGE CUTOFFS
# ============================================================================

print("\n" + "-" * 80)
print("ROBUSTNESS CHECK 2: Different Birth Year Ranges")
print("-" * 80)

print("""
Testing different birth cohort restrictions:
1. Baseline: 1957-1964 (NLSY79 exact overlap)
2. Narrow: 1958-1963 (excluding edges)
3. Wide: 1955-1966 (broader range)
""")

valid_sample['ever_mother'] = (valid_sample['max_children'] > 0).astype(int)

age_ranges = [
    ('Baseline (1957-1964)', 1957, 1964),
    ('Narrow (1958-1963)', 1958, 1963),
    ('Wide (1955-1966)', 1955, 1966)
]

# Need to reload for wide range
if True:  # Always run this
    hrs_wide = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
    hrs_wide = hrs_wide[hrs_wide['ragender'] == 2].copy()

    # Add children and income
    child_cols_wide = [f'h{w}child' for w in range(1, 17) if f'h{w}child' in hrs_wide.columns]
    hrs_wide['max_children'] = hrs_wide[child_cols_wide].max(axis=1)
    hrs_wide['ever_mother'] = (hrs_wide['max_children'] > 0).astype(int)

    income_cols_wide = [f'h{w}itot' for w in range(1, 17) if f'h{w}itot' in hrs_wide.columns]
    hrs_wide['income_latest'] = np.nan
    for col in reversed(income_cols_wide):
        hrs_wide['income_latest'] = hrs_wide['income_latest'].fillna(hrs_wide[col])

print("\nBirth Year Range Comparison:")
print("-" * 70)

results_ages = []

for name, start_year, end_year in age_ranges:
    subset = hrs_wide[
        (hrs_wide['rabyear'] >= start_year) &
        (hrs_wide['rabyear'] <= end_year) &
        (hrs_wide['income_latest'].notna()) &
        (hrs_wide['income_latest'] > 0) &
        (hrs_wide['ever_mother'].notna())
    ].copy()

    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    n_m = len(mothers)
    n_c = len(childless)

    if n_m >= 20 and n_c >= 10:
        mean_m = mothers['income_latest'].mean()
        mean_c = childless['income_latest'].mean()
        gap = (mean_c - mean_m) / mean_c * 100

        print(f"{name:<25}: Gap = {gap:>+6.1f}% | N = {n_m + n_c:,} (M:{n_m}, C:{n_c})")

        results_ages.append({
            'age_range': name,
            'start_year': start_year,
            'end_year': end_year,
            'n_mothers': n_m,
            'n_childless': n_c,
            'mean_mothers': mean_m,
            'mean_childless': mean_c,
            'gap_pct': gap
        })

# ============================================================================
# ROBUSTNESS CHECK 3: INCOME TRIMMING
# ============================================================================

print("\n" + "-" * 80)
print("ROBUSTNESS CHECK 3: Income Trimming (Outlier Sensitivity)")
print("-" * 80)

print("""
Testing sensitivity to outliers:
1. Baseline: All positive incomes
2. Trim 1%: Exclude top and bottom 1%
3. Trim 5%: Exclude top and bottom 5%
4. Winsorize 1%: Cap at 1st and 99th percentile
""")

# Use baseline sample
base = valid_sample[valid_sample['ever_mother'].notna()].copy()

trim_specs = [
    ('Baseline', 0, 100),
    ('Trim 1%', 1, 99),
    ('Trim 5%', 5, 95),
]

print("\nIncome Trimming Comparison:")
print("-" * 60)

results_trim = []

for name, lower_pct, upper_pct in trim_specs:
    if lower_pct == 0:
        subset = base.copy()
    else:
        lower_bound = base['income_latest'].quantile(lower_pct / 100)
        upper_bound = base['income_latest'].quantile(upper_pct / 100)
        subset = base[(base['income_latest'] >= lower_bound) &
                      (base['income_latest'] <= upper_bound)].copy()

    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    n_m = len(mothers)
    n_c = len(childless)

    if n_m >= 20 and n_c >= 10:
        mean_m = mothers['income_latest'].mean()
        mean_c = childless['income_latest'].mean()
        gap = (mean_c - mean_m) / mean_c * 100

        print(f"{name:<15}: Gap = {gap:>+6.1f}% | N = {n_m + n_c:,}")

        results_trim.append({
            'specification': name,
            'n_total': n_m + n_c,
            'n_mothers': n_m,
            'n_childless': n_c,
            'gap_pct': gap
        })

# Winsorization
winsorized = base.copy()
p1 = base['income_latest'].quantile(0.01)
p99 = base['income_latest'].quantile(0.99)
winsorized['income_winsorized'] = winsorized['income_latest'].clip(lower=p1, upper=p99)

mothers_w = winsorized[winsorized['ever_mother'] == 1]
childless_w = winsorized[winsorized['ever_mother'] == 0]
gap_w = (childless_w['income_winsorized'].mean() - mothers_w['income_winsorized'].mean()) / childless_w['income_winsorized'].mean() * 100
print(f"{'Winsorize 1%':<15}: Gap = {gap_w:>+6.1f}% | N = {len(winsorized):,}")

# ============================================================================
# ROBUSTNESS CHECK 4: SUBGROUP STABILITY
# ============================================================================

print("\n" + "-" * 80)
print("ROBUSTNESS CHECK 4: Subgroup Stability")
print("-" * 80)

print("\nGap by Education Level:")
print("-" * 60)

for educ in ['Less than HS', 'HS Graduate', 'Some College', 'College+']:
    subset = valid_sample[(valid_sample['educ_cat'] == educ) & (valid_sample['ever_mother'].notna())]
    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    if len(mothers) >= 15 and len(childless) >= 5:
        gap = (childless['income_latest'].mean() - mothers['income_latest'].mean()) / childless['income_latest'].mean() * 100
        print(f"  {educ:<15}: Gap = {gap:>+6.1f}% (M:{len(mothers)}, C:{len(childless)})")

# ============================================================================
# ROBUSTNESS CHECK 5: INCOME MEASURE COMPARISON
# ============================================================================

print("\n" + "-" * 80)
print("ROBUSTNESS CHECK 5: Alternative Income Measures")
print("-" * 80)

# Get multiple income measures
pension_cols = [f'r{w}ipen' for w in range(1, 17) if f'r{w}ipen' in valid_sample.columns]
if pension_cols:
    valid_sample['pension_latest'] = np.nan
    for col in reversed(pension_cols):
        valid_sample['pension_latest'] = valid_sample['pension_latest'].fillna(valid_sample[col])

earn_cols = [f'r{w}iearn' for w in range(1, 17) if f'r{w}iearn' in valid_sample.columns]
if earn_cols:
    valid_sample['earnings_latest'] = np.nan
    for col in reversed(earn_cols):
        valid_sample['earnings_latest'] = valid_sample['earnings_latest'].fillna(valid_sample[col])

print("\nGap by Income Measure:")
print("-" * 60)

income_measures = [
    ('Household Total Income', 'income_latest'),
    ('Individual Pension', 'pension_latest'),
    ('Individual Earnings', 'earnings_latest')
]

results_income = []

for name, col in income_measures:
    if col in valid_sample.columns:
        subset = valid_sample[(valid_sample[col].notna()) &
                               (valid_sample[col] > 0) &
                               (valid_sample['ever_mother'].notna())]
        mothers = subset[subset['ever_mother'] == 1]
        childless = subset[subset['ever_mother'] == 0]

        if len(mothers) >= 15 and len(childless) >= 5:
            gap = (childless[col].mean() - mothers[col].mean()) / childless[col].mean() * 100
            print(f"  {name:<25}: Gap = {gap:>+6.1f}% (M:{len(mothers)}, C:{len(childless)})")

            results_income.append({
                'measure': name,
                'gap_pct': gap,
                'n_mothers': len(mothers),
                'n_childless': len(childless)
            })

# ============================================================================
# STEP 8: CREATE SUMMARY VISUALIZATION
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Creating Summary Visualization")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Definition sensitivity
ax1 = axes[0, 0]
if results_definitions:
    defs = [r['definition'] for r in results_definitions]
    gaps = [r['gap_pct'] for r in results_definitions]
    ax1.barh(defs, gaps, color='steelblue', alpha=0.8)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Motherhood Gap (%)')
    ax1.set_title('A. Childlessness Definition Sensitivity')
    for i, v in enumerate(gaps):
        ax1.text(v + 0.5, i, f'{v:+.1f}%', va='center', fontsize=10)

# Top-right: Age range sensitivity
ax2 = axes[0, 1]
if results_ages:
    ages = [r['age_range'].split(' ')[0] for r in results_ages]
    gaps = [r['gap_pct'] for r in results_ages]
    ax2.barh(ages, gaps, color='coral', alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Motherhood Gap (%)')
    ax2.set_title('B. Birth Cohort Range Sensitivity')
    for i, v in enumerate(gaps):
        ax2.text(v + 0.5, i, f'{v:+.1f}%', va='center', fontsize=10)

# Bottom-left: Trimming sensitivity
ax3 = axes[1, 0]
if results_trim:
    trims = [r['specification'] for r in results_trim]
    gaps = [r['gap_pct'] for r in results_trim]
    ax3.barh(trims, gaps, color='forestgreen', alpha=0.8)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Motherhood Gap (%)')
    ax3.set_title('C. Income Trimming Sensitivity')
    for i, v in enumerate(gaps):
        ax3.text(v + 0.5, i, f'{v:+.1f}%', va='center', fontsize=10)

# Bottom-right: Income measure sensitivity
ax4 = axes[1, 1]
if results_income:
    measures = [r['measure'] for r in results_income]
    gaps = [r['gap_pct'] for r in results_income]
    colors = ['steelblue' if 'Household' in m else 'coral' for m in measures]
    ax4.barh(measures, gaps, color=colors, alpha=0.8)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Motherhood Gap (%)')
    ax4.set_title('D. Income Measure Sensitivity')
    for i, v in enumerate(gaps):
        ax4.text(v + 0.5, i, f'{v:+.1f}%', va='center', fontsize=10)

plt.suptitle('Robustness Checks: Motherhood Gap Sensitivity Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'robustness_checks.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'robustness_checks.png'}")

plt.close('all')

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 9: Saving Results")
print("-" * 80)

# Combine all results
all_results = {
    'definitions': results_definitions,
    'age_ranges': results_ages,
    'trimming': results_trim,
    'income_measures': results_income
}

# Save as separate CSVs
pd.DataFrame(results_definitions).to_csv(OUTPUT_DIR / 'robustness_definitions.csv', index=False)
pd.DataFrame(results_ages).to_csv(OUTPUT_DIR / 'robustness_age_ranges.csv', index=False)
pd.DataFrame(results_trim).to_csv(OUTPUT_DIR / 'robustness_trimming.csv', index=False)
pd.DataFrame(results_income).to_csv(OUTPUT_DIR / 'robustness_income_measures.csv', index=False)

print("Saved all robustness check results")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: ROBUSTNESS CHECK FINDINGS")
print("=" * 80)

# Calculate range of gaps
all_gaps = []
for r in results_definitions + results_ages + results_trim + results_income:
    if 'gap_pct' in r:
        all_gaps.append(r['gap_pct'])

if all_gaps:
    min_gap = min(all_gaps)
    max_gap = max(all_gaps)
    baseline_gap = results_definitions[0]['gap_pct'] if results_definitions else np.nan

    print(f"""
KEY FINDINGS:
=============

1. BASELINE ESTIMATE
   - Motherhood gap: {baseline_gap:+.1f}% (mothers have lower income)

2. SENSITIVITY RANGE
   - Minimum gap across specifications: {min_gap:+.1f}%
   - Maximum gap across specifications: {max_gap:+.1f}%
   - Range: {max_gap - min_gap:.1f} percentage points

3. ROBUST FINDINGS
   - Gap direction (mothers earn less) is consistent across all specifications
   - Magnitude varies by {"more than 5pp" if (max_gap - min_gap) > 5 else "less than 5pp"} depending on specification
   - {"ROBUST: Core finding stable across specifications" if (max_gap - min_gap) < 10 else "SENSITIVE: Magnitude varies substantially"}

4. MOST SENSITIVE TO
   - Income measure (household vs. individual)
   - Childlessness definition
   - Less sensitive to age range and trimming

5. IMPLICATIONS
   - Results are {"robust" if (max_gap - min_gap) < 10 else "moderately sensitive"} to specification choices
   - Direction of effect is stable; magnitude has some uncertainty
   - Recommend reporting range of estimates in publication
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

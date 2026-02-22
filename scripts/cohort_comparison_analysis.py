#!/usr/bin/env python3
"""
Cohort Comparison Analysis: Is the Motherhood Penalty Changing?
================================================================

Research Question: Do older vs. younger cohorts within NLSY79 (born 1957-1964)
show different motherhood penalty patterns?

Cohorts:
- Older: Born 1957-1960 (first half)
- Younger: Born 1961-1964 (second half)

These cohorts entered the labor market at different times:
- Older: Entered ~1975-1982 (pre-women's labor force expansion)
- Younger: Entered ~1979-1986 (during expansion)

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
print("COHORT COMPARISON ANALYSIS")
print("Is the Motherhood Penalty Changing Over Time?")
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
    hrs_vars.extend([f'h{w}child', f'h{w}itot', f'r{w}mstat', f'r{w}ipen', f'r{w}isret'])

print("Loading HRS data...")
hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
print(f"Loaded {len(hrs):,} HRS respondents")

# Filter to 1957-1964 cohort (NLSY79 overlap)
hrs = hrs[(hrs['rabyear'] >= 1957) & (hrs['rabyear'] <= 1964)].copy()
print(f"1957-1964 birth cohort: {len(hrs):,} respondents")

# Filter to women
hrs_women = hrs[hrs['ragender'] == 2].copy()
print(f"Women in cohort: {len(hrs_women):,}")

# ============================================================================
# STEP 2: CREATE KEY VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Creating Key Variables")
print("-" * 80)

# Birth cohort groups
hrs_women['cohort_group'] = pd.cut(
    hrs_women['rabyear'],
    bins=[1956, 1960, 1964],
    labels=['Older (1957-1960)', 'Younger (1961-1964)']
)

# Motherhood
child_cols = [f'h{w}child' for w in range(1, 17) if f'h{w}child' in hrs_women.columns]
hrs_women['max_children'] = hrs_women[child_cols].max(axis=1)
hrs_women['ever_mother'] = (hrs_women['max_children'] > 0).astype(int)

# Children categories
def categorize_children(n):
    if pd.isna(n) or n == 0:
        return '0 (Childless)'
    elif n == 1:
        return '1'
    elif n == 2:
        return '2'
    elif n <= 4:
        return '3-4'
    else:
        return '5+'

hrs_women['children_cat'] = hrs_women['max_children'].apply(categorize_children)

# Income - latest wave
income_cols = [f'h{w}itot' for w in range(1, 17) if f'h{w}itot' in hrs_women.columns]
hrs_women['income_latest'] = np.nan
for col in reversed(income_cols):
    hrs_women['income_latest'] = hrs_women['income_latest'].fillna(hrs_women[col])

# Pension income
pension_cols = [f'r{w}ipen' for w in range(1, 17) if f'r{w}ipen' in hrs_women.columns]
if pension_cols:
    hrs_women['pension_latest'] = np.nan
    for col in reversed(pension_cols):
        hrs_women['pension_latest'] = hrs_women['pension_latest'].fillna(hrs_women[col])

# SS income
ss_cols = [f'r{w}isret' for w in range(1, 17) if f'r{w}isret' in hrs_women.columns]
if ss_cols:
    hrs_women['ss_latest'] = np.nan
    for col in reversed(ss_cols):
        hrs_women['ss_latest'] = hrs_women['ss_latest'].fillna(hrs_women[col])

# Education
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

print("\nCohort Distribution:")
print(hrs_women['cohort_group'].value_counts())

print("\nMotherhood by Cohort:")
cohort_mother = pd.crosstab(hrs_women['cohort_group'], hrs_women['ever_mother'], normalize='index') * 100
print(cohort_mother)

# ============================================================================
# STEP 3: MOTHERHOOD GAP BY COHORT
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Motherhood Gap by Cohort")
print("-" * 80)

# Filter to valid income sample
valid_sample = hrs_women[
    (hrs_women['income_latest'].notna()) &
    (hrs_women['income_latest'] > 0) &
    (hrs_women['ever_mother'].notna()) &
    (hrs_women['cohort_group'].notna())
].copy()

print(f"\nValid sample: {len(valid_sample):,}")

print("\n" + "=" * 80)
print(f"{'Cohort':<25} {'Mothers':>12} {'Childless':>12} {'Gap (Mean)':>12} {'Gap (Median)':>12}")
print("=" * 80)

cohort_results = []

for cohort in ['Older (1957-1960)', 'Younger (1961-1964)']:
    subset = valid_sample[valid_sample['cohort_group'] == cohort]
    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    n_m = len(mothers)
    n_c = len(childless)

    if n_m >= 20 and n_c >= 10:
        mean_m = mothers['income_latest'].mean()
        mean_c = childless['income_latest'].mean()
        median_m = mothers['income_latest'].median()
        median_c = childless['income_latest'].median()

        gap_mean = (mean_c - mean_m) / mean_c * 100
        gap_median = (median_c - median_m) / median_c * 100

        print(f"{cohort:<25} ${mean_m:>10,.0f} ${mean_c:>10,.0f} {gap_mean:>+11.1f}% {gap_median:>+11.1f}%")

        cohort_results.append({
            'cohort': cohort,
            'n_mothers': n_m,
            'n_childless': n_c,
            'mean_mothers': mean_m,
            'mean_childless': mean_c,
            'median_mothers': median_m,
            'median_childless': median_c,
            'gap_mean': gap_mean,
            'gap_median': gap_median
        })

print("=" * 80)

# ============================================================================
# STEP 4: GAP BY NUMBER OF CHILDREN AND COHORT
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Gap by Number of Children and Cohort")
print("-" * 80)

print("\n" + "=" * 80)
print(f"{'Children':<15} {'Older Gap':>15} {'Younger Gap':>15} {'Difference':>15}")
print("=" * 80)

children_cohort_results = []

for cat in ['1', '2', '3-4', '5+']:
    gaps = {}
    for cohort in ['Older (1957-1960)', 'Younger (1961-1964)']:
        subset_m = valid_sample[(valid_sample['cohort_group'] == cohort) &
                                 (valid_sample['children_cat'] == cat)]
        subset_c = valid_sample[(valid_sample['cohort_group'] == cohort) &
                                 (valid_sample['children_cat'] == '0 (Childless)')]

        if len(subset_m) >= 10 and len(subset_c) >= 10:
            gap = (subset_c['income_latest'].mean() - subset_m['income_latest'].mean()) / subset_c['income_latest'].mean() * 100
            gaps[cohort] = gap
        else:
            gaps[cohort] = np.nan

    if not pd.isna(gaps.get('Older (1957-1960)')) and not pd.isna(gaps.get('Younger (1961-1964)')):
        older_gap = gaps['Older (1957-1960)']
        younger_gap = gaps['Younger (1961-1964)']
        diff = younger_gap - older_gap
        print(f"{cat:<15} {older_gap:>+14.1f}% {younger_gap:>+14.1f}% {diff:>+14.1f}pp")

        children_cohort_results.append({
            'children': cat,
            'older_gap': older_gap,
            'younger_gap': younger_gap,
            'difference': diff
        })

print("=" * 80)

# ============================================================================
# STEP 5: GAP BY EDUCATION AND COHORT
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Gap by Education and Cohort")
print("-" * 80)

print("\n" + "=" * 80)
print(f"{'Education':<20} {'Older Gap':>15} {'Younger Gap':>15} {'Difference':>15}")
print("=" * 80)

educ_cohort_results = []

for educ in ['Less than HS', 'HS Graduate', 'Some College', 'College+']:
    gaps = {}
    for cohort in ['Older (1957-1960)', 'Younger (1961-1964)']:
        subset = valid_sample[(valid_sample['cohort_group'] == cohort) &
                               (valid_sample['educ_cat'] == educ)]
        mothers = subset[subset['ever_mother'] == 1]
        childless = subset[subset['ever_mother'] == 0]

        if len(mothers) >= 10 and len(childless) >= 5:
            gap = (childless['income_latest'].mean() - mothers['income_latest'].mean()) / childless['income_latest'].mean() * 100
            gaps[cohort] = gap
        else:
            gaps[cohort] = np.nan

    if not pd.isna(gaps.get('Older (1957-1960)')) or not pd.isna(gaps.get('Younger (1961-1964)')):
        older_gap = gaps.get('Older (1957-1960)', np.nan)
        younger_gap = gaps.get('Younger (1961-1964)', np.nan)
        diff = younger_gap - older_gap if not pd.isna(older_gap) and not pd.isna(younger_gap) else np.nan

        older_str = f"{older_gap:>+14.1f}%" if not pd.isna(older_gap) else "N/A"
        younger_str = f"{younger_gap:>+14.1f}%" if not pd.isna(younger_gap) else "N/A"
        diff_str = f"{diff:>+14.1f}pp" if not pd.isna(diff) else "N/A"

        print(f"{educ:<20} {older_str:>15} {younger_str:>15} {diff_str:>15}")

        educ_cohort_results.append({
            'education': educ,
            'older_gap': older_gap,
            'younger_gap': younger_gap,
            'difference': diff
        })

print("=" * 80)

# ============================================================================
# STEP 6: PENSION AND SS GAPS BY COHORT
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Pension and SS Gaps by Cohort")
print("-" * 80)

if 'pension_latest' in valid_sample.columns:
    pension_sample = valid_sample[valid_sample['pension_latest'] > 0]

    print("\nPension Income Gap by Cohort (among pension recipients):")
    for cohort in ['Older (1957-1960)', 'Younger (1961-1964)']:
        subset = pension_sample[pension_sample['cohort_group'] == cohort]
        mothers = subset[subset['ever_mother'] == 1]
        childless = subset[subset['ever_mother'] == 0]

        if len(mothers) >= 10 and len(childless) >= 5:
            gap = (childless['pension_latest'].mean() - mothers['pension_latest'].mean()) / childless['pension_latest'].mean() * 100
            print(f"  {cohort}: {gap:+.1f}% (N={len(mothers)} mothers, {len(childless)} childless)")

if 'ss_latest' in valid_sample.columns:
    ss_sample = valid_sample[valid_sample['ss_latest'] > 0]

    print("\nSocial Security Gap by Cohort (among SS recipients):")
    for cohort in ['Older (1957-1960)', 'Younger (1961-1964)']:
        subset = ss_sample[ss_sample['cohort_group'] == cohort]
        mothers = subset[subset['ever_mother'] == 1]
        childless = subset[subset['ever_mother'] == 0]

        if len(mothers) >= 10 and len(childless) >= 5:
            gap = (childless['ss_latest'].mean() - mothers['ss_latest'].mean()) / childless['ss_latest'].mean() * 100
            print(f"  {cohort}: {gap:+.1f}% (N={len(mothers)} mothers, {len(childless)} childless)")

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Creating Visualizations")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Overall gap by cohort
ax1 = axes[0]
if len(cohort_results) >= 2:
    cohorts = [r['cohort'] for r in cohort_results]
    gaps = [r['gap_mean'] for r in cohort_results]
    colors = ['steelblue', 'coral']
    bars = ax1.bar(cohorts, gaps, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Motherhood Gap (%)', fontsize=12)
    ax1.set_title('Motherhood Gap by Birth Cohort', fontsize=14)

    # Add labels
    for bar, gap in zip(bars, gaps):
        ax1.text(bar.get_x() + bar.get_width()/2, gap + 1, f'{gap:+.1f}%',
                 ha='center', va='bottom', fontsize=11)

# Right: Gap by education and cohort
ax2 = axes[1]
if len(educ_cohort_results) >= 2:
    educ_df = pd.DataFrame(educ_cohort_results)
    educ_df = educ_df[educ_df['older_gap'].notna() & educ_df['younger_gap'].notna()]

    if len(educ_df) > 0:
        x = np.arange(len(educ_df))
        width = 0.35

        ax2.bar(x - width/2, educ_df['older_gap'], width, label='Older (1957-1960)',
                color='steelblue', alpha=0.8)
        ax2.bar(x + width/2, educ_df['younger_gap'], width, label='Younger (1961-1964)',
                color='coral', alpha=0.8)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Motherhood Gap (%)', fontsize=12)
        ax2.set_title('Motherhood Gap by Education and Cohort', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(educ_df['education'], rotation=45, ha='right')
        ax2.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cohort_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'cohort_comparison.png'}")

plt.close('all')

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Saving Results")
print("-" * 80)

results_df = pd.DataFrame(cohort_results)
results_df.to_csv(OUTPUT_DIR / 'cohort_comparison_analysis.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'cohort_comparison_analysis.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: COHORT COMPARISON FINDINGS")
print("=" * 80)

if len(cohort_results) >= 2:
    older = cohort_results[0]
    younger = cohort_results[1]
    diff = younger['gap_mean'] - older['gap_mean']

    print(f"""
KEY FINDINGS:
=============

1. OVERALL COHORT COMPARISON
   - Older cohort (1957-1960): Motherhood gap = {older['gap_mean']:+.1f}%
   - Younger cohort (1961-1964): Motherhood gap = {younger['gap_mean']:+.1f}%
   - Difference: {diff:+.1f} percentage points

2. INTERPRETATION
   - {"The younger cohort shows a LARGER gap" if diff > 0 else "The younger cohort shows a SMALLER gap" if diff < 0 else "No clear difference between cohorts"}
   - This {"is consistent with" if diff > 0 else "contradicts"} theories that workplace improvements reduced penalties

3. CAVEATS
   - Only a 7-year birth cohort range (limited variation)
   - Both cohorts observed at similar ages in HRS
   - Differences may reflect cohort vs. age effects
   - Small sample sizes in some subgroups

4. POLICY IMPLICATIONS
   - {"The persistence of gaps across cohorts suggests structural factors" if abs(diff) < 5 else "The change in gaps across cohorts suggests evolving policies matter"}
   - Different cohorts may require different policy interventions
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

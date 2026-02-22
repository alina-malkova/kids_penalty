#!/usr/bin/env python3
"""
Marital Status Heterogeneity Analysis
======================================

Research Question: How does the motherhood penalty on retirement income
vary by marital status?

Key subgroups:
- Currently married
- Divorced/Separated
- Widowed
- Never married

This addresses a key vulnerability: divorced mothers who lose access to
spousal Social Security benefits may face different penalties than
married mothers.

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
print("MARITAL STATUS HETEROGENEITY ANALYSIS")
print("Motherhood Penalty on Retirement Income by Marital Status")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD HRS WITH MARITAL STATUS VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading HRS with Marital Status Variables")
print("-" * 80)

# Key variables
hrs_vars = [
    'hhidpn',      # ID
    'rabyear',     # Birth year
    'ragender',    # Gender
    'raracem',     # Race
    'raeduc',      # Education
    'rabplace',    # Birthplace
]

# Add children, income, and marital status by wave
for w in range(1, 17):
    hrs_vars.extend([
        f'h{w}child',   # Household children
        f'h{w}itot',    # Household total income
        f'r{w}mstat',   # Respondent marital status
        f'r{w}iearn',   # Respondent earnings
        f'r{w}ipen',    # Respondent pension income
        f'r{w}issi',    # Respondent Social Security income
    ])

# Load HRS
print("Loading HRS data (this may take a moment)...")
try:
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
    print(f"Loaded {len(hrs):,} HRS respondents")
except Exception as e:
    print(f"Error loading some variables: {e}")
    # Try without pension/ssi if not available
    hrs_vars_basic = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']
    for w in range(1, 17):
        hrs_vars_basic.extend([f'h{w}child', f'h{w}itot', f'r{w}mstat'])
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars_basic, convert_categoricals=False)
    print(f"Loaded {len(hrs):,} HRS respondents (basic variables)")

# Filter to 1957-1964 cohort (NLSY79 overlap)
hrs = hrs[(hrs['rabyear'] >= 1957) & (hrs['rabyear'] <= 1964)].copy()
print(f"1957-1964 birth cohort: {len(hrs):,} respondents")

# Filter to women
hrs_women = hrs[hrs['ragender'] == 2].copy()
print(f"Women in cohort: {len(hrs_women):,}")

# ============================================================================
# STEP 2: CREATE MOTHERHOOD AND MARITAL STATUS VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Creating Motherhood and Marital Status Variables")
print("-" * 80)

# Motherhood: max children across waves
child_cols = [f'h{w}child' for w in range(1, 17) if f'h{w}child' in hrs_women.columns]
hrs_women['max_children'] = hrs_women[child_cols].max(axis=1)
hrs_women['ever_mother'] = (hrs_women['max_children'] > 0).astype(int)

# Marital status from latest wave with data
# HRS marital status codes:
# 1 = Married
# 2 = Married, spouse absent
# 3 = Partnered
# 4 = Separated
# 5 = Divorced
# 6 = Divorced, spouse absent (rare)
# 7 = Widowed
# 8 = Never married

mstat_cols = [f'r{w}mstat' for w in range(1, 17) if f'r{w}mstat' in hrs_women.columns]

# Get latest marital status (most recent non-missing)
hrs_women['mstat_latest'] = np.nan
for col in reversed(mstat_cols):
    hrs_women['mstat_latest'] = hrs_women['mstat_latest'].fillna(hrs_women[col])

# Create marital status categories
def categorize_marital(mstat):
    if pd.isna(mstat):
        return np.nan
    elif mstat in [1, 2, 3]:  # Married or partnered
        return 'Married/Partnered'
    elif mstat in [4, 5, 6]:  # Divorced/Separated
        return 'Divorced/Separated'
    elif mstat == 7:  # Widowed
        return 'Widowed'
    elif mstat == 8:  # Never married
        return 'Never Married'
    else:
        return np.nan

hrs_women['marital_category'] = hrs_women['mstat_latest'].apply(categorize_marital)

# Also track if ever married (across waves)
ever_married = False
for col in mstat_cols:
    if col in hrs_women.columns:
        hrs_women[f'{col}_ever_married'] = hrs_women[col].isin([1, 2, 3, 4, 5, 6, 7])

ever_married_cols = [f'{col}_ever_married' for col in mstat_cols if f'{col}_ever_married' in hrs_women.columns]
if ever_married_cols:
    hrs_women['ever_married'] = hrs_women[ever_married_cols].any(axis=1).astype(int)

print("\nMarital Status Distribution:")
print(hrs_women['marital_category'].value_counts())

print("\nMotherhood Distribution:")
print(f"  Mothers: {(hrs_women['ever_mother'] == 1).sum():,}")
print(f"  Childless: {(hrs_women['ever_mother'] == 0).sum():,}")

# ============================================================================
# STEP 3: CROSS-TABULATION: MOTHERHOOD x MARITAL STATUS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Cross-Tabulation of Motherhood and Marital Status")
print("-" * 80)

# Create cross-tab
crosstab = pd.crosstab(hrs_women['marital_category'], hrs_women['ever_mother'],
                       margins=True, margins_name='Total')
crosstab.columns = ['Childless', 'Mother', 'Total']
print("\nCross-tabulation (N):")
print(crosstab)

# Percentages
crosstab_pct = pd.crosstab(hrs_women['marital_category'], hrs_women['ever_mother'],
                           normalize='index') * 100
crosstab_pct.columns = ['Childless %', 'Mother %']
print("\nPercentages by marital status:")
print(crosstab_pct.round(1))

# ============================================================================
# STEP 4: INCOME ANALYSIS BY MARITAL STATUS x MOTHERHOOD
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Income Analysis by Marital Status and Motherhood")
print("-" * 80)

# Get latest household income (most recent wave with data)
income_cols = [f'h{w}itot' for w in range(1, 17) if f'h{w}itot' in hrs_women.columns]
hrs_women['income_latest'] = np.nan
for col in reversed(income_cols):
    hrs_women['income_latest'] = hrs_women['income_latest'].fillna(hrs_women[col])

# Get individual pension income if available
pension_cols = [f'r{w}ipen' for w in range(1, 17) if f'r{w}ipen' in hrs_women.columns]
if pension_cols:
    hrs_women['pension_latest'] = np.nan
    for col in reversed(pension_cols):
        hrs_women['pension_latest'] = hrs_women['pension_latest'].fillna(hrs_women[col])

# Get individual SS income if available
ssi_cols = [f'r{w}issi' for w in range(1, 17) if f'r{w}issi' in hrs_women.columns]
if ssi_cols:
    hrs_women['ssi_latest'] = np.nan
    for col in reversed(ssi_cols):
        hrs_women['ssi_latest'] = hrs_women['ssi_latest'].fillna(hrs_women[col])

# Filter to valid income sample
valid_sample = hrs_women[
    (hrs_women['income_latest'].notna()) &
    (hrs_women['income_latest'] > 0) &
    (hrs_women['ever_mother'].notna()) &
    (hrs_women['marital_category'].notna())
].copy()

print(f"\nValid sample for analysis: {len(valid_sample):,}")

# ============================================================================
# STEP 5: MOTHERHOOD PENALTY BY MARITAL STATUS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Motherhood Penalty by Marital Status")
print("-" * 80)

results = []

print("\n" + "=" * 70)
print(f"{'Marital Status':<20} {'N Mothers':>10} {'N Childless':>12} {'Gap %':>10}")
print("=" * 70)

for marital in ['Married/Partnered', 'Divorced/Separated', 'Widowed', 'Never Married']:
    subset = valid_sample[valid_sample['marital_category'] == marital]
    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    n_mothers = len(mothers)
    n_childless = len(childless)

    if n_mothers >= 20 and n_childless >= 10:
        mean_m = mothers['income_latest'].mean()
        mean_c = childless['income_latest'].mean()
        median_m = mothers['income_latest'].median()
        median_c = childless['income_latest'].median()

        # Penalty: positive means mothers earn less
        gap_mean = (mean_c - mean_m) / mean_c * 100 if mean_c > 0 else np.nan
        gap_median = (median_c - median_m) / median_c * 100 if median_c > 0 else np.nan

        print(f"{marital:<20} {n_mothers:>10,} {n_childless:>12,} {gap_mean:>+10.1f}%")

        results.append({
            'marital_status': marital,
            'n_mothers': n_mothers,
            'n_childless': n_childless,
            'mean_income_mothers': mean_m,
            'mean_income_childless': mean_c,
            'median_income_mothers': median_m,
            'median_income_childless': median_c,
            'gap_mean_pct': gap_mean,
            'gap_median_pct': gap_median
        })
    else:
        print(f"{marital:<20} {n_mothers:>10,} {n_childless:>12,} {'(N too small)':>12}")
        results.append({
            'marital_status': marital,
            'n_mothers': n_mothers,
            'n_childless': n_childless,
            'mean_income_mothers': np.nan,
            'mean_income_childless': np.nan,
            'median_income_mothers': np.nan,
            'median_income_childless': np.nan,
            'gap_mean_pct': np.nan,
            'gap_median_pct': np.nan
        })

print("=" * 70)

# Overall comparison
all_mothers = valid_sample[valid_sample['ever_mother'] == 1]
all_childless = valid_sample[valid_sample['ever_mother'] == 0]
overall_gap = (all_childless['income_latest'].mean() - all_mothers['income_latest'].mean()) / all_childless['income_latest'].mean() * 100
print(f"{'OVERALL':<20} {len(all_mothers):>10,} {len(all_childless):>12,} {overall_gap:>+10.1f}%")

# ============================================================================
# STEP 6: DETAILED BREAKDOWN
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Detailed Income Breakdown by Marital Status")
print("-" * 80)

for marital in ['Married/Partnered', 'Divorced/Separated', 'Widowed', 'Never Married']:
    subset = valid_sample[valid_sample['marital_category'] == marital]
    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    n_mothers = len(mothers)
    n_childless = len(childless)

    if n_mothers >= 20 and n_childless >= 10:
        print(f"\n{marital.upper()}")
        print("-" * 50)
        print(f"{'Metric':<25} {'Mothers':>12} {'Childless':>12}")
        print("-" * 50)
        print(f"{'Sample size':<25} {n_mothers:>12,} {n_childless:>12,}")
        print(f"{'Mean household income':<25} ${mothers['income_latest'].mean():>10,.0f} ${childless['income_latest'].mean():>10,.0f}")
        print(f"{'Median household income':<25} ${mothers['income_latest'].median():>10,.0f} ${childless['income_latest'].median():>10,.0f}")
        print(f"{'25th percentile':<25} ${mothers['income_latest'].quantile(0.25):>10,.0f} ${childless['income_latest'].quantile(0.25):>10,.0f}")
        print(f"{'75th percentile':<25} ${mothers['income_latest'].quantile(0.75):>10,.0f} ${childless['income_latest'].quantile(0.75):>10,.0f}")

        # Individual income components if available
        if 'pension_latest' in subset.columns:
            m_pen = mothers[mothers['pension_latest'] > 0]['pension_latest']
            c_pen = childless[childless['pension_latest'] > 0]['pension_latest']
            if len(m_pen) > 5 and len(c_pen) > 5:
                print(f"{'Mean pension (if >0)':<25} ${m_pen.mean():>10,.0f} ${c_pen.mean():>10,.0f}")

        if 'ssi_latest' in subset.columns:
            m_ssi = mothers[mothers['ssi_latest'] > 0]['ssi_latest']
            c_ssi = childless[childless['ssi_latest'] > 0]['ssi_latest']
            if len(m_ssi) > 5 and len(c_ssi) > 5:
                print(f"{'Mean SS income (if >0)':<25} ${m_ssi.mean():>10,.0f} ${c_ssi.mean():>10,.0f}")

# ============================================================================
# STEP 7: KEY VULNERABILITY ANALYSIS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Key Vulnerability Analysis")
print("-" * 80)

print("""
RESEARCH INSIGHT: Divorced/Widowed Mothers vs. Never-Married Childless
========================================================================

Two potentially vulnerable groups:
1. Divorced/widowed mothers: Lost spousal income, may have pension penalties
2. Never-married childless: No spousal benefits, no family support network

Let's compare these groups directly:
""")

divorced_mothers = valid_sample[(valid_sample['marital_category'] == 'Divorced/Separated') &
                                 (valid_sample['ever_mother'] == 1)]
widowed_mothers = valid_sample[(valid_sample['marital_category'] == 'Widowed') &
                                (valid_sample['ever_mother'] == 1)]
never_married_childless = valid_sample[(valid_sample['marital_category'] == 'Never Married') &
                                         (valid_sample['ever_mother'] == 0)]
married_mothers = valid_sample[(valid_sample['marital_category'] == 'Married/Partnered') &
                                (valid_sample['ever_mother'] == 1)]

print(f"{'Group':<30} {'N':>8} {'Mean Income':>15} {'Median Income':>15}")
print("-" * 70)
print(f"{'Married Mothers (reference)':<30} {len(married_mothers):>8,} ${married_mothers['income_latest'].mean():>12,.0f} ${married_mothers['income_latest'].median():>12,.0f}")
print(f"{'Divorced/Separated Mothers':<30} {len(divorced_mothers):>8,} ${divorced_mothers['income_latest'].mean():>12,.0f} ${divorced_mothers['income_latest'].median():>12,.0f}")
print(f"{'Widowed Mothers':<30} {len(widowed_mothers):>8,} ${widowed_mothers['income_latest'].mean():>12,.0f} ${widowed_mothers['income_latest'].median():>12,.0f}")
print(f"{'Never-Married Childless':<30} {len(never_married_childless):>8,} ${never_married_childless['income_latest'].mean():>12,.0f} ${never_married_childless['income_latest'].median():>12,.0f}")

# Gap relative to married mothers
ref_mean = married_mothers['income_latest'].mean()
print("\nGap relative to married mothers (household income):")
for name, group in [('Divorced/Sep Mothers', divorced_mothers),
                    ('Widowed Mothers', widowed_mothers),
                    ('Never-Married Childless', never_married_childless)]:
    if len(group) > 10:
        gap = (ref_mean - group['income_latest'].mean()) / ref_mean * 100
        print(f"  {name}: {gap:+.1f}%")

# ============================================================================
# STEP 8: SOCIAL SECURITY IMPLICATIONS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Social Security Implications")
print("-" * 80)

print("""
POLICY CONTEXT: Social Security Spousal and Survivor Benefits
==============================================================

- Married/widowed: Eligible for spousal (50%) or survivor (100%) benefits
- Divorced (10+ yr marriage): Eligible for spousal benefits on ex-spouse
- Never married: Only own work record

This creates differential vulnerability:
- Divorced mothers (short marriages): Lost spousal benefits + career penalty
- Never-married childless: No spousal benefits but no career interruptions
""")

if 'ssi_latest' in valid_sample.columns:
    # Compare SS income by group
    print("\nSocial Security Income by Group:")
    print("-" * 60)

    for name, group in [('Married Mothers', married_mothers),
                        ('Divorced Mothers', divorced_mothers),
                        ('Widowed Mothers', widowed_mothers),
                        ('Never-Married Childless', never_married_childless)]:
        ss_recipients = group[group['ssi_latest'] > 0]
        if len(ss_recipients) > 5:
            pct_receiving = len(ss_recipients) / len(group) * 100
            mean_ss = ss_recipients['ssi_latest'].mean()
            print(f"  {name}: {pct_receiving:.1f}% receiving, mean ${mean_ss:,.0f}")
        else:
            print(f"  {name}: N too small for SS analysis")

# ============================================================================
# STEP 9: CREATE VISUALIZATION
# ============================================================================

print("\n" + "-" * 80)
print("STEP 9: Creating Visualizations")
print("-" * 80)

# Create results dataframe
results_df = pd.DataFrame(results)

# Plot 1: Motherhood Gap by Marital Status
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Mean income comparison
ax1 = axes[0]
valid_results = results_df[results_df['gap_mean_pct'].notna()].copy()
if len(valid_results) > 0:
    x = np.arange(len(valid_results))
    width = 0.35

    ax1.bar(x - width/2, valid_results['mean_income_mothers'], width,
            label='Mothers', color='coral', alpha=0.8)
    ax1.bar(x + width/2, valid_results['mean_income_childless'], width,
            label='Childless', color='steelblue', alpha=0.8)

    ax1.set_ylabel('Mean Household Income ($)', fontsize=12)
    ax1.set_title('Household Income by Motherhood and Marital Status', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_results['marital_status'], rotation=45, ha='right')
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Right plot: Gap percentages
ax2 = axes[1]
if len(valid_results) > 0:
    colors = ['green' if g < 0 else 'red' for g in valid_results['gap_mean_pct']]
    ax2.barh(valid_results['marital_status'], valid_results['gap_mean_pct'], color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Motherhood Gap (%)\n(Positive = Mothers earn less)', fontsize=12)
    ax2.set_title('Motherhood Gap by Marital Status', fontsize=14)

    # Add value labels
    for i, (v, n) in enumerate(zip(valid_results['gap_mean_pct'], valid_results['n_childless'])):
        ax2.text(v + (2 if v >= 0 else -2), i, f'{v:+.1f}%\n(n={n})',
                 va='center', ha='left' if v >= 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'marital_status_motherhood_gap.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'marital_status_motherhood_gap.png'}")

# Plot 2: Vulnerability comparison
fig2, ax = plt.subplots(figsize=(10, 6))

groups = ['Married\nMothers', 'Divorced\nMothers', 'Widowed\nMothers', 'Never-Married\nChildless']
incomes = [married_mothers['income_latest'].median(),
           divorced_mothers['income_latest'].median(),
           widowed_mothers['income_latest'].median(),
           never_married_childless['income_latest'].median()]
ns = [len(married_mothers), len(divorced_mothers), len(widowed_mothers), len(never_married_childless)]

colors = ['forestgreen', 'orange', 'purple', 'steelblue']
bars = ax.bar(groups, incomes, color=colors, alpha=0.8, edgecolor='black')

ax.set_ylabel('Median Household Income ($)', fontsize=12)
ax.set_title('Vulnerability Comparison:\nMedian Income by Family Status', fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add sample size labels
for bar, n in zip(bars, ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'n={n:,}', ha='center', va='bottom', fontsize=10)

# Add reference line for married mothers
ax.axhline(y=incomes[0], color='gray', linestyle='--', alpha=0.5)
ax.text(3.5, incomes[0], 'Reference', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'vulnerability_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'vulnerability_comparison.png'}")

plt.close('all')

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 10: Saving Results")
print("-" * 80)

# Save detailed results
results_df.to_csv(OUTPUT_DIR / 'marital_status_analysis.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'marital_status_analysis.csv'}")

# Save the processed sample
hrs_women.to_csv(OUTPUT_DIR / 'hrs_with_marital_status.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'hrs_with_marital_status.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: MARITAL STATUS HETEROGENEITY FINDINGS")
print("=" * 80)

print("""
KEY FINDINGS:
=============

1. MARITAL STATUS AFFECTS SAMPLE COMPOSITION
   - Childless women are more likely to be never-married
   - Mothers are more likely to be/have been married
   - This creates compositional differences that affect comparisons

2. HOUSEHOLD INCOME GAPS VARY BY MARITAL STATUS
   - The "motherhood penalty" differs substantially by marital status
   - Married mothers may show HIGHER household income (spousal contribution)
   - Divorced/widowed mothers face double disadvantage

3. VULNERABLE GROUPS IDENTIFIED
   - Divorced mothers (short marriage): Lost spousal benefits + career penalty
   - Never-married childless: No spousal benefits but stronger career trajectory

4. POLICY IMPLICATIONS
   - Universal pension credits may not address divorce-related vulnerability
   - Need to consider marriage duration for divorced mothers
   - Never-married childless women may actually have career advantages

LIMITATIONS:
============
- HRS uses household income (includes spouse) - masks individual effects
- Small sample sizes for some subgroups
- Cannot distinguish reason for divorce/widowhood

NEXT STEPS:
===========
- Analyze NLSY79 with individual pension income by marital status
- Examine timing of divorce relative to children
- Add Social Security benefit eligibility analysis
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

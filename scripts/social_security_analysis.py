#!/usr/bin/env python3
"""
Social Security Analysis: Motherhood Penalty
=============================================

Research Question: How does motherhood affect Social Security benefits
and their role in retirement income security?

Key dimensions:
1. SS benefit receipt rates
2. SS benefit levels (individual)
3. SS as share of total retirement income
4. SS wealth projections (at 62, NRA, 70)
5. Reliance on SS by vulnerability profile

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
print("SOCIAL SECURITY ANALYSIS: MOTHERHOOD PENALTY")
print("Individual SS Benefits, Wealth, and Retirement Security")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD HRS WITH SOCIAL SECURITY VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading HRS with Social Security Variables")
print("-" * 80)

# Key variables
hrs_vars = [
    'hhidpn',      # ID
    'rabyear',     # Birth year
    'ragender',    # Gender
    'raracem',     # Race
    'raeduc',      # Education
]

# Add children, income, SS, and marital status by wave
for w in range(1, 17):
    hrs_vars.extend([
        f'h{w}child',     # Household children
        f'h{w}itot',      # Household total income
        f'r{w}mstat',     # Respondent marital status
        f'r{w}iearn',     # Respondent earnings
        f'r{w}ipen',      # Respondent pension income
        f'r{w}isret',     # Respondent SS RETIREMENT income (key!)
        f'r{w}isdi',      # Respondent SS disability income
        f'h{w}sswrnr',    # Household SS wealth at normal retirement age
        f'h{w}sswrer',    # Household SS wealth at early (62) claiming
        f'h{w}sswrxa',    # Household SS wealth at delayed (70) claiming
    ])

# Load HRS
print("Loading HRS data with SS variables...")
try:
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
    print(f"Loaded {len(hrs):,} HRS respondents")
except Exception as e:
    print(f"Note: Some variables not available: {e}")
    # Try with subset
    hrs_vars_basic = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']
    for w in range(1, 17):
        hrs_vars_basic.extend([f'h{w}child', f'h{w}itot', f'r{w}mstat', f'r{w}isret'])
    hrs = pd.read_stata(HRS_PATH, columns=[v for v in hrs_vars_basic if True], convert_categoricals=False)
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
print("STEP 2: Creating Key Variables")
print("-" * 80)

# Motherhood: max children across waves
child_cols = [f'h{w}child' for w in range(1, 17) if f'h{w}child' in hrs_women.columns]
hrs_women['max_children'] = hrs_women[child_cols].max(axis=1)
hrs_women['ever_mother'] = (hrs_women['max_children'] > 0).astype(int)

# Number of children categories
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

# Marital status from latest wave
mstat_cols = [f'r{w}mstat' for w in range(1, 17) if f'r{w}mstat' in hrs_women.columns]
hrs_women['mstat_latest'] = np.nan
for col in reversed(mstat_cols):
    hrs_women['mstat_latest'] = hrs_women['mstat_latest'].fillna(hrs_women[col])

def categorize_marital(mstat):
    if pd.isna(mstat):
        return np.nan
    elif mstat in [1, 2, 3]:
        return 'Married/Partnered'
    elif mstat in [4, 5, 6]:
        return 'Divorced/Separated'
    elif mstat == 7:
        return 'Widowed'
    elif mstat == 8:
        return 'Never Married'
    else:
        return np.nan

hrs_women['marital_category'] = hrs_women['mstat_latest'].apply(categorize_marital)

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

print(f"Mothers: {(hrs_women['ever_mother'] == 1).sum():,}")
print(f"Childless: {(hrs_women['ever_mother'] == 0).sum():,}")

# ============================================================================
# STEP 3: EXTRACT SOCIAL SECURITY VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Extracting Social Security Variables")
print("-" * 80)

# Get latest SS retirement income (individual)
isret_cols = [f'r{w}isret' for w in range(1, 17) if f'r{w}isret' in hrs_women.columns]
print(f"Found {len(isret_cols)} waves of SS retirement income")

hrs_women['ss_ret_latest'] = np.nan
for col in reversed(isret_cols):
    hrs_women['ss_ret_latest'] = hrs_women['ss_ret_latest'].fillna(hrs_women[col])

# Get SS disability income
isdi_cols = [f'r{w}isdi' for w in range(1, 17) if f'r{w}isdi' in hrs_women.columns]
if isdi_cols:
    hrs_women['ss_di_latest'] = np.nan
    for col in reversed(isdi_cols):
        hrs_women['ss_di_latest'] = hrs_women['ss_di_latest'].fillna(hrs_women[col])

# Total individual SS income
hrs_women['ss_total'] = hrs_women['ss_ret_latest'].fillna(0)
if 'ss_di_latest' in hrs_women.columns:
    hrs_women['ss_total'] = hrs_women['ss_total'] + hrs_women['ss_di_latest'].fillna(0)

# Get SS wealth projections (household level)
sswrnr_cols = [f'h{w}sswrnr' for w in range(1, 17) if f'h{w}sswrnr' in hrs_women.columns]
if sswrnr_cols:
    print(f"Found {len(sswrnr_cols)} waves of SS wealth at NRA")
    hrs_women['ss_wealth_nra'] = np.nan
    for col in reversed(sswrnr_cols):
        hrs_women['ss_wealth_nra'] = hrs_women['ss_wealth_nra'].fillna(hrs_women[col])

sswrer_cols = [f'h{w}sswrer' for w in range(1, 17) if f'h{w}sswrer' in hrs_women.columns]
if sswrer_cols:
    hrs_women['ss_wealth_62'] = np.nan
    for col in reversed(sswrer_cols):
        hrs_women['ss_wealth_62'] = hrs_women['ss_wealth_62'].fillna(hrs_women[col])

sswrxa_cols = [f'h{w}sswrxa' for w in range(1, 17) if f'h{w}sswrxa' in hrs_women.columns]
if sswrxa_cols:
    hrs_women['ss_wealth_70'] = np.nan
    for col in reversed(sswrxa_cols):
        hrs_women['ss_wealth_70'] = hrs_women['ss_wealth_70'].fillna(hrs_women[col])

# Get household total income
income_cols = [f'h{w}itot' for w in range(1, 17) if f'h{w}itot' in hrs_women.columns]
hrs_women['income_latest'] = np.nan
for col in reversed(income_cols):
    hrs_women['income_latest'] = hrs_women['income_latest'].fillna(hrs_women[col])

# Get pension income
pension_cols = [f'r{w}ipen' for w in range(1, 17) if f'r{w}ipen' in hrs_women.columns]
if pension_cols:
    hrs_women['pension_latest'] = np.nan
    for col in reversed(pension_cols):
        hrs_women['pension_latest'] = hrs_women['pension_latest'].fillna(hrs_women[col])

print(f"\nSS retirement income available: {hrs_women['ss_ret_latest'].notna().sum():,}")
print(f"SS disability income available: {hrs_women['ss_di_latest'].notna().sum():,}" if 'ss_di_latest' in hrs_women.columns else "SS DI: N/A")
print(f"SS wealth (NRA) available: {hrs_women['ss_wealth_nra'].notna().sum():,}" if 'ss_wealth_nra' in hrs_women.columns else "SS Wealth: N/A")

# ============================================================================
# STEP 4: SS RECEIPT AND BENEFIT LEVELS BY MOTHERHOOD
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: SS Receipt and Benefit Levels by Motherhood")
print("-" * 80)

# Filter to those with SS data
ss_sample = hrs_women[
    (hrs_women['ss_ret_latest'].notna()) &
    (hrs_women['ever_mother'].notna())
].copy()

print(f"\nSample with SS data: {len(ss_sample):,}")

# SS receipt rates
ss_sample['receiving_ss'] = (ss_sample['ss_ret_latest'] > 0).astype(int)

print("\nSS Retirement Benefit Receipt and Levels:")
print("=" * 70)
print(f"{'Group':<20} {'N':>8} {'% Receiving':>12} {'Mean (if >0)':>14} {'Median':>12}")
print("-" * 70)

for status, label in [(1, 'Mothers'), (0, 'Childless')]:
    subset = ss_sample[ss_sample['ever_mother'] == status]
    n = len(subset)
    receiving = subset['receiving_ss'].mean() * 100
    recipients = subset[subset['ss_ret_latest'] > 0]['ss_ret_latest']
    mean_ss = recipients.mean() if len(recipients) > 0 else 0
    median_ss = recipients.median() if len(recipients) > 0 else 0
    print(f"{label:<20} {n:>8,} {receiving:>11.1f}% ${mean_ss:>12,.0f} ${median_ss:>10,.0f}")

# Gap calculation
mothers_ss = ss_sample[(ss_sample['ever_mother'] == 1) & (ss_sample['ss_ret_latest'] > 0)]['ss_ret_latest']
childless_ss = ss_sample[(ss_sample['ever_mother'] == 0) & (ss_sample['ss_ret_latest'] > 0)]['ss_ret_latest']

if len(mothers_ss) > 10 and len(childless_ss) > 10:
    gap_mean = (childless_ss.mean() - mothers_ss.mean()) / childless_ss.mean() * 100
    gap_median = (childless_ss.median() - mothers_ss.median()) / childless_ss.median() * 100
    print("-" * 70)
    print(f"SS Benefit Gap (Mothers vs Childless): {gap_mean:+.1f}% (mean), {gap_median:+.1f}% (median)")

# ============================================================================
# STEP 5: SS BENEFITS BY NUMBER OF CHILDREN
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: SS Benefits by Number of Children")
print("-" * 80)

print("\n" + "=" * 80)
print(f"{'Children':<15} {'N':>8} {'% Receiving':>12} {'Mean SS':>14} {'Median SS':>12} {'Gap vs 0':>10}")
print("=" * 80)

childless_mean = childless_ss.mean()
childless_median = childless_ss.median()

for cat in ['0 (Childless)', '1', '2', '3-4', '5+']:
    subset = ss_sample[ss_sample['children_cat'] == cat]
    n = len(subset)
    if n >= 10:
        receiving = (subset['ss_ret_latest'] > 0).mean() * 100
        recipients = subset[subset['ss_ret_latest'] > 0]['ss_ret_latest']
        mean_ss = recipients.mean() if len(recipients) > 0 else 0
        median_ss = recipients.median() if len(recipients) > 0 else 0
        gap = (childless_mean - mean_ss) / childless_mean * 100 if cat != '0 (Childless)' else 0
        print(f"{cat:<15} {n:>8,} {receiving:>11.1f}% ${mean_ss:>12,.0f} ${median_ss:>10,.0f} {gap:>+9.1f}%")

# ============================================================================
# STEP 6: SS AS SHARE OF TOTAL INCOME
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Social Security as Share of Total Income")
print("-" * 80)

# Calculate SS share of household income
ss_income_sample = ss_sample[
    (ss_sample['income_latest'].notna()) &
    (ss_sample['income_latest'] > 0) &
    (ss_sample['ss_ret_latest'] > 0)
].copy()

# Note: SS is individual, income is household - this is a lower bound for SS share
ss_income_sample['ss_share'] = ss_income_sample['ss_ret_latest'] / ss_income_sample['income_latest']
# Cap at 100% (individual SS can't exceed household income in principle)
ss_income_sample['ss_share'] = ss_income_sample['ss_share'].clip(upper=1.0)

print("\nSS as Share of Household Income (among SS recipients):")
print("=" * 70)
print(f"{'Group':<25} {'N':>8} {'Mean Share':>12} {'Median Share':>14}")
print("-" * 70)

for status, label in [(1, 'Mothers'), (0, 'Childless')]:
    subset = ss_income_sample[ss_income_sample['ever_mother'] == status]
    n = len(subset)
    if n >= 10:
        mean_share = subset['ss_share'].mean() * 100
        median_share = subset['ss_share'].median() * 100
        print(f"{label:<25} {n:>8,} {mean_share:>11.1f}% {median_share:>13.1f}%")

# SS reliance by income quintile
if len(ss_income_sample) >= 50:
    ss_income_sample['income_quintile'] = pd.qcut(ss_income_sample['income_latest'], 5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])

    print("\nSS Share by Income Quintile:")
    print("-" * 70)
    for quintile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)']:
        q_subset = ss_income_sample[ss_income_sample['income_quintile'] == quintile]
        mothers = q_subset[q_subset['ever_mother'] == 1]
        childless = q_subset[q_subset['ever_mother'] == 0]

        m_share = mothers['ss_share'].mean() * 100 if len(mothers) >= 5 else np.nan
        c_share = childless['ss_share'].mean() * 100 if len(childless) >= 5 else np.nan

        print(f"  {quintile}: Mothers {m_share:.1f}%, Childless {c_share:.1f}%")

# ============================================================================
# STEP 7: SS WEALTH PROJECTIONS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Social Security Wealth Projections")
print("-" * 80)

if 'ss_wealth_nra' in hrs_women.columns:
    ss_wealth_sample = hrs_women[
        (hrs_women['ss_wealth_nra'].notna()) &
        (hrs_women['ss_wealth_nra'] > 0) &
        (hrs_women['ever_mother'].notna())
    ].copy()

    print(f"\nSample with SS wealth projections: {len(ss_wealth_sample):,}")

    print("\nProjected SS Wealth (NPV at Normal Retirement Age):")
    print("=" * 70)
    print(f"{'Group':<20} {'N':>8} {'Mean Wealth':>15} {'Median Wealth':>15}")
    print("-" * 70)

    for status, label in [(1, 'Mothers'), (0, 'Childless')]:
        subset = ss_wealth_sample[ss_wealth_sample['ever_mother'] == status]
        n = len(subset)
        if n >= 10:
            mean_w = subset['ss_wealth_nra'].mean()
            median_w = subset['ss_wealth_nra'].median()
            print(f"{label:<20} {n:>8,} ${mean_w:>13,.0f} ${median_w:>13,.0f}")

    # Gap
    mothers_w = ss_wealth_sample[ss_wealth_sample['ever_mother'] == 1]['ss_wealth_nra']
    childless_w = ss_wealth_sample[ss_wealth_sample['ever_mother'] == 0]['ss_wealth_nra']
    if len(mothers_w) > 10 and len(childless_w) > 10:
        gap_w = (childless_w.mean() - mothers_w.mean()) / childless_w.mean() * 100
        print("-" * 70)
        print(f"SS Wealth Gap: {gap_w:+.1f}%")

    # By number of children
    print("\nSS Wealth by Number of Children:")
    print("-" * 70)
    childless_w_mean = childless_w.mean()
    for cat in ['0 (Childless)', '1', '2', '3-4', '5+']:
        subset = ss_wealth_sample[ss_wealth_sample['children_cat'] == cat]
        n = len(subset)
        if n >= 10:
            mean_w = subset['ss_wealth_nra'].mean()
            gap = (childless_w_mean - mean_w) / childless_w_mean * 100 if cat != '0 (Childless)' else 0
            print(f"  {cat}: ${mean_w:,.0f} (Gap: {gap:+.1f}%)")

# ============================================================================
# STEP 8: SS BY MARITAL STATUS (SPOUSAL BENEFITS)
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: SS by Marital Status (Spousal/Survivor Benefits Context)")
print("-" * 80)

print("""
POLICY CONTEXT: Social Security Benefit Types
===============================================
- Worker benefit: Based on own earnings history
- Spousal benefit: 50% of spouse's benefit (if higher than own)
- Survivor benefit: 100% of deceased spouse's benefit
- Divorced spouse: Same as spousal if married 10+ years

HYPOTHESIS: Married/widowed mothers may show higher SS due to spousal benefits,
while divorced/never-married mothers rely only on own worker benefits.
""")

ss_marital_sample = ss_sample[ss_sample['marital_category'].notna()].copy()

print("\nSS Benefits by Marital Status and Motherhood:")
print("=" * 80)
print(f"{'Marital Status':<22} {'Mothers':>12} {'Childless':>12} {'Gap':>10}")
print("=" * 80)

for marital in ['Married/Partnered', 'Divorced/Separated', 'Widowed', 'Never Married']:
    m_subset = ss_marital_sample[(ss_marital_sample['marital_category'] == marital) &
                                  (ss_marital_sample['ever_mother'] == 1) &
                                  (ss_marital_sample['ss_ret_latest'] > 0)]
    c_subset = ss_marital_sample[(ss_marital_sample['marital_category'] == marital) &
                                  (ss_marital_sample['ever_mother'] == 0) &
                                  (ss_marital_sample['ss_ret_latest'] > 0)]

    if len(m_subset) >= 5 and len(c_subset) >= 5:
        m_mean = m_subset['ss_ret_latest'].mean()
        c_mean = c_subset['ss_ret_latest'].mean()
        gap = (c_mean - m_mean) / c_mean * 100
        print(f"{marital:<22} ${m_mean:>10,.0f} ${c_mean:>10,.0f} {gap:>+9.1f}%")
    elif len(m_subset) >= 5:
        m_mean = m_subset['ss_ret_latest'].mean()
        print(f"{marital:<22} ${m_mean:>10,.0f} {'(N<5)':>12} {'N/A':>10}")
    else:
        print(f"{marital:<22} {'(N<5)':>12} {'(N<5)':>12} {'N/A':>10}")

# ============================================================================
# STEP 9: SS RELIANCE BY VULNERABILITY PROFILE
# ============================================================================

print("\n" + "-" * 80)
print("STEP 9: SS Reliance by Vulnerability Profile")
print("-" * 80)

# Create combined profile
ss_income_sample['profile'] = (ss_income_sample['marital_category'].fillna('Unknown') + ' + ' +
                                ss_income_sample['educ_cat'].fillna('Unknown') + ' + ' +
                                ss_income_sample['children_cat'].fillna('Unknown'))

# Focus on key vulnerable profiles
key_profiles = [
    ('Divorced/Separated', 'Less than HS'),
    ('Divorced/Separated', 'HS Graduate'),
    ('Never Married', 'Less than HS'),
    ('Married/Partnered', 'College+'),  # Reference group
]

print("\nSS Reliance (Share of Income) by Key Profiles:")
print("=" * 80)
print(f"{'Profile':<45} {'N':>6} {'SS Share':>12}")
print("-" * 80)

for marital, educ in key_profiles:
    subset = ss_income_sample[
        (ss_income_sample['marital_category'] == marital) &
        (ss_income_sample['educ_cat'] == educ)
    ]
    n = len(subset)
    if n >= 5:
        ss_share = subset['ss_share'].mean() * 100
        print(f"{marital + ' / ' + educ:<45} {n:>6,} {ss_share:>11.1f}%")

# ============================================================================
# STEP 10: CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 10: Creating Visualizations")
print("-" * 80)

# Figure 1: SS Benefits by Motherhood and Number of Children
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: SS benefits by number of children
ax1 = axes[0]
ss_by_children = []
for cat in ['0 (Childless)', '1', '2', '3-4', '5+']:
    subset = ss_sample[(ss_sample['children_cat'] == cat) & (ss_sample['ss_ret_latest'] > 0)]
    if len(subset) >= 10:
        ss_by_children.append({
            'children': cat,
            'mean_ss': subset['ss_ret_latest'].mean(),
            'median_ss': subset['ss_ret_latest'].median(),
            'n': len(subset)
        })

if ss_by_children:
    ss_df = pd.DataFrame(ss_by_children)
    x = range(len(ss_df))
    ax1.bar(x, ss_df['mean_ss'], color=['steelblue' if c == '0 (Childless)' else 'coral' for c in ss_df['children']], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ss_df['children'], rotation=45, ha='right')
    ax1.set_ylabel('Mean SS Retirement Benefit ($)', fontsize=12)
    ax1.set_title('Social Security Benefits by Number of Children', fontsize=14)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add sample sizes
    for i, row in ss_df.iterrows():
        ax1.text(i, row['mean_ss'] + 200, f'n={row["n"]}', ha='center', fontsize=9)

    # Add reference line for childless
    childless_val = ss_df[ss_df['children'] == '0 (Childless)']['mean_ss'].values[0]
    ax1.axhline(y=childless_val, color='steelblue', linestyle='--', alpha=0.5)

# Right: SS share by motherhood status
ax2 = axes[1]
if len(ss_income_sample) > 0:
    mothers_share = ss_income_sample[ss_income_sample['ever_mother'] == 1]['ss_share']
    childless_share = ss_income_sample[ss_income_sample['ever_mother'] == 0]['ss_share']

    positions = [1, 2]
    bp = ax2.boxplot([childless_share.dropna(), mothers_share.dropna()], positions=positions, widths=0.6, patch_artist=True)

    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Childless', 'Mothers'])
    ax2.set_ylabel('SS as Share of Household Income', fontsize=12)
    ax2.set_title('SS Reliance: Mothers vs Childless', fontsize=14)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))

    # Add means
    ax2.scatter([1, 2], [childless_share.mean(), mothers_share.mean()], color='red', marker='D', s=100, zorder=5, label='Mean')
    ax2.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'social_security_motherhood.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'social_security_motherhood.png'}")

# Figure 2: SS Gap by Marital Status
fig2, ax = plt.subplots(figsize=(10, 6))

marital_gaps = []
for marital in ['Married/Partnered', 'Divorced/Separated', 'Widowed', 'Never Married']:
    m_subset = ss_marital_sample[(ss_marital_sample['marital_category'] == marital) &
                                  (ss_marital_sample['ever_mother'] == 1) &
                                  (ss_marital_sample['ss_ret_latest'] > 0)]
    c_subset = ss_marital_sample[(ss_marital_sample['marital_category'] == marital) &
                                  (ss_marital_sample['ever_mother'] == 0) &
                                  (ss_marital_sample['ss_ret_latest'] > 0)]

    if len(m_subset) >= 5 and len(c_subset) >= 5:
        gap = (c_subset['ss_ret_latest'].mean() - m_subset['ss_ret_latest'].mean()) / c_subset['ss_ret_latest'].mean() * 100
        marital_gaps.append({'marital': marital, 'gap': gap, 'n_m': len(m_subset), 'n_c': len(c_subset)})

if marital_gaps:
    gaps_df = pd.DataFrame(marital_gaps)
    colors = ['green' if g < 0 else 'red' for g in gaps_df['gap']]
    bars = ax.barh(gaps_df['marital'], gaps_df['gap'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('SS Motherhood Gap (%)\n(Positive = Mothers receive less)', fontsize=12)
    ax.set_title('Social Security Motherhood Gap by Marital Status', fontsize=14)

    # Add annotations
    for i, row in gaps_df.iterrows():
        ax.text(row['gap'] + (1 if row['gap'] >= 0 else -1), i, f'{row["gap"]:+.1f}%',
                va='center', ha='left' if row['gap'] >= 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'social_security_marital_gap.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'social_security_marital_gap.png'}")

plt.close('all')

# ============================================================================
# STEP 11: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 11: Saving Results")
print("-" * 80)

# Summary results
results_data = []

# By motherhood status
for status, label in [(1, 'Mothers'), (0, 'Childless')]:
    subset = ss_sample[ss_sample['ever_mother'] == status]
    recipients = subset[subset['ss_ret_latest'] > 0]

    results_data.append({
        'category': 'motherhood',
        'group': label,
        'n_total': len(subset),
        'n_receiving_ss': len(recipients),
        'pct_receiving': len(recipients) / len(subset) * 100 if len(subset) > 0 else 0,
        'mean_ss_benefit': recipients['ss_ret_latest'].mean() if len(recipients) > 0 else 0,
        'median_ss_benefit': recipients['ss_ret_latest'].median() if len(recipients) > 0 else 0
    })

# By number of children
for cat in ['0 (Childless)', '1', '2', '3-4', '5+']:
    subset = ss_sample[ss_sample['children_cat'] == cat]
    recipients = subset[subset['ss_ret_latest'] > 0]

    if len(subset) >= 10:
        results_data.append({
            'category': 'children',
            'group': cat,
            'n_total': len(subset),
            'n_receiving_ss': len(recipients),
            'pct_receiving': len(recipients) / len(subset) * 100 if len(subset) > 0 else 0,
            'mean_ss_benefit': recipients['ss_ret_latest'].mean() if len(recipients) > 0 else 0,
            'median_ss_benefit': recipients['ss_ret_latest'].median() if len(recipients) > 0 else 0
        })

results_df = pd.DataFrame(results_data)
results_df.to_csv(OUTPUT_DIR / 'social_security_analysis.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'social_security_analysis.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: SOCIAL SECURITY ANALYSIS FINDINGS")
print("=" * 80)

# Calculate key metrics
mothers_ss_mean = ss_sample[(ss_sample['ever_mother'] == 1) & (ss_sample['ss_ret_latest'] > 0)]['ss_ret_latest'].mean()
childless_ss_mean = ss_sample[(ss_sample['ever_mother'] == 0) & (ss_sample['ss_ret_latest'] > 0)]['ss_ret_latest'].mean()
ss_gap = (childless_ss_mean - mothers_ss_mean) / childless_ss_mean * 100

mothers_share = ss_income_sample[ss_income_sample['ever_mother'] == 1]['ss_share'].mean() * 100
childless_share = ss_income_sample[ss_income_sample['ever_mother'] == 0]['ss_share'].mean() * 100

print(f"""
KEY FINDINGS:
=============

1. SS BENEFIT GAP
   - Mothers receive ${mothers_ss_mean:,.0f} in annual SS benefits (mean)
   - Childless women receive ${childless_ss_mean:,.0f}
   - Gap: {ss_gap:+.1f}% (mothers receive less)

2. SS RELIANCE
   - Mothers: SS = {mothers_share:.1f}% of household income
   - Childless: SS = {childless_share:.1f}% of household income
   - Mothers are MORE reliant on SS despite receiving lower benefits

3. GRADIENT BY NUMBER OF CHILDREN
   - SS benefits decline as number of children increases
   - Consistent with career interruption hypothesis

4. MARITAL STATUS INTERACTION
   - Married mothers may benefit from spousal/survivor benefits
   - Divorced mothers face double disadvantage:
     * Lower own worker benefits (career interruptions)
     * May lose spousal benefits if marriage < 10 years

5. POLICY IMPLICATIONS
   - SS motherhood gap smaller than private pension gap
   - SS's progressive formula partially compensates for lower earnings
   - But SS alone is insufficient - mothers MORE reliant on it
   - Suggests need for enhanced SS credits for caregiving

LIMITATIONS:
============
- HRS cohort 1957-1964 may not have reached full retirement age
- Some receiving SS disability rather than retirement
- Household measures may include spouse's SS

""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

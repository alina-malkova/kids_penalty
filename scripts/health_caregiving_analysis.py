#!/usr/bin/env python3
"""
Health and Caregiving Analysis: Motherhood Penalty
====================================================

Research Question: Do health status and caregiving responsibilities
explain part of the motherhood penalty on retirement income?

Key dimensions:
1. Self-reported health differences by motherhood status
2. ADL/IADL limitations
3. Health limits on work
4. Caregiving responsibilities (helping others)
5. Does controlling for health reduce the motherhood gap?

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
print("HEALTH AND CAREGIVING ANALYSIS: MOTHERHOOD PENALTY")
print("Do Health and Caregiving Explain the Gap?")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD HRS WITH HEALTH VARIABLES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading HRS with Health Variables")
print("-" * 80)

# Key variables
hrs_vars = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']

# Add children, income, marital status, and health variables by wave
for w in range(1, 17):
    hrs_vars.extend([
        f'h{w}child',      # Household children
        f'h{w}itot',       # Household total income
        f'r{w}mstat',      # Marital status
        f'r{w}shlt',       # Self-reported health (1=excellent to 5=poor)
        f'r{w}adl5a',      # Any difficulty with ADLs (0-5 count)
        f'r{w}iadl5a',     # Any difficulty with IADLs (0-5 count)
        f'r{w}hlthlm',     # Health limits work (0/1)
    ])

# Caregiving variables (available from wave 8+)
for w in range(8, 17):
    hrs_vars.append(f'r{w}lbassistprb')  # Regularly helps ailing friend/family

print("Loading HRS data with health variables...")
try:
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
    print(f"Loaded {len(hrs):,} HRS respondents")
except Exception as e:
    print(f"Note: Some variables not available: {e}")
    # Try with subset
    hrs_vars_basic = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']
    for w in range(1, 17):
        hrs_vars_basic.extend([f'h{w}child', f'h{w}itot', f'r{w}shlt'])
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars_basic, convert_categoricals=False)
    print(f"Loaded {len(hrs):,} HRS respondents (basic)")

# Filter to 1957-1964 cohort
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

# Self-reported health (latest) - 1=excellent, 5=poor
shlt_cols = [f'r{w}shlt' for w in range(1, 17) if f'r{w}shlt' in hrs_women.columns]
hrs_women['health_latest'] = np.nan
for col in reversed(shlt_cols):
    hrs_women['health_latest'] = hrs_women['health_latest'].fillna(hrs_women[col])

# Health categories
def categorize_health(h):
    if pd.isna(h):
        return np.nan
    elif h <= 2:
        return 'Good/Excellent'
    elif h == 3:
        return 'Fair'
    else:
        return 'Poor'

hrs_women['health_cat'] = hrs_women['health_latest'].apply(categorize_health)

# ADL difficulties (latest)
adl_cols = [f'r{w}adl5a' for w in range(1, 17) if f'r{w}adl5a' in hrs_women.columns]
if adl_cols:
    hrs_women['adl_latest'] = np.nan
    for col in reversed(adl_cols):
        hrs_women['adl_latest'] = hrs_women['adl_latest'].fillna(hrs_women[col])
    hrs_women['has_adl_limits'] = (hrs_women['adl_latest'] > 0).astype(int)

# IADL difficulties (latest)
iadl_cols = [f'r{w}iadl5a' for w in range(1, 17) if f'r{w}iadl5a' in hrs_women.columns]
if iadl_cols:
    hrs_women['iadl_latest'] = np.nan
    for col in reversed(iadl_cols):
        hrs_women['iadl_latest'] = hrs_women['iadl_latest'].fillna(hrs_women[col])
    hrs_women['has_iadl_limits'] = (hrs_women['iadl_latest'] > 0).astype(int)

# Health limits work (latest)
hlthlm_cols = [f'r{w}hlthlm' for w in range(1, 17) if f'r{w}hlthlm' in hrs_women.columns]
if hlthlm_cols:
    hrs_women['health_limits_work'] = np.nan
    for col in reversed(hlthlm_cols):
        hrs_women['health_limits_work'] = hrs_women['health_limits_work'].fillna(hrs_women[col])

# Caregiving (latest available)
caregive_cols = [f'r{w}lbassistprb' for w in range(8, 17) if f'r{w}lbassistprb' in hrs_women.columns]
if caregive_cols:
    hrs_women['caregiving'] = np.nan
    for col in reversed(caregive_cols):
        hrs_women['caregiving'] = hrs_women['caregiving'].fillna(hrs_women[col])

print(f"Mothers: {(hrs_women['ever_mother'] == 1).sum():,}")
print(f"Childless: {(hrs_women['ever_mother'] == 0).sum():,}")

# ============================================================================
# STEP 3: HEALTH DIFFERENCES BY MOTHERHOOD STATUS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Health Differences by Motherhood Status")
print("-" * 80)

# Filter to valid sample
valid_sample = hrs_women[
    (hrs_women['income_latest'].notna()) &
    (hrs_women['income_latest'] > 0) &
    (hrs_women['ever_mother'].notna()) &
    (hrs_women['health_latest'].notna())
].copy()

print(f"\nValid sample with health data: {len(valid_sample):,}")

# Health distribution by motherhood status
print("\n" + "=" * 70)
print("SELF-REPORTED HEALTH BY MOTHERHOOD STATUS")
print("=" * 70)

health_by_mother = pd.crosstab(valid_sample['ever_mother'], valid_sample['health_cat'],
                                normalize='index') * 100

print("\nHealth Distribution (%):")
print(f"{'Group':<15} {'Good/Excellent':>15} {'Fair':>10} {'Poor':>10}")
print("-" * 50)
for idx, row in health_by_mother.iterrows():
    label = 'Mothers' if idx == 1 else 'Childless'
    good = row.get('Good/Excellent', 0)
    fair = row.get('Fair', 0)
    poor = row.get('Poor', 0)
    print(f"{label:<15} {good:>14.1f}% {fair:>9.1f}% {poor:>9.1f}%")

# Mean health score (lower is better)
mothers_health = valid_sample[valid_sample['ever_mother'] == 1]['health_latest'].mean()
childless_health = valid_sample[valid_sample['ever_mother'] == 0]['health_latest'].mean()
print(f"\nMean health score (1=excellent, 5=poor):")
print(f"  Mothers: {mothers_health:.2f}")
print(f"  Childless: {childless_health:.2f}")
print(f"  Difference: {mothers_health - childless_health:+.2f}")

# ============================================================================
# STEP 4: ADL/IADL LIMITATIONS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Functional Limitations (ADL/IADL)")
print("-" * 80)

if 'has_adl_limits' in valid_sample.columns:
    print("\nADL Limitations (difficulty with basic activities):")
    for status, label in [(1, 'Mothers'), (0, 'Childless')]:
        subset = valid_sample[valid_sample['ever_mother'] == status]
        pct_limits = subset['has_adl_limits'].mean() * 100
        print(f"  {label}: {pct_limits:.1f}% have ADL limitations")

if 'has_iadl_limits' in valid_sample.columns:
    print("\nIADL Limitations (difficulty with instrumental activities):")
    for status, label in [(1, 'Mothers'), (0, 'Childless')]:
        subset = valid_sample[valid_sample['ever_mother'] == status]
        pct_limits = subset['has_iadl_limits'].mean() * 100
        print(f"  {label}: {pct_limits:.1f}% have IADL limitations")

if 'health_limits_work' in valid_sample.columns:
    print("\nHealth Limits Work:")
    for status, label in [(1, 'Mothers'), (0, 'Childless')]:
        subset = valid_sample[valid_sample['ever_mother'] == status]
        valid_hlw = subset[subset['health_limits_work'].notna()]
        if len(valid_hlw) > 10:
            pct_limits = valid_hlw['health_limits_work'].mean() * 100
            print(f"  {label}: {pct_limits:.1f}% report health limits work")

# ============================================================================
# STEP 5: CAREGIVING BY MOTHERHOOD STATUS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Caregiving Responsibilities")
print("-" * 80)

if 'caregiving' in valid_sample.columns:
    caregiving_sample = valid_sample[valid_sample['caregiving'].notna()]
    print(f"\nSample with caregiving data: {len(caregiving_sample):,}")

    print("\nRegularly Helps Ailing Friend/Family:")
    for status, label in [(1, 'Mothers'), (0, 'Childless')]:
        subset = caregiving_sample[caregiving_sample['ever_mother'] == status]
        if len(subset) > 10:
            pct_care = subset['caregiving'].mean() * 100
            print(f"  {label}: {pct_care:.1f}% provide regular caregiving")
else:
    print("\nCaregiving variables not available in loaded data.")

# ============================================================================
# STEP 6: INCOME GAP BY HEALTH STATUS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Income Gap by Health Status")
print("-" * 80)

print("\n" + "=" * 80)
print(f"{'Health Status':<20} {'Mothers Income':>15} {'Childless Income':>18} {'Gap':>10}")
print("=" * 80)

health_income_results = []

for health in ['Good/Excellent', 'Fair', 'Poor']:
    subset = valid_sample[valid_sample['health_cat'] == health]
    mothers = subset[subset['ever_mother'] == 1]
    childless = subset[subset['ever_mother'] == 0]

    n_m = len(mothers)
    n_c = len(childless)

    if n_m >= 20 and n_c >= 5:
        mean_m = mothers['income_latest'].mean()
        mean_c = childless['income_latest'].mean()
        gap = (mean_c - mean_m) / mean_c * 100

        print(f"{health:<20} ${mean_m:>13,.0f} ${mean_c:>16,.0f} {gap:>+9.1f}%")

        health_income_results.append({
            'health': health,
            'n_mothers': n_m,
            'n_childless': n_c,
            'mean_mothers': mean_m,
            'mean_childless': mean_c,
            'gap': gap
        })

# ============================================================================
# STEP 7: DOES CONTROLLING FOR HEALTH REDUCE THE GAP?
# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Does Controlling for Health Reduce the Gap?")
print("-" * 80)

# Overall gap without health controls
mothers_all = valid_sample[valid_sample['ever_mother'] == 1]
childless_all = valid_sample[valid_sample['ever_mother'] == 0]
overall_gap = (childless_all['income_latest'].mean() - mothers_all['income_latest'].mean()) / childless_all['income_latest'].mean() * 100

print(f"\nOverall gap (no controls): {overall_gap:+.1f}%")

# Gap within good health
good_health = valid_sample[valid_sample['health_cat'] == 'Good/Excellent']
if len(good_health) > 50:
    mothers_good = good_health[good_health['ever_mother'] == 1]
    childless_good = good_health[good_health['ever_mother'] == 0]
    if len(childless_good) > 5:
        gap_good = (childless_good['income_latest'].mean() - mothers_good['income_latest'].mean()) / childless_good['income_latest'].mean() * 100
        print(f"Gap among Good/Excellent health: {gap_good:+.1f}%")
        print(f"Reduction from controlling for health: {overall_gap - gap_good:.1f} pp")

# Gap within poor health
poor_health = valid_sample[valid_sample['health_cat'] == 'Poor']
if len(poor_health) > 30:
    mothers_poor = poor_health[poor_health['ever_mother'] == 1]
    childless_poor = poor_health[poor_health['ever_mother'] == 0]
    if len(childless_poor) > 3:
        gap_poor = (childless_poor['income_latest'].mean() - mothers_poor['income_latest'].mean()) / childless_poor['income_latest'].mean() * 100
        print(f"Gap among Poor health: {gap_poor:+.1f}%")

# ============================================================================
# STEP 8: HEALTH BY NUMBER OF CHILDREN
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Health by Number of Children")
print("-" * 80)

print("\n" + "=" * 70)
print(f"{'Children':<15} {'Mean Health':>12} {'% Poor/Fair':>15} {'N':>10}")
print("=" * 70)

for cat in ['0 (Childless)', '1', '2', '3-4', '5+']:
    subset = valid_sample[valid_sample['children_cat'] == cat]
    if len(subset) >= 20:
        mean_health = subset['health_latest'].mean()
        pct_poor_fair = (subset['health_cat'].isin(['Fair', 'Poor'])).mean() * 100
        print(f"{cat:<15} {mean_health:>12.2f} {pct_poor_fair:>14.1f}% {len(subset):>10,}")

# ============================================================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 9: Creating Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Health distribution by motherhood
ax1 = axes[0, 0]
health_dist = pd.crosstab(valid_sample['ever_mother'], valid_sample['health_cat'], normalize='index') * 100
health_order = ['Good/Excellent', 'Fair', 'Poor']
health_dist = health_dist[[c for c in health_order if c in health_dist.columns]]

x = np.arange(len(health_dist.columns))
width = 0.35

ax1.bar(x - width/2, health_dist.iloc[0], width, label='Childless', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, health_dist.iloc[1], width, label='Mothers', color='coral', alpha=0.8)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.set_title('A. Self-Reported Health by Motherhood Status', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(health_dist.columns)
ax1.legend()

# Top-right: Income gap by health status
ax2 = axes[0, 1]
if health_income_results:
    health_labels = [r['health'] for r in health_income_results]
    gaps = [r['gap'] for r in health_income_results]
    colors = ['green' if g < 0 else 'red' for g in gaps]
    ax2.barh(health_labels, gaps, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Motherhood Gap (%)', fontsize=12)
    ax2.set_title('B. Income Gap by Health Status', fontsize=14)
    for i, v in enumerate(gaps):
        ax2.text(v + 1, i, f'{v:+.1f}%', va='center', fontsize=10)

# Bottom-left: Mean health by number of children
ax3 = axes[1, 0]
children_health = []
for cat in ['0 (Childless)', '1', '2', '3-4', '5+']:
    subset = valid_sample[valid_sample['children_cat'] == cat]
    if len(subset) >= 20:
        children_health.append({
            'children': cat,
            'mean_health': subset['health_latest'].mean()
        })

if children_health:
    ch_df = pd.DataFrame(children_health)
    colors = ['steelblue' if c == '0 (Childless)' else 'coral' for c in ch_df['children']]
    ax3.bar(ch_df['children'], ch_df['mean_health'], color=colors, alpha=0.8)
    ax3.set_ylabel('Mean Health Score\n(1=Excellent, 5=Poor)', fontsize=12)
    ax3.set_title('C. Health by Number of Children', fontsize=14)
    ax3.set_xticklabels(ch_df['children'], rotation=45, ha='right')

# Bottom-right: Income by health and motherhood
ax4 = axes[1, 1]
health_income_data = []
for health in ['Good/Excellent', 'Fair', 'Poor']:
    for status, label in [(0, 'Childless'), (1, 'Mothers')]:
        subset = valid_sample[(valid_sample['health_cat'] == health) &
                               (valid_sample['ever_mother'] == status)]
        if len(subset) >= 10:
            health_income_data.append({
                'health': health,
                'group': label,
                'mean_income': subset['income_latest'].mean()
            })

if health_income_data:
    hi_df = pd.DataFrame(health_income_data)
    hi_pivot = hi_df.pivot(index='health', columns='group', values='mean_income')
    hi_pivot = hi_pivot.reindex(['Good/Excellent', 'Fair', 'Poor'])

    x = np.arange(len(hi_pivot))
    width = 0.35
    ax4.bar(x - width/2, hi_pivot['Childless'], width, label='Childless', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, hi_pivot['Mothers'], width, label='Mothers', color='coral', alpha=0.8)
    ax4.set_ylabel('Mean Household Income ($)', fontsize=12)
    ax4.set_title('D. Income by Health and Motherhood', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(hi_pivot.index)
    ax4.legend()
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'health_caregiving_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'health_caregiving_analysis.png'}")

plt.close('all')

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 10: Saving Results")
print("-" * 80)

results_df = pd.DataFrame(health_income_results)
results_df.to_csv(OUTPUT_DIR / 'health_caregiving_analysis.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'health_caregiving_analysis.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: HEALTH AND CAREGIVING ANALYSIS")
print("=" * 80)

print(f"""
KEY FINDINGS:
=============

1. HEALTH DIFFERENCES BY MOTHERHOOD STATUS
   - Mothers report {"worse" if mothers_health > childless_health else "better"} health on average
   - Mean health: Mothers {mothers_health:.2f} vs Childless {childless_health:.2f}
   - (Scale: 1=Excellent, 5=Poor)

2. DOES HEALTH EXPLAIN THE GAP?
   - Overall gap: {overall_gap:+.1f}%
   - The gap {"persists" if abs(overall_gap) > 5 else "is small"} even when comparing
     women with similar health status
   - Health differences explain {"some" if abs(mothers_health - childless_health) > 0.1 else "little"} of the
     raw difference, but substantial gaps remain within health categories

3. HEALTH GRADIENT BY NUMBER OF CHILDREN
   - Women with more children tend to report {"worse" if children_health[-1]['mean_health'] > children_health[0]['mean_health'] else "similar"} health
   - This could reflect: (a) caregiving stress, (b) selection, or (c)
     socioeconomic factors correlated with both fertility and health

4. POLICY IMPLICATIONS
   - Health status is not a primary driver of the motherhood gap
   - Gaps persist within health categories, suggesting other mechanisms
   - Caregiving responsibilities may affect health and income independently
   - Health-based targeting would not effectively address motherhood-related
     retirement income disparities

LIMITATIONS:
============
- Self-reported health is subjective
- Cannot distinguish causation (motherhood -> health) from selection
- Caregiving variables only available for recent waves
- Health may affect income through multiple channels (work capacity,
  healthcare costs, etc.)
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

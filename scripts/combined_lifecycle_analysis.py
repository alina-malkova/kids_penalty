#!/usr/bin/env python3
"""
Combined NLSY79 + HRS Lifecycle Analysis
=========================================

Combines NLSY79 (early/mid career) with HRS (late career/retirement) to study
the evolution of the motherhood penalty across the lifecycle.

Birth cohort: 1957-1964 (same in both datasets)

NLSY79 Coverage:
- 1979-2018 surveys
- Ages 14-22 (1979) to 54-61 (2018)
- Strong fertility data (NUMKID - children ever born)

HRS Coverage:
- 1992-2022 surveys
- Ages 50+ at entry
- Strong retirement income data

This is a SYNTHETIC COHORT comparison (not individual-level linkage).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

print("=" * 80)
print("COMBINED NLSY79 + HRS LIFECYCLE ANALYSIS")
print("Motherhood Penalty Across the Lifecycle")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD NLSY79 DATA (Multiple Extracts)
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading NLSY79 Data")
print("-" * 80)

# Load NLSY with fertility (nlsy2) and retirement (nlsy21) data
nlsy2 = pd.read_csv("/Users/amalkova/Downloads/NLSY2_Data/nlsy2.csv")
nlsy21 = pd.read_csv("/Users/amalkova/Downloads/NLSY21_Data/nlsy21.csv")

print(f"NLSY2 (fertility/income): {len(nlsy2):,} respondents, {len(nlsy2.columns)} variables")
print(f"NLSY21 (retirement): {len(nlsy21):,} respondents, {len(nlsy21.columns)} variables")

# Merge NLSY datasets on CASEID
nlsy = nlsy2.merge(nlsy21[['R0000100'] + [c for c in nlsy21.columns if c.startswith('T8') or c == 'R9908000']],
                   on='R0000100', how='left', suffixes=('', '_ret'))

print(f"Merged NLSY: {len(nlsy):,} respondents")

# ============================================================================
# STEP 2: CREATE NLSY PANEL DATA
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Creating NLSY Panel Data")
print("-" * 80)

# NLSY variable mappings
# Income (TNFI_TRUNC) by year
nlsy_income_vars = {
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

# Create long-format panel for NLSY
nlsy_panels = []

for year, income_var in nlsy_income_vars.items():
    if income_var in nlsy.columns:
        panel = pd.DataFrame()
        panel['id'] = nlsy['R0000100']
        panel['source'] = 'NLSY79'
        panel['year'] = year

        # Demographics (time-invariant)
        panel['female'] = (nlsy['R0214800'] == 2).astype(int)
        panel['race'] = nlsy['R0214700']  # 1=Hispanic, 2=Black, 3=White

        # Harmonize race (1=White, 2=Black, 3=Other)
        panel['race_harmonized'] = np.where(panel['race'] == 3, 1,
                                   np.where(panel['race'] == 2, 2, 3))

        # Motherhood (use cross-round NUMKID if available)
        if 'R9908000' in nlsy.columns:
            panel['num_children'] = pd.to_numeric(nlsy['R9908000'], errors='coerce')
            panel['num_children'] = panel['num_children'].replace({-1: np.nan, -2: np.nan, -3: np.nan, -4: np.nan, -5: np.nan})

        panel['mother'] = (panel['num_children'] > 0).astype(int)
        panel.loc[panel['num_children'].isna(), 'mother'] = np.nan

        # Income
        panel['income'] = pd.to_numeric(nlsy[income_var], errors='coerce')
        panel.loc[panel['income'] < 0, 'income'] = np.nan

        # Calculate age (birth year is 1957-1964, use midpoint 1960.5)
        # Actually use birth year from R0000500 if available
        panel['age'] = year - 1960  # Approximate

        nlsy_panels.append(panel)

nlsy_panel = pd.concat(nlsy_panels, ignore_index=True)
print(f"NLSY panel: {len(nlsy_panel):,} person-years")

# Add 2018 retirement data
if 'T8117700' in nlsy.columns:  # Pension amount
    panel_2018 = pd.DataFrame()
    panel_2018['id'] = nlsy['R0000100']
    panel_2018['source'] = 'NLSY79'
    panel_2018['year'] = 2018
    panel_2018['female'] = (nlsy['R0214800'] == 2).astype(int)
    panel_2018['race'] = nlsy['R0214700']
    panel_2018['race_harmonized'] = np.where(panel_2018['race'] == 3, 1,
                                    np.where(panel_2018['race'] == 2, 2, 3))
    if 'R9908000' in nlsy.columns:
        panel_2018['num_children'] = pd.to_numeric(nlsy['R9908000'], errors='coerce')
        panel_2018['num_children'] = panel_2018['num_children'].replace({-1: np.nan, -2: np.nan, -3: np.nan, -4: np.nan, -5: np.nan})
    panel_2018['mother'] = (panel_2018['num_children'] > 0).astype(int)
    panel_2018.loc[panel_2018['num_children'].isna(), 'mother'] = np.nan

    # Use pension income for 2018
    panel_2018['pension_income'] = pd.to_numeric(nlsy['T8117700'], errors='coerce')
    panel_2018.loc[panel_2018['pension_income'] < 0, 'pension_income'] = np.nan

    # IRA amount
    if 'T8119300' in nlsy.columns:
        panel_2018['ira_amount'] = pd.to_numeric(nlsy['T8119300'], errors='coerce')
        panel_2018.loc[panel_2018['ira_amount'] < 0, 'ira_amount'] = np.nan

    panel_2018['age'] = 2018 - 1960
    nlsy_panel = pd.concat([nlsy_panel, panel_2018], ignore_index=True)

# ============================================================================
# STEP 3: LOAD HRS DATA
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Loading HRS Data")
print("-" * 80)

HRS_PATH = Path("/Users/amalkova/Downloads/RAND_HRS_2022/randhrs1992_2022v1.dta")

# Load HRS with key variables
hrs_vars = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc']
for w in range(1, 17):
    hrs_vars.extend([f'h{w}child', f'h{w}itot'])

try:
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars, convert_categoricals=False)
except:
    # Fallback
    hrs_vars_basic = ['hhidpn', 'rabyear', 'ragender', 'raracem', 'raeduc',
                      'h1child', 'h8child', 'h16child', 'h1itot', 'h8itot', 'h16itot']
    hrs = pd.read_stata(HRS_PATH, columns=hrs_vars_basic, convert_categoricals=False)

# Filter to 1957-1964 cohort
hrs = hrs[(hrs['rabyear'] >= 1957) & (hrs['rabyear'] <= 1964)].copy()
print(f"HRS (1957-1964 cohort): {len(hrs):,} respondents")

# Create HRS panel
hrs_panels = []
wave_years = {1: 1992, 2: 1994, 3: 1996, 4: 1998, 5: 2000, 6: 2002, 7: 2004,
              8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020, 16: 2022}

for wave, year in wave_years.items():
    child_var = f'h{wave}child'
    income_var = f'h{wave}itot'

    if child_var in hrs.columns and income_var in hrs.columns:
        panel = pd.DataFrame()
        panel['id'] = hrs['hhidpn']
        panel['source'] = 'HRS'
        panel['year'] = year
        panel['female'] = (hrs['ragender'] == 2).astype(int)
        panel['race_harmonized'] = hrs['raracem']  # Already 1=White, 2=Black, 3=Other
        panel['birth_year'] = hrs['rabyear']
        panel['age'] = year - hrs['rabyear']

        # Children (max across waves for "ever mother")
        child_cols = [f'h{w}child' for w in range(1, wave+1) if f'h{w}child' in hrs.columns]
        if child_cols:
            panel['max_children'] = hrs[child_cols].max(axis=1)
            panel['mother'] = (panel['max_children'] > 0).astype(int)

        # Income
        panel['income'] = pd.to_numeric(hrs[income_var], errors='coerce')

        hrs_panels.append(panel)

hrs_panel = pd.concat(hrs_panels, ignore_index=True)
print(f"HRS panel: {len(hrs_panel):,} person-years")

# ============================================================================
# STEP 4: COMBINE DATASETS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Combining Datasets")
print("-" * 80)

# Harmonize column names
nlsy_panel_harmonized = nlsy_panel[['id', 'source', 'year', 'female', 'race_harmonized',
                                     'mother', 'income', 'age']].copy()
nlsy_panel_harmonized['age'] = nlsy_panel_harmonized['year'] - 1960  # Approximate

hrs_panel_harmonized = hrs_panel[['id', 'source', 'year', 'female', 'race_harmonized',
                                   'mother', 'income', 'age']].copy()

# Combine
combined = pd.concat([nlsy_panel_harmonized, hrs_panel_harmonized], ignore_index=True)
print(f"Combined panel: {len(combined):,} person-years")

# Filter to women only
combined_women = combined[combined['female'] == 1].copy()
print(f"Women only: {len(combined_women):,} person-years")

# ============================================================================
# STEP 5: LIFECYCLE ANALYSIS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Lifecycle Analysis - Motherhood Penalty by Age")
print("-" * 80)

# Create age groups
combined_women['age_group'] = pd.cut(combined_women['age'],
                                      bins=[15, 25, 35, 45, 55, 65, 75],
                                      labels=['15-25', '25-35', '35-45', '45-55', '55-65', '65+'])

# Calculate motherhood penalty by age group
results = []

for age_group in ['15-25', '25-35', '35-45', '45-55', '55-65', '65+']:
    subset = combined_women[
        (combined_women['age_group'] == age_group) &
        (combined_women['income'].notna()) &
        (combined_women['income'] > 0) &
        (combined_women['mother'].notna())
    ]

    mothers = subset[subset['mother'] == 1]
    childless = subset[subset['mother'] == 0]

    if len(mothers) >= 50 and len(childless) >= 30:
        m_mean = mothers['income'].mean()
        c_mean = childless['income'].mean()
        penalty = (c_mean - m_mean) / c_mean * 100

        results.append({
            'age_group': age_group,
            'n_mothers': len(mothers),
            'n_childless': len(childless),
            'mean_mothers': m_mean,
            'mean_childless': c_mean,
            'penalty_pct': penalty
        })

        print(f"\n{age_group}:")
        print(f"  Mothers: n={len(mothers):,}, mean=${m_mean:,.0f}")
        print(f"  Childless: n={len(childless):,}, mean=${c_mean:,.0f}")
        print(f"  Penalty: {penalty:+.1f}%")

results_df = pd.DataFrame(results)

# ============================================================================
# STEP 6: ANALYSIS BY DATA SOURCE
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Comparison by Data Source")
print("-" * 80)

for source in ['NLSY79', 'HRS']:
    subset = combined_women[
        (combined_women['source'] == source) &
        (combined_women['income'].notna()) &
        (combined_women['income'] > 0) &
        (combined_women['mother'].notna())
    ]

    mothers = subset[subset['mother'] == 1]
    childless = subset[subset['mother'] == 0]

    if len(mothers) >= 50 and len(childless) >= 30:
        m_mean = mothers['income'].mean()
        c_mean = childless['income'].mean()
        penalty = (c_mean - m_mean) / c_mean * 100

        print(f"\n{source}:")
        print(f"  Age range: {subset['age'].min():.0f} - {subset['age'].max():.0f}")
        print(f"  Years: {subset['year'].min()} - {subset['year'].max()}")
        print(f"  Mothers: n={len(mothers):,}, mean=${m_mean:,.0f}")
        print(f"  Childless: n={len(childless):,}, mean=${c_mean:,.0f}")
        print(f"  Motherhood Penalty: {penalty:+.1f}%")

# ============================================================================
# STEP 7: CREATE VISUALIZATION
# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Creating Lifecycle Visualization")
print("-" * 80)

if len(results_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Motherhood penalty by age
    ax1 = axes[0]
    colors = ['#e74c3c' if p > 0 else '#27ae60' for p in results_df['penalty_pct']]
    bars = ax1.bar(results_df['age_group'], results_df['penalty_pct'], color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Age Group', fontsize=12)
    ax1.set_ylabel('Motherhood Penalty (%)', fontsize=12)
    ax1.set_title('Motherhood Penalty Across the Lifecycle\n(Combined NLSY79 + HRS)', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, results_df['penalty_pct']):
        height = bar.get_height()
        ax1.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')

    # Plot 2: Sample sizes
    ax2 = axes[1]
    x = np.arange(len(results_df))
    width = 0.35
    ax2.bar(x - width/2, results_df['n_mothers'], width, label='Mothers', color='#e74c3c', alpha=0.7)
    ax2.bar(x + width/2, results_df['n_childless'], width, label='Childless', color='#3498db', alpha=0.7)
    ax2.set_xlabel('Age Group', fontsize=12)
    ax2.set_ylabel('Sample Size (person-years)', fontsize=12)
    ax2.set_title('Sample Sizes by Age Group', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['age_group'])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(BASE_DIR / 'lifecycle_penalty_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: lifecycle_penalty_combined.png")

# ============================================================================
# STEP 8: SUMMARY TABLE
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Summary - Lifecycle Evolution of Motherhood Penalty")
print("-" * 80)

print("""
LIFECYCLE EVOLUTION OF MOTHERHOOD PENALTY
==========================================
(Combined NLSY79 + HRS, Birth Cohort 1957-1964)

Life Stage          Data Source    Ages      Penalty
---------------------------------------------------------""")

if len(results_df) > 0:
    for _, row in results_df.iterrows():
        source = "NLSY79" if int(row['age_group'].split('-')[0]) < 50 else "HRS"
        print(f"{row['age_group']:18} {source:14} {row['age_group']:9} {row['penalty_pct']:+.1f}%")

print("""
KEY FINDINGS:
-------------
1. Motherhood penalty is LARGEST in prime working years (25-45)
2. Penalty may persist or reverse in late career (55+)
3. HRS results at 55+ show mothers with HIGHER income
   - Likely due to household income measure (includes spouse)
   - Small childless sample in HRS

METHODOLOGICAL NOTE:
This is a synthetic cohort comparison - same birth cohort (1957-1964)
observed at different ages across two complementary datasets.
Not individual-level panel tracking.
""")

# ============================================================================
# STEP 9: SAVE OUTPUTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 9: Saving Outputs")
print("-" * 80)

# Save combined panel
combined_women.to_csv(OUTPUT_DIR / "combined_lifecycle_panel.csv", index=False)
print(f"Saved: combined_lifecycle_panel.csv ({len(combined_women):,} rows)")

# Save results summary
results_df.to_csv(OUTPUT_DIR / "lifecycle_penalty_by_age.csv", index=False)
print(f"Saved: lifecycle_penalty_by_age.csv")

print("\n" + "=" * 80)
print("COMBINED LIFECYCLE ANALYSIS COMPLETE")
print("=" * 80)

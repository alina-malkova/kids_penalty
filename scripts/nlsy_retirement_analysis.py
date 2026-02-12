#!/usr/bin/env python3
"""
NLSY79 Retirement Income Analysis
==================================

Analyzes motherhood penalty using NLSY79 retirement variables:
- Pension income (RETINCR-PENSIONS)
- Annuity income (RETINCR-ANNUITIES)
- IRA/Keogh (RETINCR-IRA)
- Social Security (RETINCR-SOCSEC)
- Spouse retirement income (RETINCSP-*)

Survey Year 2018: NLSY79 respondents aged 54-61 (born 1957-1964)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

print("=" * 80)
print("NLSY79 RETIREMENT INCOME ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD NLSY21 (RETIREMENT VARIABLES)
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading NLSY21 (Retirement Variables)")
print("-" * 80)

nlsy21 = pd.read_csv("/Users/amalkova/Downloads/NLSY21_Data/nlsy21.csv")
print(f"Loaded {len(nlsy21):,} respondents with {len(nlsy21.columns)} variables")

# Key variable mappings (from codebook)
# Demographics
# R0000100 = CASEID
# R0214700 = SAMPLE_RACE (1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic)
# R0214800 = SAMPLE_SEX (1=Male, 2=Female)

# Fertility
# R0013300 = Q9-72 (Has R ever had children?)
# R0013400 = FER-2A (Number of children)

# Retirement Income (2018 survey - ages 54-61)
# T8117500 = RETINCR-PENSIONS-1 (Receiving pension income? 0/1)
# T8117700 = RETINCR-PENSIONS-2_TRUNC (Amount of pension income)
# T8118300 = RETINCR-ANNUITIES-1 (Receiving annuity income?)
# T8118500 = RETINCR-ANNUITIES-2_TRUNC (Amount of annuity income)
# T8119100 = RETINCR-IRA-1 (Has IRA/personal retirement?)
# T8119300 = RETINCR-IRA-2_TRUNC (Amount in IRA)
# T8120800 = RETINCR-SOCSEC-1 (Receiving Social Security?)

# Check which columns exist
print("\nChecking retirement variable availability...")
ret_vars = {
    'pension_receiving': 'T8117500',
    'pension_amount': 'T8117700',
    'annuity_receiving': 'T8118300',
    'annuity_amount': 'T8118500',
    'ira_has': 'T8119100',
    'ira_amount': 'T8119300',
    'socsec_receiving': 'T8120800',
}

available_ret_vars = {}
for name, code in ret_vars.items():
    if code in nlsy21.columns:
        available_ret_vars[name] = code
        print(f"  {name}: {code} - Found")
    else:
        print(f"  {name}: {code} - NOT FOUND")

# ============================================================================
# CREATE HARMONIZED DATASET
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Creating Harmonized Dataset")
print("-" * 80)

nlsy_ret = pd.DataFrame()
nlsy_ret['id'] = nlsy21['R0000100']

# Demographics
nlsy_ret['race'] = nlsy21['R0214700']  # 1=Hispanic, 2=Black, 3=White
nlsy_ret['female'] = (nlsy21['R0214800'] == 2).astype(int)

# Recode race to match HRS (1=White, 2=Black, 3=Hispanic/Other)
nlsy_ret['race_harmonized'] = np.where(nlsy_ret['race'] == 3, 1,  # Non-Black/Non-Hispanic -> White
                               np.where(nlsy_ret['race'] == 2, 2,   # Black -> Black
                               3))  # Hispanic -> Other

# Fertility - use NUMKID (children EVER born, not children in household)
# R9908000 = NUMKID (cross-round summary variable - total children ever born)
# This is different from NUMCH which is children currently in household

if 'R9908000' in nlsy21.columns:
    nlsy_ret['num_children_ever'] = pd.to_numeric(nlsy21['R9908000'], errors='coerce')
    nlsy_ret['num_children_ever'] = nlsy_ret['num_children_ever'].replace({-1: np.nan, -2: np.nan, -3: np.nan, -4: np.nan, -5: np.nan})
    print(f"  Using NUMKID cross-round (R9908000): {nlsy_ret['num_children_ever'].notna().sum()} valid values")

    # Define mother based on total children ever born
    nlsy_ret['mother'] = (nlsy_ret['num_children_ever'] > 0).astype(int)
    nlsy_ret.loc[nlsy_ret['num_children_ever'].isna(), 'mother'] = np.nan
else:
    # Fallback to survey-year NUMKID variables
    numkid_vars = [
        ('T8226700', 'NUMKID18'),
        ('T5779600', 'NUMKID16'),
        ('T2217700', 'NUMKID08'),
        ('R8504200', 'NUMKID04'),
    ]
    nlsy_ret['num_children_ever'] = np.nan
    for code, name in numkid_vars:
        if code in nlsy21.columns:
            temp = pd.to_numeric(nlsy21[code], errors='coerce')
            temp = temp.replace({-1: np.nan, -2: np.nan, -3: np.nan, -4: np.nan, -5: np.nan})
            nlsy_ret['num_children_ever'] = nlsy_ret['num_children_ever'].fillna(temp)
            print(f"  Using {name} ({code}): {temp.notna().sum()} valid values")

    nlsy_ret['mother'] = (nlsy_ret['num_children_ever'] > 0).astype(int)
    nlsy_ret.loc[nlsy_ret['num_children_ever'].isna(), 'mother'] = np.nan

# Retirement income variables (2018)
for name, code in available_ret_vars.items():
    if code in nlsy21.columns:
        nlsy_ret[name] = pd.to_numeric(nlsy21[code], errors='coerce')
        nlsy_ret.loc[nlsy_ret[name] < 0, name] = np.nan

# Also get total family income for comparison
if 'T8116200' in nlsy21.columns:  # TNFI_TRUNC 2018
    nlsy_ret['total_income_2018'] = pd.to_numeric(nlsy21['T8116200'], errors='coerce')
    nlsy_ret.loc[nlsy_ret['total_income_2018'] < 0, 'total_income_2018'] = np.nan

# Look for any TNFI variable
tnfi_cols = [c for c in nlsy21.columns if 'T8' in c]
print(f"\nFound {len(tnfi_cols)} T8* columns (2018 survey year)")

print(f"\nDataset created: {len(nlsy_ret):,} observations")
print(f"Variables: {list(nlsy_ret.columns)}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Summary Statistics")
print("-" * 80)

# Filter to women
women = nlsy_ret[nlsy_ret['female'] == 1].copy()
print(f"\nWomen in sample: {len(women):,}")

# Motherhood status (using total children ever born)
if 'mother' in women.columns:
    n_mothers = women['mother'].sum()
    n_childless = (women['mother'] == 0).sum()
    n_missing = women['mother'].isna().sum()
    print(f"\nMotherhood status (total children ever born):")
    print(f"  Mothers: {n_mothers:,.0f} ({n_mothers/len(women)*100:.1f}%)")
    print(f"  Childless: {n_childless:,} ({n_childless/len(women)*100:.1f}%)")
    print(f"  Missing: {n_missing:,}")

    if 'num_children_ever' in women.columns:
        valid = women['num_children_ever'].dropna()
        print(f"\nChildren distribution (among mothers):")
        print(f"  Mean: {valid[valid > 0].mean():.2f}")
        print(f"  Median: {valid[valid > 0].median():.0f}")

# Retirement income (2018)
print(f"\nRetirement Income Sources (2018, ages 54-61):")
for name in available_ret_vars.keys():
    if name in women.columns:
        if 'receiving' in name or 'has' in name:
            n_yes = (women[name] == 1).sum()
            n_no = (women[name] == 0).sum()
            pct = n_yes / (n_yes + n_no) * 100 if (n_yes + n_no) > 0 else 0
            print(f"  {name}: {n_yes:,} receiving ({pct:.1f}%)")
        elif 'amount' in name:
            valid = women[name].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                print(f"  {name}: mean=${valid.mean():,.0f}, median=${valid.median():,.0f}, n={len(valid)}")

# ============================================================================
# MOTHERHOOD PENALTY ON RETIREMENT INCOME
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Motherhood Penalty on Retirement Income")
print("-" * 80)

# Use any available retirement income measure
women_valid = women.dropna(subset=['mother'])

# Check pension amount
if 'pension_amount' in women_valid.columns:
    pension_valid = women_valid[women_valid['pension_amount'].notna() & (women_valid['pension_amount'] > 0)]
    if len(pension_valid) >= 100:
        mothers_pension = pension_valid[pension_valid['mother'] == 1]['pension_amount']
        childless_pension = pension_valid[pension_valid['mother'] == 0]['pension_amount']

        if len(mothers_pension) >= 30 and len(childless_pension) >= 30:
            print(f"\nPENSION INCOME:")
            print(f"  Mothers (n={len(mothers_pension)}): mean=${mothers_pension.mean():,.0f}, median=${mothers_pension.median():,.0f}")
            print(f"  Childless (n={len(childless_pension)}): mean=${childless_pension.mean():,.0f}, median=${childless_pension.median():,.0f}")
            penalty = (childless_pension.mean() - mothers_pension.mean()) / childless_pension.mean() * 100
            print(f"  Motherhood Penalty: {penalty:+.1f}%")

# Check IRA amount
if 'ira_amount' in women_valid.columns:
    ira_valid = women_valid[women_valid['ira_amount'].notna() & (women_valid['ira_amount'] > 0)]
    if len(ira_valid) >= 100:
        mothers_ira = ira_valid[ira_valid['mother'] == 1]['ira_amount']
        childless_ira = ira_valid[ira_valid['mother'] == 0]['ira_amount']

        if len(mothers_ira) >= 30 and len(childless_ira) >= 30:
            print(f"\nIRA/PERSONAL RETIREMENT:")
            print(f"  Mothers (n={len(mothers_ira)}): mean=${mothers_ira.mean():,.0f}, median=${mothers_ira.median():,.0f}")
            print(f"  Childless (n={len(childless_ira)}): mean=${childless_ira.mean():,.0f}, median=${childless_ira.median():,.0f}")
            penalty = (childless_ira.mean() - mothers_ira.mean()) / childless_ira.mean() * 100
            print(f"  Motherhood Penalty: {penalty:+.1f}%")

# Check total income 2018
if 'total_income_2018' in women_valid.columns:
    income_valid = women_valid[women_valid['total_income_2018'].notna() & (women_valid['total_income_2018'] > 0)]
    if len(income_valid) >= 100:
        mothers_inc = income_valid[income_valid['mother'] == 1]['total_income_2018']
        childless_inc = income_valid[income_valid['mother'] == 0]['total_income_2018']

        if len(mothers_inc) >= 30 and len(childless_inc) >= 30:
            print(f"\nTOTAL FAMILY INCOME (2018):")
            print(f"  Mothers (n={len(mothers_inc)}): mean=${mothers_inc.mean():,.0f}, median=${mothers_inc.median():,.0f}")
            print(f"  Childless (n={len(childless_inc)}): mean=${childless_inc.mean():,.0f}, median=${childless_inc.median():,.0f}")
            penalty = (childless_inc.mean() - mothers_inc.mean()) / childless_inc.mean() * 100
            print(f"  Motherhood Penalty: {penalty:+.1f}%")

# ============================================================================
# RETIREMENT READINESS BY MOTHERHOOD
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Retirement Readiness by Motherhood Status")
print("-" * 80)

# Has any retirement savings?
if 'ira_has' in women_valid.columns:
    print("\nHas IRA/Personal Retirement Account:")
    mothers = women_valid[women_valid['mother'] == 1]
    childless = women_valid[women_valid['mother'] == 0]

    m_has_ira = (mothers['ira_has'] == 1).sum() / len(mothers) * 100
    c_has_ira = (childless['ira_has'] == 1).sum() / len(childless) * 100
    print(f"  Mothers: {m_has_ira:.1f}%")
    print(f"  Childless: {c_has_ira:.1f}%")
    print(f"  Gap: {c_has_ira - m_has_ira:+.1f} percentage points")

# Receiving pension
if 'pension_receiving' in women_valid.columns:
    print("\nCurrently Receiving Pension:")
    m_pension = (mothers['pension_receiving'] == 1).sum() / len(mothers) * 100
    c_pension = (childless['pension_receiving'] == 1).sum() / len(childless) * 100
    print(f"  Mothers: {m_pension:.1f}%")
    print(f"  Childless: {c_pension:.1f}%")
    print(f"  Gap: {c_pension - m_pension:+.1f} percentage points")

# Social Security
if 'socsec_receiving' in women_valid.columns:
    print("\nReceiving Social Security:")
    m_ss = (mothers['socsec_receiving'] == 1).sum() / len(mothers) * 100
    c_ss = (childless['socsec_receiving'] == 1).sum() / len(childless) * 100
    print(f"  Mothers: {m_ss:.1f}%")
    print(f"  Childless: {c_ss:.1f}%")
    print(f"  Gap: {c_ss - m_ss:+.1f} percentage points")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Saving Results")
print("-" * 80)

nlsy_ret.to_csv(OUTPUT_DIR / "nlsy_retirement_harmonized.csv", index=False)
print(f"Saved: nlsy_retirement_harmonized.csv")

# Women only
women.to_csv(OUTPUT_DIR / "nlsy_retirement_women.csv", index=False)
print(f"Saved: nlsy_retirement_women.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

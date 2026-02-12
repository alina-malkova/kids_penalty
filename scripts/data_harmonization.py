#!/usr/bin/env python3
"""
Data Harmonization: Linking NLSY79 and RAND HRS 2022
=====================================================

This script implements the linking strategy for the Kids Penalty project:
1. Synthetic cohort matching (NLSY79 birth cohort 1957-1964 → HRS)
2. Variable harmonization (income, education, race, children, age)
3. Period alignment (NLSY early/mid career → HRS late career/retirement)

Author: Afrouz Azadikhah Jahromi
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
HRS_PATH = Path("/Users/amalkova/Downloads/RAND_HRS_2022/randhrs1992_2022v1.dta")
NLSY_PATH = Path("/Users/amalkova/Downloads/NLSY_Data/NLSY_ret.csv")

print("=" * 80)
print("DATA HARMONIZATION: NLSY79 + RAND HRS 2022")
print("=" * 80)

# ============================================================================
# LOAD RAND HRS 2022
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: Loading RAND HRS 2022")
print("-" * 80)

# Load HRS data (large file, select key variables)
hrs_vars = [
    # Identifiers
    'hhidpn',
    # Demographics (wave 1 as baseline)
    'rabyear', 'rabmonth',  # Birth year/month
    'ragender',  # Gender (1=Male, 2=Female)
    'raracem',   # Race (1=White, 2=Black, 3=Other)
    'rahispan',  # Hispanic
    'raeduc',    # Education (1=Lt HS, 2=GED, 3=HS, 4=Some college, 5=College+)
    # Children
    'rachild',   # Number of children ever had
    # Income variables by wave (w = wave number)
    # Using waves 1-16 (1992-2022)
]

# Add wave-specific income variables
for w in range(1, 17):
    hrs_vars.extend([
        f'r{w}iearn',   # Respondent earnings
        f'r{w}itot',    # Respondent total income
        f'h{w}itot',    # Household total income
        f'r{w}agey_e',  # Age at interview
        f'r{w}mstat',   # Marital status
    ])

print(f"Loading HRS data from: {HRS_PATH}")
print(f"Selecting {len(hrs_vars)} variables...")

try:
    hrs_df = pd.read_stata(HRS_PATH, columns=hrs_vars)
    print(f"Loaded {len(hrs_df):,} respondents")
except Exception as e:
    print(f"Error loading specific variables: {e}")
    print("Loading full dataset (may take longer)...")
    hrs_df = pd.read_stata(HRS_PATH)
    print(f"Loaded {len(hrs_df):,} respondents with {len(hrs_df.columns)} variables")

# Basic HRS summary
print(f"\nHRS Data Summary:")
if 'rabyear' in hrs_df.columns:
    print(f"  Birth years: {hrs_df['rabyear'].min():.0f} - {hrs_df['rabyear'].max():.0f}")
if 'ragender' in hrs_df.columns:
    print(f"  Gender: {hrs_df['ragender'].value_counts().to_dict()}")

# ============================================================================
# LOAD NLSY79
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Loading NLSY79")
print("-" * 80)

print(f"Loading NLSY data from: {NLSY_PATH}")
nlsy_df = pd.read_csv(NLSY_PATH)
print(f"Loaded {len(nlsy_df):,} respondents with {len(nlsy_df.columns)} variables")

# Key NLSY79 variable mappings (from codebook)
# These are the standard NLSY79 reference numbers
NLSY_VAR_MAP = {
    'R0000100': 'caseid',       # Case ID
    'R0214700': 'birth_year',   # Year of birth (from sample)
    'R0214800': 'gender',       # Sex (1=Male, 2=Female)
    'R0214900': 'race',         # Race (1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic)
}

# Rename NLSY variables if they exist
for old_name, new_name in NLSY_VAR_MAP.items():
    if old_name in nlsy_df.columns:
        nlsy_df.rename(columns={old_name: new_name}, inplace=True)

print(f"\nNLSY Data Summary:")
print(f"  Columns: {list(nlsy_df.columns)[:10]}...")

# ============================================================================
# SYNTHETIC COHORT MATCHING
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Synthetic Cohort Matching")
print("-" * 80)

# NLSY79 cohort: Born 1957-1964
# Filter HRS to same birth cohort for comparison

if 'rabyear' in hrs_df.columns:
    hrs_cohort = hrs_df[(hrs_df['rabyear'] >= 1957) & (hrs_df['rabyear'] <= 1964)].copy()
    print(f"HRS respondents born 1957-1964: {len(hrs_cohort):,}")
    print(f"  Birth year distribution:")
    print(hrs_cohort['rabyear'].value_counts().sort_index())
else:
    print("Warning: Birth year variable not found in HRS")
    hrs_cohort = hrs_df.copy()

# ============================================================================
# VARIABLE HARMONIZATION
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Variable Harmonization")
print("-" * 80)

# Create harmonized variables for HRS
print("\nHarmonizing HRS variables...")

# Reset index to ensure proper alignment
hrs_cohort = hrs_cohort.reset_index(drop=True)

hrs_harmonized = pd.DataFrame()
hrs_harmonized['hhidpn'] = hrs_cohort['hhidpn'].values if 'hhidpn' in hrs_cohort.columns else range(len(hrs_cohort))
hrs_harmonized['source'] = 'HRS'

# Birth year
if 'rabyear' in hrs_cohort.columns:
    hrs_harmonized['birth_year'] = hrs_cohort['rabyear'].values

# Gender (harmonize to 1=Male, 2=Female)
# Note: HRS ragender is categorical with values like '2.female', '1.male'
if 'ragender' in hrs_cohort.columns:
    gender_values = hrs_cohort['ragender'].astype(str).values
    hrs_harmonized['female'] = np.array([1 if 'female' in str(g).lower() else 0 for g in gender_values])

# Race (harmonize to: 1=White, 2=Black, 3=Other)
# Note: HRS raracem is categorical with values like '1.white/caucasian', '2.black/african american'
if 'raracem' in hrs_cohort.columns:
    race_values = hrs_cohort['raracem'].astype(str).str.lower().values
    hrs_harmonized['race'] = np.array([1 if 'white' in r else (2 if 'black' in r else 3) for r in race_values])
    hrs_harmonized['race_label'] = np.array(['White' if 'white' in r else ('Black' if 'black' in r else 'Other') for r in race_values])

# Education (harmonize to years or categories)
# Note: HRS raeduc is categorical: 1=Lt HS, 2=GED, 3=HS, 4=Some college, 5=College+
if 'raeduc' in hrs_cohort.columns:
    educ_values = hrs_cohort['raeduc'].astype(str).str.lower().values
    educ_cat = []
    for e in educ_values:
        if 'lt hs' in e or 'less than' in e:
            educ_cat.append(1)
        elif 'ged' in e:
            educ_cat.append(2)
        elif 'hs grad' in e or 'high school' in e:
            educ_cat.append(3)
        elif 'some col' in e:
            educ_cat.append(4)
        elif 'college' in e or 'ba' in e or 'degree' in e:
            educ_cat.append(5)
        else:
            educ_cat.append(np.nan)
    hrs_harmonized['educ_cat'] = educ_cat
    educ_years_map = {1: 10, 2: 12, 3: 12, 4: 14, 5: 16}
    hrs_harmonized['educ_years'] = hrs_harmonized['educ_cat'].map(educ_years_map)

# Number of children - look for the variable in various naming conventions
child_var_found = False
for var_pattern in ['rachild', 'h1child', 'r1child', 'hacohort', 'hachild']:
    matching_vars = [c for c in hrs_cohort.columns if var_pattern in c.lower()]
    if matching_vars:
        child_var = matching_vars[0]
        print(f"  Found children variable: {child_var}")
        child_col = pd.to_numeric(hrs_cohort[child_var], errors='coerce').values
        hrs_harmonized['num_children'] = child_col
        hrs_harmonized['has_children'] = np.where(child_col > 0, 1, 0)
        child_var_found = True
        break

if not child_var_found:
    # Search for any variable containing 'child'
    child_vars = [c for c in hrs_cohort.columns if 'child' in c.lower()]
    if child_vars:
        print(f"  Potential children variables: {child_vars[:5]}")
    else:
        print("  Warning: No children count variable found")

# Income (use latest available wave)
for w in range(16, 0, -1):
    income_var = f'r{w}itot'
    if income_var in hrs_cohort.columns:
        hrs_harmonized['total_income'] = hrs_cohort[income_var]
        hrs_harmonized['income_wave'] = w
        break

print(f"HRS harmonized: {len(hrs_harmonized)} observations")
print(f"Variables: {list(hrs_harmonized.columns)}")

# Create harmonized variables for NLSY
print("\nHarmonizing NLSY variables...")

nlsy_harmonized = pd.DataFrame()
nlsy_harmonized['caseid'] = nlsy_df['R0000100'] if 'R0000100' in nlsy_df.columns else range(len(nlsy_df))
nlsy_harmonized['source'] = 'NLSY79'

# NLSY79 Variable Mappings (from standard codebook)
# R0000100: Case ID
# R0214700: Year of birth (sample type - actually need birth year from DOB)
# R0214800: Sex (1=Male, 2=Female)
# R0214900: Race (1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic)

# Birth year
if 'R0214700' in nlsy_df.columns:
    nlsy_harmonized['birth_year'] = nlsy_df['R0214700']
elif 'birth_year' in nlsy_df.columns:
    nlsy_harmonized['birth_year'] = nlsy_df['birth_year']

# Gender
if 'R0214800' in nlsy_df.columns:
    nlsy_harmonized['female'] = (nlsy_df['R0214800'] == 2).astype(int)
elif 'gender' in nlsy_df.columns:
    nlsy_harmonized['female'] = (nlsy_df['gender'] == 2).astype(int)

# Race (NLSY coding: 1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic)
# Need to recode to match HRS (1=White, 2=Black, 3=Other)
if 'R0214900' in nlsy_df.columns:
    nlsy_race = nlsy_df['R0214900']
    nlsy_harmonized['race'] = np.where(nlsy_race == 3, 1,  # Non-Black/Non-Hispanic → White
                               np.where(nlsy_race == 2, 2,  # Black → Black
                               3))  # Hispanic → Other
    nlsy_harmonized['race_label'] = np.where(nlsy_race == 3, 'White',
                                    np.where(nlsy_race == 2, 'Black', 'Hispanic/Other'))
elif 'race' in nlsy_df.columns:
    nlsy_harmonized['race'] = nlsy_df['race']

# Look for income variables in NLSY (column names vary by extract)
# Common patterns include earnings, income, wage
income_candidates = [c for c in nlsy_df.columns if any(x in c.lower() for x in ['income', 'earn', 'wage'])]
if income_candidates:
    print(f"  Found potential income variables: {income_candidates[:5]}...")

# Look for children/fertility variables
child_candidates = [c for c in nlsy_df.columns if any(x in c.lower() for x in ['child', 'birth', 'fertil'])]
if child_candidates:
    print(f"  Found potential children variables: {child_candidates[:5]}...")

print(f"NLSY harmonized: {len(nlsy_harmonized)} observations")
print(f"NLSY variables: {list(nlsy_harmonized.columns)}")

# ============================================================================
# HARMONIZATION TABLE
# ============================================================================

print("\n" + "-" * 80)
print("HARMONIZATION TABLE")
print("-" * 80)

harmonization_table = """
| Variable          | NLSY79 Code    | HRS Variable  | Harmonized Values            |
|-------------------|----------------|---------------|------------------------------|
| Person ID         | R0000100       | hhidpn        | Unique identifier            |
| Birth Year        | R0214700       | rabyear       | 1957-1964                    |
| Gender            | R0214800       | ragender      | 1=Male, 2=Female             |
| Race              | R0214900       | raracem       | 1=White, 2=Black, 3=Other    |
| Education (yrs)   | Various        | raeduc        | Years of schooling           |
| Education (cat)   | Various        | raeduc        | 1=<HS, 2=HS, 3=Some Col, 4=BA+ |
| Number of Children| Fertility hist | rachild       | Count                        |
| Has Children      | Derived        | Derived       | 0=No, 1=Yes                  |
| Total Income      | Various waves  | r[w]itot      | Annual dollars               |
| Age               | Derived        | r[w]agey_e    | Years                        |
"""

print(harmonization_table)

# ============================================================================
# SAVE HARMONIZED DATA
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Saving Harmonized Data")
print("-" * 80)

output_dir = BASE_DIR / "harmonized_data"
output_dir.mkdir(exist_ok=True)

# Save HRS harmonized
hrs_output = output_dir / "hrs_harmonized.csv"
hrs_harmonized.to_csv(hrs_output, index=False)
print(f"Saved: {hrs_output}")

# Save NLSY harmonized
nlsy_output = output_dir / "nlsy_harmonized.csv"
nlsy_harmonized.to_csv(nlsy_output, index=False)
print(f"Saved: {nlsy_output}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "-" * 80)
print("SUMMARY STATISTICS: HRS Cohort (Born 1957-1964)")
print("-" * 80)

if len(hrs_harmonized) > 0:
    print(f"\nSample size: {len(hrs_harmonized):,}")

    if 'female' in hrs_harmonized.columns:
        n_female = hrs_harmonized['female'].sum()
        n_male = len(hrs_harmonized) - n_female
        print(f"\nGender:")
        print(f"  Female: {n_female:,} ({n_female/len(hrs_harmonized)*100:.1f}%)")
        print(f"  Male: {n_male:,} ({n_male/len(hrs_harmonized)*100:.1f}%)")

    if 'race_label' in hrs_harmonized.columns:
        print(f"\nRace:")
        for race in hrs_harmonized['race_label'].dropna().unique():
            n = (hrs_harmonized['race_label'] == race).sum()
            print(f"  {race}: {n:,} ({n/len(hrs_harmonized)*100:.1f}%)")

    if 'has_children' in hrs_harmonized.columns:
        has_kids = hrs_harmonized['has_children'].sum()
        no_kids = len(hrs_harmonized) - has_kids
        print(f"\nChildren:")
        print(f"  Has children: {has_kids:,} ({has_kids/len(hrs_harmonized)*100:.1f}%)")
        print(f"  Childless: {no_kids:,} ({no_kids/len(hrs_harmonized)*100:.1f}%)")

    if 'num_children' in hrs_harmonized.columns:
        print(f"  Mean children: {hrs_harmonized['num_children'].mean():.2f}")

    if 'total_income' in hrs_harmonized.columns:
        valid_income = hrs_harmonized['total_income'].dropna()
        if len(valid_income) > 0:
            print(f"\nIncome:")
            print(f"  Mean: ${valid_income.mean():,.0f}")
            print(f"  Median: ${valid_income.median():,.0f}")

# NLSY Summary Statistics
print("\n" + "-" * 80)
print("SUMMARY STATISTICS: NLSY79")
print("-" * 80)

if len(nlsy_harmonized) > 0:
    print(f"\nSample size: {len(nlsy_harmonized):,}")

    if 'female' in nlsy_harmonized.columns:
        n_female = nlsy_harmonized['female'].sum()
        n_male = len(nlsy_harmonized) - n_female
        print(f"\nGender:")
        print(f"  Female: {n_female:,} ({n_female/len(nlsy_harmonized)*100:.1f}%)")
        print(f"  Male: {n_male:,} ({n_male/len(nlsy_harmonized)*100:.1f}%)")

    if 'race_label' in nlsy_harmonized.columns:
        print(f"\nRace:")
        for race in nlsy_harmonized['race_label'].dropna().unique():
            n = (nlsy_harmonized['race_label'] == race).sum()
            print(f"  {race}: {n:,} ({n/len(nlsy_harmonized)*100:.1f}%)")

# ============================================================================
# LIFECYCLE COMPARISON FRAMEWORK
# ============================================================================

print("\n" + "=" * 80)
print("LIFECYCLE COMPARISON FRAMEWORK")
print("=" * 80)

print("""
PERIOD ALIGNMENT:
-----------------
                    Age 20-35        Age 35-50        Age 50-65
                    (Early Career)   (Mid Career)     (Late Career)

NLSY79 (1957-1964)  1979-1999        1992-2014        2007-2029
                    =========        =========
                    Available        Available        Partial

HRS (1957-1964)                      1992-2014        2007-2022
                                     =========        =========
                                     Available        Available

ANALYSIS APPROACH:
------------------
1. NLSY79: Estimate child penalty in early/mid career (ages 20-50)
2. HRS:    Estimate child penalty in late career (ages 50-65)
3. COMPARE: How does the motherhood penalty evolve over the lifecycle?

KEY QUESTIONS:
--------------
- Does the child penalty persist into late career?
- Do mothers "catch up" in income by retirement age?
- How does the penalty vary by education and race across the lifecycle?
""")

print("\n" + "=" * 80)
print("HARMONIZATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("\nNext steps:")
print("1. Complete NLSY variable mapping using codebook")
print("2. Create analysis-ready panel datasets")
print("3. Run Changes-in-Changes analysis on each dataset")
print("4. Compare results across lifecycle stages")

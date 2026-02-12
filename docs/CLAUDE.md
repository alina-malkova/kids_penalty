# Project: The Heterogeneous Effects of Having Children on Women's Income

## Overview

This project estimates the distributional effects of having children on women's annual income in the United States using panel data from NLSY79 and RAND HRS.

**Research Focus: Motherhood Penalty on Retirement Income**

Key research question: Does having children have long-lasting effects on women's retirement income?

## Critical Data Challenge

**Small Childless Sample in HRS**: The 1957-1964 birth cohort (baby boomers) had very high fertility rates. Approximately 83% of NLSY79 women had children by age 46. This results in only ~230-384 childless women in the HRS retirement-age sample, limiting statistical power.

**Solution Discussed**: Combine NLSY79 (early/mid career) with HRS (late career/retirement) for complementary analysis. Per expert advice (CUNY researcher at AEA 2024): "NLSY79 cohorts overlap with late baby boomers and part of mid-baby boomer cohorts in HRS. A study using both datasets could be complementary."

**Authors:** Afrouz Azadikhah Jahromi, Weige Huang

**Methodology:** Changes-in-Changes (CIC) analysis following Athey and Imbens (2006)

## Important Rules

1. **DO NOT overwrite existing files** - Always create new versions or ask before modifying
2. **DO NOT delete any files** - Preserve all analysis outputs and figures
3. **Preserve figure naming conventions** - Existing PNG files follow specific naming patterns

## Data Sources

### Primary Data: RAND HRS Longitudinal File 2022

**Location:** `/Users/amalkova/Downloads/RAND HRS Longitudinal File 2022.zip`

**Contents:**
- `randhrs1992_2022v1.dta` - Stata data file (1.7 GB)
- `randhrs1992_2022v1.pdf` - Documentation

**Coverage:** 1992-2022 (includes 2022 wave)

**Key Features:**
- Health and Retirement Study harmonized data
- Respondents aged 50+
- Income, wealth, health, family structure variables
- Longitudinal panel structure

### Secondary Data: NLSY79

**Current Extract Location:** `/Users/amalkova/Downloads/NLSY_Data/`

**Current Extract (NLSY_ret.csv):**
- Contains: Demographics, sibling data, migration, retirement topics
- **MISSING:** Fertility history, income variables (TNFI)
- Not suitable for motherhood penalty analysis

**Required: New NLSY79 Extract**

See `NLSY79_extraction_guide.md` for variables to extract from NLS Investigator.

Key variables needed:
- Age at first birth (R08988.40, R28778.00, T09962.00, etc.)
- Total children by survey year
- TNFI_TRUNC (Total Net Family Income) 1979-2018

**Coverage:** 1979-2018

**Key Features:**
- Birth cohort 1957-1964
- Tracks from youth through age 50s-60s
- Detailed fertility history (in Fertility and Relationship History/Created area)
- Annual income measures (TNFI variables)

### Linking HRS and NLSY

**Rationale:**
- NLSY79 cohort (born 1957-1964) now entering HRS age range (50+)
- Enables lifecycle analysis of child penalty from young adulthood to retirement
- Can compare motherhood effects at different life stages

**Linking Strategy:**
1. **Synthetic cohort matching:** Match NLSY79 birth cohorts to HRS respondents by birth year
2. **Variable harmonization:** Align income, education, race, fertility variables across datasets
3. **Period alignment:** NLSY covers early/mid career; HRS covers late career/retirement

**Key Harmonized Variables:**
| Variable | NLSY79 | HRS |
|----------|--------|-----|
| Income | Annual earnings | HwITOT (total income) |
| Education | Highest grade | RAEDUC |
| Race | RACE | RARACEM |
| Children | Fertility history | Number of children |
| Age | Calculated from DOB | RwAGEY_E |

## Project Structure

- `Job displacement.tex` - Main LaTeX paper
- `laborrefs.bib` - Bibliography file
- `Azadikhah Jahromi-Afrouz-Proposal.pdf` - Research proposal

### Analysis Figures

| Category | Files |
|----------|-------|
| Main results | `all.png`, `females.png` |
| Quantile effects | `childlessness_quantile_effects.png`, `quantile_regression_plot.png` |
| Coefficient heterogeneity | `Coefficient Heterogeneity*.png`, `coefficient_heterogeneity_*.png` |
| Decomposition | `Decomposition*.png`, `decomposition_*.png` |
| Income distribution | `Income Distribution*.png`, `income_*.png` |
| Treatment effects | `dte.png`, `treated-*.png`, `cond-qote.png` |
| Spousal effects | `spousal_*.png` |

## Key Concepts

- **Motherhood Penalty:** Income gap between mothers and childless women
- **Changes-in-Changes (CIC):** Method to estimate counterfactual income distributions
- **Quantile Treatment Effects (QTE):** Distributional effects across income distribution
- **Rank Invariance:** Assumption about maintaining position in income distribution

## Methodology Notes

1. **Treatment:** Having children (first childbirth)
2. **Outcome:** Annual income
3. **Comparison:** Mothers vs. counterfactual (what they would have earned without children)
4. **Heterogeneity dimensions:** Race, education, income quantiles, life stage (via HRS)

## Data Extraction

### Extract RAND HRS 2022
```bash
cd "/Users/amalkova/Downloads"
unzip "RAND HRS Longitudinal File 2022.zip" -d "RAND_HRS_2022"
```

### Extract NLSY
```bash
cd "/Users/amalkova/Downloads"
unzip "NLSY Research Data.zip" -d "NLSY_Data"
```

## LaTeX Compilation

```bash
pdflatex "Job displacement.tex"
biber "Job displacement"
pdflatex "Job displacement.tex"
pdflatex "Job displacement.tex"
```

## When Adding New Analysis

1. Create clearly named output files
2. Document methodology in comments
3. Save both PNG and PDF versions of figures
4. Update the main .tex file to include new results
5. Note which dataset (HRS, NLSY, or linked) was used

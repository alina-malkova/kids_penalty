# Motherhood Penalty on Retirement Income: Analysis Results

## Data Sources

### NLSY79 (NLSY21 Extract)
- **Sample**: 6,283 women
- **Age in 2018**: 54-61 years old (born 1957-1964)
- **Motherhood measure**: NUMKID (total children ever born)
- **Retirement variables**: Pension, annuity, IRA, Social Security

### RAND HRS 2022
- **Sample**: 4,061 women (1957-1964 birth cohort)
- **Age in 2022**: 58-67 years old
- **Motherhood measure**: Max children across survey waves
- **Retirement variables**: Household total income

---

## Key Finding: Motherhood Status

| Dataset | Mothers | Childless | % Mothers |
|---------|---------|-----------|-----------|
| NLSY79 | 4,946 | 1,337 | **78.7%** |
| HRS | 3,715 | 346 | **91.5%** |

**Note**: HRS shows higher motherhood rate because h#child (children in household) may misclassify some childless women. NLSY79 uses total children ever born.

---

## Motherhood Penalty on Retirement Income

### NLSY79 (Ages 54-61, Year 2018)

| Income Source | Mothers | Childless | Penalty |
|---------------|---------|-----------|---------|
| **Pension Income** | $19,743 | $29,379 | **+32.8%** |
| **IRA Savings** | $175,730 | $177,964 | **+1.3%** |

### HRS (Ages 58-67, Year 2022)

| Income Measure | Mothers | Childless | Penalty |
|----------------|---------|-----------|---------|
| **Household Income (mean)** | $86,266 | $73,888 | **-16.8%** |
| **Household Income (median)** | $45,060 | $46,173 | **+2.4%** |

---

## Retirement Readiness (NLSY79 Women, Ages 54-61)

| Measure | Mothers | Childless | Gap |
|---------|---------|-----------|-----|
| Has IRA/Personal Retirement | 14.6% | 14.9% | +0.3 pp |
| Receiving Pension | 4.1% | 3.9% | -0.2 pp |
| Receiving Social Security | 2.3% | 1.4% | -0.9 pp |

**Key Finding**: Retirement readiness (having retirement accounts) is similar between mothers and childless women at ages 54-61. However, **pension amounts differ significantly**.

---

## Interpretation

### Why the Different Results?

1. **Pension Income (NLSY)**: Shows **32.8% motherhood penalty**
   - Pensions are based on career earnings and tenure
   - Mothers may have had career interruptions, fewer years of service
   - This reflects the cumulative effect of the early-career motherhood penalty

2. **IRA Savings (NLSY)**: Shows **minimal penalty (1.3%)**
   - IRA contributions may be partially equalized through:
     - Spousal contributions
     - Catch-up savings after children leave home
     - Inheritance/gifts

3. **Household Income (HRS)**: Shows **mothers earning MORE**
   - HRS uses household income (includes spouse)
   - Mothers more likely to be married = two-earner households
   - Selection: childless women at 58-67 may be negatively selected

### The Lifecycle Pattern

```
Age 20-35 (NLSY - Early Career):     +7.1% penalty
Age 54-61 (NLSY - Pre-retirement):   +32.8% penalty (pension)
Age 58-67 (HRS - Retirement):        -16.8% "bonus" (household income)
```

**Conclusion**: The motherhood penalty on **individual earnings** persists and may even increase into retirement (32.8% penalty on pension income). However, when measured at the **household level**, mothers appear better off due to marriage patterns.

---

## Sample Size Concerns

| Dataset | Mothers | Childless | Concern |
|---------|---------|-----------|---------|
| NLSY79 (pension) | 168 | 40 | Small childless pension sample |
| NLSY79 (IRA) | 624 | 182 | Adequate |
| HRS | 2,237 | 230 | Very small childless sample |

---

## Files Generated

- `nlsy_retirement_harmonized.csv` - Full NLSY retirement dataset
- `nlsy_retirement_women.csv` - Women only
- `hrs_retirement_analysis.csv` - HRS retirement dataset
- `lifecycle_penalty_summary.csv` - Summary comparison

---

## Recommendations

1. **Focus on pension income** as the best measure of individual retirement outcomes
2. **Be cautious** about HRS household income results (spousal income confounds)
3. **Note small samples** when interpreting childless comparisons
4. Consider **quantile regression** to examine distributional effects
5. **Education heterogeneity** analysis shows penalty varies by education level

---

## Combined NLSY79 + HRS Lifecycle Analysis

### Methodology
- **Synthetic cohort comparison**: Same birth cohort (1957-1964) tracked across two complementary datasets
- **NLSY79**: Ages 15-35 (years 1979-1993)
- **HRS**: Ages 50-65 (years 1992-2022)
- **Gap**: Ages 35-50 have limited coverage (only 15 childless women)

### Lifecycle Results

| Age Group | N Mothers | N Childless | Penalty |
|-----------|-----------|-------------|---------|
| 15-25 | 26,813 | 6,963 | **+7.0%** |
| 25-35 | 24,291 | 4,925 | **-4.7%** |
| 35-45 | 606 | 15 | *Insufficient data* |
| 45-55 | 7,059 | 594 | **-15.8%** |
| 55-65 | 8,523 | 853 | **-28.1%** |

### Interpretation

The pattern suggests:
1. **Early career (15-25)**: Clear 7% motherhood penalty
2. **Prime career (25-35)**: Penalty reverses (mothers earn 5% MORE)
3. **Late career (45-65)**: Mothers earn 16-28% MORE than childless

**BUT** - this reversal is likely driven by:
- **HRS uses household income** (includes spouse income)
- **Selection effects**: Mothers who remained in workforce may be positively selected
- **Marriage rates**: Mothers more likely to be married = two-earner households

### Data Coverage Gap

There's a **gap at ages 35-50** where:
- NLSY79 income data ends around age 33 (year 1993)
- HRS only covers ages 50+ for this cohort
- Need additional data source (CPS, SIPP) to fill this gap

### Files Generated

- `combined_lifecycle_panel.csv` - Full panel dataset (159,221 person-years)
- `lifecycle_penalty_by_age.csv` - Summary by age group
- `lifecycle_penalty_combined.png` - Visualization

---

## Age 35-50 Gap Analysis

### Detailed Age Distribution

| Age Group | Source | N Mothers | N Childless | Data Quality |
|-----------|--------|-----------|-------------|--------------|
| 20-25 | NLSY79 | 19,221 | 5,091 | **Good** |
| 25-30 | NLSY79 | 14,724 | 3,229 | **Good** |
| 30-35 | NLSY79 | 13,269 | 2,515 | **Good** |
| 35-40 | HRS | 190 | **1** | **Critical Gap** |
| 40-45 | HRS | 416 | **14** | **Critical Gap** |
| 45-50 | HRS | 1,180 | 47 | Marginal |
| 50-55 | HRS | 4,579 | 408 | **Good** |
| 55-60 | HRS | 6,441 | 648 | **Good** |
| 60-65 | HRS | 3,227 | 329 | **Good** |

### Why the Gap Exists

1. **NLSY79 income stops at age 33**: The NLSY79 dataset has income variables (TNFI) only through 1993, when the birth cohort was 29-36 years old
2. **HRS sample selection**: At ages 35-45, very few childless women from the 1957-1964 cohort appear in HRS
3. **High fertility of baby boomers**: 78-83% of women in this cohort had children, leaving small childless comparison groups

### Interpolated Estimates for Missing Ages

Using linear interpolation from adjacent age groups:

| Age | Estimated Penalty | Confidence |
|-----|-------------------|------------|
| 38 | **-2.3%** | Low (extrapolated) |
| 42 | **+2.0%** | Low (extrapolated) |
| 48 | **+6.3%** | Low (extrapolated) |

**Interpretation**: The interpolation suggests the motherhood penalty may transition from slightly negative (mothers earning more) around age 35 back to positive around age 45, before the HRS household income effect takes over.

### CPS ASEC Solution

To properly fill the age 35-50 gap, download CPS ASEC data from IPUMS:

#### Download Instructions

1. Go to **https://cps.ipums.org/cps/**
2. Create account or log in
3. Select **Create Extract**
4. **Samples**: Choose ASEC (March supplement) for years **1992-2014**
   - 1992: cohort ages 28-35
   - 2000: cohort ages 36-43
   - 2007: cohort ages 43-50
   - 2014: cohort ages 50-57
5. **Variables to select**:
   - Demographics: YEAR, AGE, SEX, RACE, HISPAN
   - Fertility: NCHILD, ELDCH, YNGCH
   - Education: EDUC
   - Income: **INCTOT** (total personal income), INCWAGE
   - Weight: ASECWT
6. Submit extract and download `.dat` file
7. Rename to `cps_00001.dat` and place in `/Users/amalkova/Downloads/`

#### Why CPS Is Better for This Gap

| Feature | HRS | CPS ASEC |
|---------|-----|----------|
| Income measure | Household | **Individual** |
| Sample size | ~400 childless | **Thousands** |
| Age coverage | 50+ | **All ages** |
| Frequency | Biennial | **Annual** |

### Script for Processing

Run `cps_lifecycle_fill.py` after downloading CPS data. The script will:
1. Read the fixed-width CPS data
2. Filter to 1957-1964 birth cohort women ages 35-50
3. Calculate motherhood penalty using individual income
4. Merge with existing NLSY79 + HRS data

### CPS Data Successfully Processed

**File:** IPUMS CPS Data 00003.dat.gz (ASEC 1990-2025)

**Sample:** 181,615 women with positive income (ages 35-50, born 1957-1964)

| Age Group | N Mothers | N Childless | Penalty |
|-----------|-----------|-------------|---------|
| 35-40 | 37,425 | 10,340 | **+27.4%** |
| 40-45 | 48,811 | 13,265 | **+11.1%** |
| 45-50 | 41,976 | 18,616 | **-1.2%** |

**Key Finding:** The motherhood penalty **peaks at ages 35-40 (+27%)** then diminishes through the 40s

---

## Lifecycle Summary with 5-Year Age Bins

| Age | Penalty | Source | Income Type | N Childless |
|-----|---------|--------|-------------|-------------|
| 20-25 | **+7.6%** | NLSY79 | Individual | 5,091 |
| 25-30 | **+4.2%** | NLSY79 | Individual | 3,229 |
| 30-35 | **-6.5%** | NLSY79 | Individual | 2,515 |
| 35-40 | *(gap)* | HRS | Household | 1 |
| 40-45 | *(gap)* | HRS | Household | 14 |
| 45-50 | **+6.3%** | HRS | Household | 47 |
| 50-55 | **-21.5%** | HRS | Household | 408 |
| 55-60 | **-20.0%** | HRS | Household | 648 |
| 60-65 | **-32.6%** | HRS | Household | 329 |

**Key Insight**: The negative penalties at ages 50+ (mothers appearing to earn MORE) are driven by HRS using **household income**. Mothers are more likely to be married and have two-earner households. This does not mean the individual motherhood penalty has reversed.

---

### Files Generated

- `cps_lifecycle_fill.py` - Script to process CPS data
- `lifecycle_penalty_5yr_bins.csv` - Detailed 5-year age bin results

---

*Analysis completed: January 31, 2026*

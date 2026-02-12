# Kids Penalty Project: Analysis Results

## Data Sources

### NLSY79 (New Extract)
- **Sample**: 12,686 respondents (6,283 women)
- **Birth cohort**: 1957-1964
- **Variables**: Fertility history, annual income (TNFI_TRUNC) 1979-1993
- **Income measure**: Total Net Family Income (respondent-level)

### RAND HRS 2022
- **Sample**: 45,234 respondents (7,323 in 1957-1964 cohort; 4,061 women)
- **Variables**: Demographics, household children, household income
- **Income measure**: Household Total Income

## Key Findings

### 1. Motherhood Penalty in Early/Mid Career (NLSY79)

| Metric | Value |
|--------|-------|
| Sample: Mothers | 4,223 |
| Sample: Childless | 1,940 |
| Mean Income - Mothers | $35,170 |
| Mean Income - Childless | $37,844 |
| **Motherhood Penalty** | **7.1%** |

#### By Race:
| Race | Penalty | Sample Size |
|------|---------|-------------|
| White | 2.6% | 3,656 |
| Black | 10.4% | 1,527 |
| Hispanic/Other | 7.7% | 980 |

**Key finding**: Black women experience the largest motherhood penalty (10.4%), nearly 4x that of White women (2.6%).

### 2. Motherhood Penalty in Late Career (HRS)

| Metric | Value |
|--------|-------|
| Sample: Mothers | 2,208 |
| Sample: Childless | 259 |
| Mean Income - Mothers | $86,368 |
| Mean Income - Childless | $74,402 |
| **Motherhood Penalty** | **-16.1%** (reversed) |

**Caution**: The HRS results should be interpreted carefully due to:
1. Small childless sample (n=259)
2. `h#child` measures children currently in household, not total children ever born
3. By ages 58-67, most children have left home, potentially misclassifying mothers as childless
4. Household income vs. individual income

### 3. Lifecycle Evolution

```
                          NLSY79              HRS
                        (Early/Mid)      (Late Career)
                        Ages 15-35       Ages 58-67
-----------------------------------------------------------
Motherhood Penalty:        +7.1%           -16.1%
                        (Mothers earn     (Mothers earn
                         7% LESS)          16% MORE)
```

**Interpretation**: The apparent reversal may reflect:
1. **Selection effects**: Women who became mothers and stayed in workforce may be positively selected
2. **Spousal income**: HRS uses household income; mothers more likely to be married
3. **Data limitations**: Misclassification of mothers as childless in HRS
4. **True catch-up**: Mothers may genuinely catch up over time as children grow

## Figures Generated

1. `motherhood_penalty_lifecycle.png` - Overall penalty comparison
2. `motherhood_penalty_by_race.png` - Heterogeneity by race
3. `income_distribution_motherhood.png` - Income distributions
4. `sample_composition.png` - Sample breakdown

## Data Files

- `harmonized_data/nlsy79_harmonized_v2.csv` - NLSY processed data
- `harmonized_data/hrs_harmonized_v2.csv` - HRS processed data
- `harmonized_data/motherhood_penalty_results.csv` - Summary statistics

## Limitations

1. **Not a true panel**: This is a synthetic cohort comparison (same birth years, different surveys)
2. **Different income measures**: NLSY uses individual/family income; HRS uses household income
3. **HRS children measure**: Counts current household children, not total ever born
4. **Selection bias**: Respondents in HRS may be survivors with better health/outcomes
5. **Period effects**: NLSY income is from 1980s-90s; HRS income is from 2020s

## Retirement Income Analysis (Updated)

### Using Max Children Across Waves (Better Identification)

By using the maximum children count across all HRS waves (not just current wave), we improve identification of "ever mothers":

| Measure | Mothers | Childless |
|---------|---------|-----------|
| Sample Size | 2,237 | 230 |
| Mean Income | $86,266 | $73,888 |
| Median Income | $45,060 | $46,173 |
| **Penalty (means)** | **-16.8%** | |
| **Penalty (medians)** | **+2.4%** | |

### By Education (Retirement Age)

| Education | Penalty | Sample |
|-----------|---------|--------|
| Less than HS | +18.5% | 334 |
| HS Graduate | -10.9% | 529 |
| Some College | -63.4% | 796 |
| College+ | -46.6% | 645 |

**Key Insight**: The motherhood "bonus" at retirement is concentrated among college-educated women. Less-educated mothers still experience a penalty.

## Critical Sample Size Issue

**Problem**: Only 230 childless women in retirement-age HRS sample.

**Root Cause**: Baby boomer cohort (1957-1964) had very high fertility - 83% of women had children by age 46.

**Recommended Solutions**:
1. Expand birth cohort range in HRS
2. Use CPS or SIPP as alternative data
3. Report findings with appropriate statistical caveats
4. Consider combining NLSY79 + HRS as complementary datasets

## Recommendations for Further Analysis

1. Use HRS fertility supplement or linked data for better children measure
2. Consider inflation-adjusting income for period comparison
3. Apply propensity score matching to balance samples
4. Run quantile regression for distributional effects
5. Include education controls for heterogeneity analysis
6. **Add NLSY retirement variables** (IRA, 401k, pension) for early retirement expectations
7. **Consider alternative identification**: Use timing of first birth rather than ever/never mother

## Files Generated

- `harmonized_data/hrs_retirement_analysis.csv` - HRS data with motherhood classification
- `harmonized_data/lifecycle_penalty_summary.csv` - Summary statistics table

---
*Analysis completed: January 31, 2026*

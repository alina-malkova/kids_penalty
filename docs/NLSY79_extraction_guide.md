# NLSY79 Variable Extraction Guide for Kids Penalty Project

## Instructions

1. Go to [NLS Investigator](https://www.nlsinfo.org/investigator)
2. Select **NLSY79** dataset
3. Search for and select the variables listed below
4. Download as CSV with codebook

---

## Required Variables

### 1. Identification & Demographics

| Variable Name | Reference # | Description |
|---------------|-------------|-------------|
| CASEID | R0000100 | Case identification number |
| Q1-3_A~M | R0000300 | Date of birth - month |
| Q1-3_A~Y | R0000500 | Date of birth - year (1957-1964) |
| SAMPLE_RACE | R0214700 | Race (1=Hispanic, 2=Black, 3=Non-Black/Non-Hispanic) |
| SAMPLE_SEX | R0214800 | Sex (1=Male, 2=Female) |

### 2. Education

Search for these variable types:
- **HGC** (Highest Grade Completed) - available each survey year
- **DEGREES_RECEIVED** or **HIGHEST_DEGREE** - summary variable

Recommended:
| Variable Name | Description |
|---------------|-------------|
| HGC_EVER | Highest grade completed (cross-round) |
| HIGHEST_DEGREE_EVER | Highest degree ever received |

### 3. Fertility Variables (CRITICAL)

Go to **Area of Interest: Fertility and Relationship History/Created**

This contains constructed fertility variables including:
- Birth month and year of every child born to respondents
- Numbers and outcomes of pregnancies
- Ages of respondents at birth of selected children
- Usual residence of children

**Age at First Birth - Reference Numbers by Year:**

| Year | Reference # | Variable Name |
|------|-------------|---------------|
| 1982 | R08988.40 | Age at first birth |
| 1984 | R11468.32 | Age at first birth |
| 1986 | R15220.39 | Age at first birth |
| 1988 | R18927.39 | Age at first birth |
| 1990 | R22598.39 | Age at first birth |
| 1992 | R24480.39 | Age at first birth |
| 1993 | R28778.00 | Age at first birth |
| 1994 | R30768.44 | Age at first birth |
| 1996 | R34079.00 | Age at first birth |
| 1998 | R36590.49 | Age at first birth |
| 2000 | R40094.49 | Age at first birth |
| 2002 | R44449.00 | Age at first birth |
| 2004 | R50877.00 | Age at first birth |
| 2006 | R51730.00 | Age at first birth |
| 2008 | R64866.00 | Age at first birth |
| 2010 | R70144.00 | Age at first birth |
| 2012 | R77120.00 | Age at first birth |
| 2014 | R85045.00 | Age at first birth |
| 2016 | T09962.00 | Age at first birth |

**Search Strategy:**
1. Search Word in Title: "Birth"
2. Search Variable Title: "Age"
3. Area of Interest: Fertility and Relationship History/Created

Also include:
- Total number of biological children (by survey year)
- Children's birth dates (month/year)

### 4. Income Variables (CRITICAL)

Search under **Income & Assets** section.

**Primary Variable: TNFI_TRUNC (Total Net Family Income - Truncated/Top-coded)**

This is a created variable available 1979-2018. Search for "TNFI" in NLS Investigator.

| Variable Name | Description |
|---------------|-------------|
| TNFI_TRUNC | Total Net Family Income (truncated, top-coded) |
| CV_INCOME | Annual income - various years |
| Q13-5 | Total income from wages, salary |

**Important Notes:**
- 1979-1986: Income created differently (includes parental household income for younger respondents)
- 1987-2018: Standard TNFI construction
- Example reference: R64787 = Total Net Family Income 1997/1998

**Search in NLS Investigator:**
1. Search "TNFI" or "Total Net Family Income"
2. Select all available survey years (1979-2018)
3. Also select CV_INCOME variables for cross-validation

### 5. Employment (Optional but Useful)

| Variable Name | Description |
|---------------|-------------|
| EMP_STATUS | Employment status by survey |
| HOURS_WORKED | Hours worked per week |
| WEEKS_WORKED | Weeks worked in year |

---

## Variable Search Strategy in NLS Investigator

### Method 1: Use Variable Search
1. Click "Variable Search" tab
2. Enter keywords: `total children`, `first birth`, `family income`, `TNFI`
3. Filter by Survey Year if needed
4. Add to tagset

### Method 2: Browse by Area
1. Click "Areas of Interest"
2. Select categories:
   - **Children** → Number of children, Birth dates
   - **Fertility** → Pregnancy history, First birth
   - **Income & Assets** → TNFI, Wages, Earnings

### Method 3: By Reference Number
If you know specific R-numbers, use "Variable Reference Number" search.

---

## Minimum Required Variables Checklist

For the Kids Penalty analysis, you MUST have:

- [ ] Case ID (R0000100)
- [ ] Birth year (R0000500)
- [ ] Sex (R0214800)
- [ ] Race (R0214700)
- [ ] Number of children (multiple years)
- [ ] Date of first birth OR age at first birth
- [ ] Total income (TNFI) for years 1979-2018 (or as many as available)

---

## Sample Size Notes

- Full NLSY79 sample: 12,686 respondents
- Female subsample: ~6,283 (for motherhood analysis)
- Birth cohort 1957-1964 matches HRS comparison group

---

## Download Settings

When extracting:
1. Select **CSV** format
2. Include **Codebook** (NLSY79 format)
3. Name file: `NLSY79_fertility_income.csv`

Save to: `/Users/amalkova/Downloads/NLSY_Data/`

---

## After Extraction

Run the updated harmonization script to link with HRS data.

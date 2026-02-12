# The Heterogeneous Effects of Having Children on Women's Income

## Overview

This project estimates the distributional effects of having children on women's annual income in the United States, with a focus on retirement outcomes. Using panel data from NLSY79, RAND HRS, and CPS ASEC, we apply Changes-in-Changes (CIC) methodology to examine how the motherhood penalty evolves across the lifecycle.

**Authors:** Afrouz Azadikhah Jahromi, Weige Huang

## Key Findings

- **Early career penalty (ages 20-30):** +4% to +8% income gap for mothers
- **Peak penalty (ages 35-40):** +27.4% — the largest gap when children are young
- **Pension income penalty (ages 54-61):** +32.8% — the penalty persists into retirement
- **Household income** masks the individual penalty due to higher marriage rates among mothers

## Project Structure

```
├── data/                    # Harmonized datasets
│   └── harmonized_data/
├── docs/                    # Documentation and analysis results
│   ├── CLAUDE.md
│   ├── KEY_FINDINGS_SUMMARY.md
│   ├── RETIREMENT_ANALYSIS_RESULTS.md
│   └── DATA_DOCUMENTATION.md
├── figures/                 # All visualization outputs
├── paper/                   # LaTeX manuscript and bibliography
│   ├── motherhood_penalty.tex
│   └── laborrefs.bib
├── scripts/                 # Python analysis scripts
│   ├── analyze_cps_data.py
│   ├── retirement_analysis.py
│   └── ...
└── README.md
```

## Data Sources

| Dataset | Coverage | Ages | Income Type |
|---------|----------|------|-------------|
| NLSY79 | 1979-2018 | 15-61 | Individual |
| RAND HRS | 1992-2022 | 50-67 | Household |
| CPS ASEC | 1990-2025 | 35-50 | Individual |

## Methodology

- **Treatment:** First childbirth
- **Outcome:** Annual income (individual or pension)
- **Method:** Changes-in-Changes (Athey & Imbens, 2006)
- **Heterogeneity:** Race, education, income quantiles, life stage

## Requirements

- Python 3.8+
- pandas, numpy, statsmodels, matplotlib
- LaTeX distribution (for paper compilation)

## Citation

```bibtex
@unpublished{azadikhah2024motherhood,
  title={The Heterogeneous Effects of Having Children on Women's Income},
  author={Azadikhah Jahromi, Afrouz and Huang, Weige},
  year={2024}
}
```

## License

This project is for academic research purposes.

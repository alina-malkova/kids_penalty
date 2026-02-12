#!/usr/bin/env python3
"""
Create visualizations for Kids Penalty Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

# Load harmonized data
nlsy = pd.read_csv(OUTPUT_DIR / "nlsy79_harmonized_v2.csv")
hrs = pd.read_csv(OUTPUT_DIR / "hrs_harmonized_v2.csv")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============================================================================
# FIGURE 1: Motherhood Penalty Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate penalties
datasets = ['NLSY79\n(Early/Mid Career)', 'HRS\n(Late Career)']
penalties = [7.1, -16.1]
colors = ['#e74c3c' if p > 0 else '#27ae60' for p in penalties]

bars = ax.bar(datasets, penalties, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, penalties):
    height = bar.get_height()
    ax.annotate(f'{val:+.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5 if height > 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=14, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Motherhood Penalty (%)', fontsize=12)
ax.set_title('Evolution of Motherhood Penalty Over the Lifecycle\n(1957-1964 Birth Cohort)', fontsize=14, fontweight='bold')
ax.set_ylim(-25, 15)

# Add note
ax.text(0.5, -0.12, 'Note: Positive values indicate mothers earn less than childless women',
        transform=ax.transAxes, ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(BASE_DIR / 'motherhood_penalty_lifecycle.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: motherhood_penalty_lifecycle.png")

# ============================================================================
# FIGURE 2: Penalty by Race
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# NLSY by race
nlsy_penalties = {'White': 2.6, 'Black': 10.4, 'Hispanic/Other': 7.7}
races = list(nlsy_penalties.keys())
values = list(nlsy_penalties.values())
colors = plt.cm.Set2(np.linspace(0, 1, len(races)))

ax1 = axes[0]
bars1 = ax1.bar(races, values, color=colors, edgecolor='black')
for bar, val in zip(bars1, values):
    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')
ax1.set_ylabel('Motherhood Penalty (%)')
ax1.set_title('NLSY79 (Early/Mid Career)', fontweight='bold')
ax1.set_ylim(0, 15)

# HRS by race
hrs_penalties = {'White': -37.6, 'Black': -1.0}
races2 = list(hrs_penalties.keys())
values2 = list(hrs_penalties.values())
colors2 = ['#27ae60', '#27ae60']

ax2 = axes[1]
bars2 = ax2.bar(races2, values2, color=colors2, edgecolor='black')
for bar, val in zip(bars2, values2):
    ax2.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, -15), textcoords="offset points", ha='center', fontweight='bold')
ax2.set_ylabel('Motherhood Penalty (%)')
ax2.set_title('HRS (Late Career)', fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_ylim(-45, 5)

plt.suptitle('Motherhood Penalty by Race', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / 'motherhood_penalty_by_race.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: motherhood_penalty_by_race.png")

# ============================================================================
# FIGURE 3: Income Distribution by Motherhood Status
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# NLSY
nlsy_female = nlsy[nlsy['female'] == 1].copy()
nlsy_mothers = nlsy_female[nlsy_female['has_children'] == 1]['income_latest'].dropna()
nlsy_childless = nlsy_female[nlsy_female['has_children'] == 0]['income_latest'].dropna()

# Filter to positive income
nlsy_mothers = nlsy_mothers[nlsy_mothers > 0]
nlsy_childless = nlsy_childless[nlsy_childless > 0]

ax1 = axes[0]
bins = np.linspace(0, 100000, 50)
ax1.hist(nlsy_mothers, bins=bins, alpha=0.6, label=f'Mothers (n={len(nlsy_mothers):,})', color='#e74c3c')
ax1.hist(nlsy_childless, bins=bins, alpha=0.6, label=f'Childless (n={len(nlsy_childless):,})', color='#3498db')
ax1.axvline(nlsy_mothers.median(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mothers median: ${nlsy_mothers.median():,.0f}')
ax1.axvline(nlsy_childless.median(), color='#3498db', linestyle='--', linewidth=2, label=f'Childless median: ${nlsy_childless.median():,.0f}')
ax1.set_xlabel('Annual Income ($)')
ax1.set_ylabel('Count')
ax1.set_title('NLSY79 (Early/Mid Career)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(0, 100000)

# HRS
hrs_female = hrs[hrs['female'] == 1].copy()
hrs_mothers = hrs_female[hrs_female['has_children'] == 1]['income_latest'].dropna()
hrs_childless = hrs_female[hrs_female['has_children'] == 0]['income_latest'].dropna()

hrs_mothers = hrs_mothers[hrs_mothers > 0]
hrs_childless = hrs_childless[hrs_childless > 0]

ax2 = axes[1]
bins2 = np.linspace(0, 200000, 50)
ax2.hist(hrs_mothers, bins=bins2, alpha=0.6, label=f'Mothers (n={len(hrs_mothers):,})', color='#e74c3c')
ax2.hist(hrs_childless, bins=bins2, alpha=0.6, label=f'Childless (n={len(hrs_childless):,})', color='#3498db')
ax2.axvline(hrs_mothers.median(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mothers median: ${hrs_mothers.median():,.0f}')
ax2.axvline(hrs_childless.median(), color='#3498db', linestyle='--', linewidth=2, label=f'Childless median: ${hrs_childless.median():,.0f}')
ax2.set_xlabel('Annual Household Income ($)')
ax2.set_ylabel('Count')
ax2.set_title('HRS (Late Career)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xlim(0, 200000)

plt.suptitle('Income Distribution by Motherhood Status', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / 'income_distribution_motherhood.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: income_distribution_motherhood.png")

# ============================================================================
# FIGURE 4: Sample Composition
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# NLSY composition
nlsy_female = nlsy[nlsy['female'] == 1]
mothers_pct = nlsy_female['has_children'].mean() * 100

ax1 = axes[0]
ax1.pie([mothers_pct, 100-mothers_pct],
        labels=['Mothers', 'Childless'],
        autopct='%1.1f%%',
        colors=['#e74c3c', '#3498db'],
        explode=[0.02, 0],
        startangle=90)
ax1.set_title(f'NLSY79 (n={len(nlsy_female):,})', fontweight='bold')

# HRS composition
hrs_female = hrs[hrs['female'] == 1]
mothers_pct_hrs = hrs_female['has_children'].mean() * 100

ax2 = axes[1]
ax2.pie([mothers_pct_hrs, 100-mothers_pct_hrs],
        labels=['Mothers', 'Childless'],
        autopct='%1.1f%%',
        colors=['#e74c3c', '#3498db'],
        explode=[0.02, 0],
        startangle=90)
ax2.set_title(f'HRS (n={len(hrs_female):,})', fontweight='bold')

plt.suptitle('Sample Composition by Motherhood Status (Women Only)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / 'sample_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: sample_composition.png")

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"""
                                    NLSY79          HRS
                                (Early/Mid)    (Late Career)
----------------------------------------------------------------
Sample Size (Women)                {len(nlsy[nlsy['female']==1]):,}          {len(hrs[hrs['female']==1]):,}
  - Mothers                        {nlsy_female['has_children'].sum():,}          {hrs_female['has_children'].sum():,}
  - Childless                      {(nlsy_female['has_children']==0).sum():,}            {(hrs_female['has_children']==0).sum():,}

Mean Income:
  - Mothers                    ${nlsy_mothers.mean():>10,.0f}    ${hrs_mothers.mean():>10,.0f}
  - Childless                  ${nlsy_childless.mean():>10,.0f}    ${hrs_childless.mean():>10,.0f}

Motherhood Penalty:
  - Absolute                   ${nlsy_childless.mean()-nlsy_mothers.mean():>10,.0f}    ${hrs_childless.mean()-hrs_mothers.mean():>10,.0f}
  - Relative                         {(nlsy_childless.mean()-nlsy_mothers.mean())/nlsy_childless.mean()*100:>6.1f}%         {(hrs_childless.mean()-hrs_mothers.mean())/hrs_childless.mean()*100:>6.1f}%
""")

print("\nAll figures saved to:", BASE_DIR)

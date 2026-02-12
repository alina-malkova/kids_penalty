#!/usr/bin/env python3
"""
Create Lifecycle Motherhood Penalty Figure
Shows the data gap at ages 35-50
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("/Users/amalkova/Downloads/Kids Penalty (2024)")
OUTPUT_DIR = BASE_DIR / "harmonized_data"

# Load the 5-year bin results
results = pd.read_csv(OUTPUT_DIR / "lifecycle_penalty_5yr_bins.csv")

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Define colors by data quality
colors = []
for _, row in results.iterrows():
    if row['source'] == 'insufficient':
        colors.append('lightgray')
    elif row['source'] == 'NLSY79':
        colors.append('#2ecc71')  # Green for individual income
    else:
        colors.append('#e74c3c')  # Red for household income (HRS)

# Extract age midpoints
age_mids = []
for ag in results['age_group']:
    parts = ag.split('-')
    age_mids.append((int(parts[0]) + int(parts[1])) / 2)

results['age_mid'] = age_mids

# Plot bars
valid_results = results[results['source'] != 'insufficient']
invalid_results = results[results['source'] == 'insufficient']

# Plot valid data
for source in ['NLSY79', 'HRS']:
    subset = valid_results[valid_results['source'] == source]
    color = '#2ecc71' if source == 'NLSY79' else '#e74c3c'
    label = f'{source} (Individual Income)' if source == 'NLSY79' else f'{source} (Household Income)'
    bars = ax.bar(subset['age_mid'], subset['penalty_pct'], width=4,
                  color=color, alpha=0.7, label=label, edgecolor='black', linewidth=0.5)

# Plot gap region
for _, row in invalid_results.iterrows():
    ax.bar(row['age_mid'], 0, width=4, color='lightgray', alpha=0.5,
           hatch='///', edgecolor='gray')
    ax.annotate('DATA\nGAP', xy=(row['age_mid'], 0), ha='center', va='center',
                fontsize=8, color='gray', fontweight='bold')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add region labels
ax.axvspan(17.5, 37.5, alpha=0.1, color='green', label='NLSY79 coverage')
ax.axvspan(32.5, 52.5, alpha=0.1, color='red', label='Gap/Limited HRS')
ax.axvspan(47.5, 67.5, alpha=0.1, color='orange')

# Annotations
ax.annotate('Positive penalty:\nMothers earn less',
            xy=(22.5, 8), ha='center', fontsize=9, style='italic')
ax.annotate('CAUTION:\nHousehold income\nincludes spouse',
            xy=(57.5, -30), ha='center', fontsize=9, style='italic', color='red')
ax.annotate('Need CPS\ndata here',
            xy=(40, -10), ha='center', fontsize=10, fontweight='bold', color='gray',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

# Labels and title
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Motherhood Penalty (%)\n(positive = mothers earn less)', fontsize=11)
ax.set_title('Lifecycle Motherhood Penalty: 1957-1964 Birth Cohort\nShowing Data Gap at Ages 35-50',
             fontsize=14, fontweight='bold')

ax.set_xlim(17, 67)
ax.set_ylim(-40, 15)
ax.set_xticks([22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5])
ax.set_xticklabels(['20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65'])

# Legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='#2ecc71', alpha=0.7, label='NLSY79 (Individual Income)'),
    plt.Rectangle((0,0),1,1, facecolor='#e74c3c', alpha=0.7, label='HRS (Household Income)'),
    plt.Rectangle((0,0),1,1, facecolor='lightgray', alpha=0.5, hatch='///', label='Insufficient Data (Gap)')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(BASE_DIR / 'lifecycle_penalty_with_gap.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'lifecycle_penalty_with_gap.png', dpi=150, bbox_inches='tight')
print("Saved: lifecycle_penalty_with_gap.png")

# Also create a summary table
print("\n" + "="*60)
print("LIFECYCLE MOTHERHOOD PENALTY SUMMARY")
print("="*60)
print(f"\n{'Age Group':<12} {'Penalty':>10} {'Source':>10} {'N Childless':>12}")
print("-"*50)
for _, row in results.iterrows():
    if row['source'] == 'insufficient':
        print(f"{row['age_group']:<12} {'GAP':>10} {row['source']:>10} {int(row['n_childless']):>12}")
    else:
        print(f"{row['age_group']:<12} {row['penalty_pct']:>+9.1f}% {row['source']:>10} {int(row['n_childless']):>12}")

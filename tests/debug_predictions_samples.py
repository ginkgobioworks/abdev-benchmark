"""Debug sample ordering in predictions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent

baseline_cv_pred = Path(__file__).parent / "baseline_results" / "predictions" / "GDPa1_cross_validation" / "TAP" / "TAP - linear regression.csv"
new_cv_pred = project_root / "predictions" / "GDPa1_cross_validation" / "tap_linear" / "predictions.csv"

df_baseline = pd.read_csv(baseline_cv_pred)
df_new = pd.read_csv(new_cv_pred)

print("Baseline shape:", df_baseline.shape)
print("New shape:", df_new.shape)

# Check row-by-row ordering
print("\nFirst 10 samples in baseline:")
print(df_baseline[['antibody_name', 'HIC']].head(10))

print("\nFirst 10 samples in new predictions:")
print(df_new[['antibody_name', 'HIC']].head(10))

# Check if same samples in same order
print("\nSamples in same order?", (df_baseline['antibody_name'] == df_new['antibody_name']).all())

# Check if both have same samples but different order
baseline_set = set(df_baseline['antibody_name'])
new_set = set(df_new['antibody_name'])

print(f"\nSamples in baseline: {len(baseline_set)}")
print(f"Samples in new: {len(new_set)}")
print(f"Missing in new: {baseline_set - new_set}")
print(f"Extra in new: {new_set - baseline_set}")

# Check for NaN values in predictions
print(f"\nNaN values in baseline HIC: {df_baseline['HIC'].isna().sum()}")
print(f"NaN values in new HIC: {df_new['HIC'].isna().sum()}")

# Compare sample 0 specifically
print(f"\nBaseline sample 0: {df_baseline.iloc[0]['antibody_name']} -> HIC = {df_baseline.iloc[0]['HIC']}")
print(f"New sample 0: {df_new.iloc[0]['antibody_name']} -> HIC = {df_new.iloc[0]['HIC']}")

# Find abagovomab in both
baseline_aba_idx = (df_baseline['antibody_name'] == 'abagovomab').idxmax()
new_aba_idx = (df_new['antibody_name'] == 'abagovomab').idxmax()

print(f"\nabagovomab in baseline (row {baseline_aba_idx}): HIC = {df_baseline.iloc[baseline_aba_idx]['HIC']}")
print(f"abagovomab in new (row {new_aba_idx}): HIC = {df_new.iloc[new_aba_idx]['HIC']}")

# Actually re-index both to antibody_name and compare
print("\n" + "="*60)
print("Re-indexed comparison (after sorting by antibody_name)")
print("="*60)

df_baseline_sorted = df_baseline.set_index('antibody_name').sort_index()
df_new_sorted = df_new.set_index('antibody_name').sort_index()

common_samples = set(df_baseline_sorted.index) & set(df_new_sorted.index)
print(f"\nCommon samples: {len(common_samples)}")

# Compare HIC values for common samples
diffs = []
for sample in sorted(common_samples):
    baseline_hic = df_baseline_sorted.loc[sample, 'HIC']
    new_hic = df_new_sorted.loc[sample, 'HIC']
    diff = abs(baseline_hic - new_hic)
    if diff > 1e-10:
        diffs.append((sample, baseline_hic, new_hic, diff))

if diffs:
    print(f"\n❌ Found {len(diffs)} samples with different HIC values:")
    for sample, baseline_val, new_val, diff in diffs[:10]:
        print(f"  {sample}: {baseline_val} vs {new_val} (diff: {diff:.2e})")
else:
    print("\n✓ All HIC values match after re-indexing!")

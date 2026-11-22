"""PA-RL Method A: leakage-aware preprocessing and five-class relative stress-index labels.

Input : agriculture_dataset(1).csv (or pass another CSV path as the first argument)
Output: PA_RL_MethodA_preprocessed.csv, Table4_MethodA.csv,
        MethodA_label_comparison.csv

Important: labels are derived research labels, not publisher-supplied or field-validated
physiological severity labels. The 0.5/0.3/0.2 weights are exploratory and require
sensitivity analysis / agronomist validation before strong agronomic interpretation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agriculture_dataset(1).csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# ---------- 1. Load and basic quality control ----------
df = pd.read_csv(INPUT)
df = df.drop_duplicates().reset_index(drop=True)

# Invalid physical values are marked missing, not silently clipped.
range_checks = {
    "NDVI": (-1, 1),
    "SAVI": (-1, 1),
    "Soil_Moisture": (0, 100),
    "Humidity": (0, 100),
    "Soil_pH": (0, 14),
}
for col, (lo, hi) in range_checks.items():
    if col in df.columns:
        df.loc[~df[col].between(lo, hi), col] = np.nan
for col in ["Chlorophyll_Content", "Rainfall"]:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

required = ["Soil_Moisture", "Chlorophyll_Content", "Soil_pH"]
missing_required = df[required].isna().any(axis=1)
if missing_required.any():
    # The selected index cannot be calculated without its three components.
    df = df.loc[~missing_required].copy().reset_index(drop=True)

# ---------- 2. Split before fitting imputers/scalers/thresholds ----------
idx = np.arange(len(df))
train_idx, hold_idx = train_test_split(idx, test_size=0.30, random_state=SEED)
val_idx, test_idx = train_test_split(hold_idx, test_size=0.50, random_state=SEED)
split = np.full(len(df), "train", dtype=object)
split[val_idx] = "validation"
split[test_idx] = "test"
df["Data_Split"] = split

# Training-only median imputation for model-ready columns. The three index components
# are complete after the required-field check; NDVI has 108 invalid observations in this CSV.
model_numeric = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in ["Crop_Health_Label", "Crop_Stress_Indicator"]]
for col in model_numeric:
    med = df.loc[train_idx, col].median()
    df[col] = df[col].fillna(med)

# ---------- 3. Corrected Method A index ----------
# M: lower soil moisture -> higher relative moisture-stress score.
# C: lower chlorophyll -> higher relative chlorophyll-stress score.
# P: larger absolute pH deviation from the TRAIN median -> higher relative pH score.
# Use training 5th/95th percentiles for robust scaling, then freeze for val/test.
tr = df.iloc[train_idx]
q05_m, q95_m = tr["Soil_Moisture"].quantile([0.05, 0.95])
q05_c, q95_c = tr["Chlorophyll_Content"].quantile([0.05, 0.95])
pH_ref = tr["Soil_pH"].median()
pH_dev_train = (tr["Soil_pH"] - pH_ref).abs()
pH_dev_95 = pH_dev_train.quantile(0.95)

if q95_m <= q05_m or q95_c <= q05_c or pH_dev_95 <= 0:
    raise ValueError("A training-set scaling range is zero; inspect the selected components.")

M = ((q95_m - df["Soil_Moisture"]) / (q95_m - q05_m)).clip(0, 1)
C = ((q95_c - df["Chlorophyll_Content"]) / (q95_c - q05_c)).clip(0, 1)
P = ((df["Soil_pH"] - pH_ref).abs() / pH_dev_95).clip(0, 1)

# Weights retained from the user's original Method A for a controlled comparison.
df["MethodA_Moisture_Stress"] = M
df["MethodA_Chlorophyll_Stress"] = C
df["MethodA_pH_Stress"] = P
df["MethodA_CSI"] = 0.50 * M + 0.30 * C + 0.20 * P

# Five ordinal classes: cut-points are the training-set CSI quintiles.
cut_points = df.loc[train_idx, "MethodA_CSI"].quantile([0.20, 0.40, 0.60, 0.80]).to_numpy()
if len(np.unique(cut_points)) != 4:
    raise ValueError("Non-unique quintile cut-points; use rank-based quantile binning and report ties.")
df["MethodA_Stress_Level"] = np.digitize(df["MethodA_CSI"].to_numpy(), cut_points, right=True).astype(int)

# ---------- 4. Legacy Method A, reproduced for comparison only ----------
legacy_m = ((50 - df["Soil_Moisture"]) / 50).clip(0, 1)
legacy_c = ((1 - df["Chlorophyll_Content"]) / 1).clip(0, 1)
legacy_p = ((7 - df["Soil_pH"]) / 7).clip(0, 1)  # one-sided pH penalty in original code
legacy_csi = 0.50 * legacy_m + 0.30 * legacy_c + 0.20 * legacy_p
legacy_bins = [0, 0.25, 0.50, 0.70, 0.85, 1.0000001]
df["Legacy_MethodA_CSI"] = legacy_csi
df["Legacy_MethodA_Level"] = pd.cut(
    legacy_csi, bins=legacy_bins, labels=False, include_lowest=True
).astype("Int64")

# ---------- 5. Table 4 counts ----------
class_names = {
    0: "Very low relative stress-index score",
    1: "Low relative stress-index score",
    2: "Moderate relative stress-index score",
    3: "High relative stress-index score",
    4: "Very high relative stress-index score",
}
criteria = {
    0: f"CSI <= {cut_points[0]:.5f}",
    1: f"{cut_points[0]:.5f} < CSI <= {cut_points[1]:.5f}",
    2: f"{cut_points[1]:.5f} < CSI <= {cut_points[2]:.5f}",
    3: f"{cut_points[2]:.5f} < CSI <= {cut_points[3]:.5f}",
    4: f"CSI > {cut_points[3]:.5f}",
}
rows = []
for level in range(5):
    mask = df["MethodA_Stress_Level"].eq(level)
    rows.append({
        "Level": level,
        "Description": class_names[level],
        "Labelling_criterion": criteria[level],
        "Total_n": int(mask.sum()),
        "Total_percent": round(mask.mean() * 100, 2),
        "Train_n": int((mask & df["Data_Split"].eq("train")).sum()),
        "Validation_n": int((mask & df["Data_Split"].eq("validation")).sum()),
        "Test_n": int((mask & df["Data_Split"].eq("test")).sum()),
    })
table4 = pd.DataFrame(rows)
total_row = pd.DataFrame([{
    "Level": "Total", "Description": "All records", "Labelling_criterion": "—",
    "Total_n": len(df), "Total_percent": 100.00,
    "Train_n": int((df.Data_Split == "train").sum()),
    "Validation_n": int((df.Data_Split == "validation").sum()),
    "Test_n": int((df.Data_Split == "test").sum()),
}])
table4 = pd.concat([table4, total_row], ignore_index=True)

# Compare original fixed bins with revised train-fitted quintile labels.
comp_rows = []
for scheme, col in [("Legacy Method A (original fixed scaling/bins)", "Legacy_MethodA_Level"),
                    ("Corrected Method A (training-fitted robust scaling/quintiles)", "MethodA_Stress_Level")]:
    for level in range(5):
        n = int(df[col].eq(level).sum())
        comp_rows.append({"Labelling_scheme": scheme, "Level": level, "n": n,
                         "Percent": round(100*n/len(df), 2)})
comparison = pd.DataFrame(comp_rows)

# Summary of the original publisher-provided fields (not used as Method A target).
print("Records after exact-duplicate removal and required-component QC:", len(df))
print("Training/validation/test:", (df.Data_Split == "train").sum(),
      (df.Data_Split == "validation").sum(), (df.Data_Split == "test").sum())
print("Training-fitted Method A parameters:", {
    "moisture_q05": q05_m, "moisture_q95": q95_m,
    "chlorophyll_q05": q05_c, "chlorophyll_q95": q95_c,
    "pH_reference_training_median": pH_ref, "pH_abs_deviation_q95": pH_dev_95,
    "CSI_quintile_cut_points": cut_points.tolist()})
print("\nTABLE 4\n", table4.to_string(index=False))
print("\nLABEL COMPARISON\n", comparison.to_string(index=False))
if "Crop_Health_Label" in df:
    print("\nOriginal Crop_Health_Label counts:", df["Crop_Health_Label"].value_counts().sort_index().to_dict())

# Outputs
df.to_csv(OUT / "PA_RL_MethodA_preprocessed.csv", index=False)
table4.to_csv(OUT / "Table4_MethodA.csv", index=False)
comparison.to_csv(OUT / "MethodA_label_comparison.csv", index=False)
print("\nSaved outputs to:", OUT.resolve())

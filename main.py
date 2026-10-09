import os
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD SYNTHETIC CKD DATASET
# ============================================================

folder_path = r"CKD_Transitional_Care_Data"
file_path = os.path.join(folder_path, "CKD_270_Patient_Dataset.csv")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

print("\n" + "=" * 75)
print("CKD DATA PREPROCESSING AND INCONSISTENCY VERIFICATION")
print("=" * 75)
print("Original dataset shape:", df.shape)

# ============================================================
# 2. STANDARDIZE DATA TYPES
# ============================================================

numeric_columns = [
    "Cohort", "Age_years", "Sex", "Creatinine_D0", "eGFR_D0",
    "BUN_D0", "Creatinine_FU", "eGFR_FU", "BUN_FU",
    "Followup_days", "Rehospitalization_180d",
    "Number_rehospitalizations", "Days_to_first_rehospitalization",
    "Death_180d", "Lost_to_followup"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df[col] = df[col].replace("", np.nan)

# Remove exact duplicate rows only
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("\n[1] BASIC DATA CHECK")
print("Exact duplicate rows removed:", duplicates)
print("Records remaining:", len(df))
print("Duplicate patient IDs:", int(df["Patient_ID"].duplicated().sum()))

# ============================================================
# 3. VERIFY MISSING VALUES
# ============================================================

print("\n[2] MISSING VALUES BEFORE CORRECTION")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

# ============================================================
# 4. VERIFY REHOSPITALIZATION FLAG, COUNT AND TIMING
# ============================================================

print("\n[3] REHOSPITALIZATION CONSISTENCY CHECK")

readmission_mismatch = (
    (
        df["Rehospitalization_180d"].eq(0)
        & df["Number_rehospitalizations"].gt(0)
    )
    |
    (
        df["Rehospitalization_180d"].eq(1)
        & df["Number_rehospitalizations"].lt(1)
    )
)

print("Readmission flag/count inconsistencies:", int(readmission_mismatch.sum()))

# A readmission requires a valid first-readmission time.
missing_time_for_readmission = (
    df["Rehospitalization_180d"].eq(1)
    & df["Days_to_first_rehospitalization"].isna()
)

invalid_time_for_readmission = (
    df["Rehospitalization_180d"].eq(1)
    & df["Days_to_first_rehospitalization"].notna()
    & ~df["Days_to_first_rehospitalization"].between(1, 180)
)

unexpected_time_without_readmission = (
    df["Rehospitalization_180d"].eq(0)
    & df["Days_to_first_rehospitalization"].notna()
    & df["Days_to_first_rehospitalization"].ne(0)
)

print("Readmission cases with missing event time:", int(missing_time_for_readmission.sum()))
print("Readmission cases with time outside days 1-180:", int(invalid_time_for_readmission.sum()))
print("No-readmission cases with unexpected nonzero event time:", int(unexpected_time_without_readmission.sum()))

# ============================================================
# 5. FILL ZERO FOR NO READMISSION
# ============================================================

no_readmission = df["Rehospitalization_180d"].eq(0)

# Fill zero only when no readmission occurred.
df.loc[no_readmission, "Days_to_first_rehospitalization"] = 0

print("\n[4] NO-READMISSION TIME CORRECTION")
print("No-readmission records assigned time = 0:", int(no_readmission.sum()))
print(
    "Remaining missing readmission times:",
    int(df["Days_to_first_rehospitalization"].isna().sum())
)

# Do not fabricate event times for readmission cases.
# Keep those cases missing if their actual time is unavailable.

# ============================================================
# 6. VERIFY BASELINE CKD STAGE AGAINST BASELINE eGFR
# ============================================================

print("\n[5] BASELINE CKD STAGE/eGFR CHECK")

stage_ranges = {
    "G3a": (45, 60),
    "G3b": (30, 45),
    "G4": (15, 30),
    "G5": (0, 15)
}

df["CKD_stage"] = df["CKD_stage"].replace({
    "g3a": "G3a", "g3b": "G3b", "g4": "G4", "g5": "G5"
})

baseline_mismatch = pd.Series(False, index=df.index)

for stage, (lower, upper) in stage_ranges.items():
    rows = df["CKD_stage"].eq(stage) & df["eGFR_D0"].notna()
    baseline_mismatch |= rows & (
        (df["eGFR_D0"] < lower) | (df["eGFR_D0"] >= upper)
    )

print("Baseline CKD stage/eGFR inconsistencies:", int(baseline_mismatch.sum()))

if baseline_mismatch.any():
    print(df.loc[baseline_mismatch, ["Patient_ID", "CKD_stage", "eGFR_D0"]].to_string(index=False))

# ============================================================
# 7. VERIFY FOLLOW-UP eGFR DIFFERENCES
# ============================================================

print("\n[6] FOLLOW-UP CKD STAGE/eGFR REVIEW")

followup_mismatch = pd.Series(False, index=df.index)

for stage, (lower, upper) in stage_ranges.items():
    rows = df["CKD_stage"].eq(stage) & df["eGFR_FU"].notna()
    followup_mismatch |= rows & (
        (df["eGFR_FU"] < lower) | (df["eGFR_FU"] >= upper)
    )

print("Follow-up stage/eGFR differences:", int(followup_mismatch.sum()))

if followup_mismatch.any():
    print("\nRecords for review:")
    print(
        df.loc[
            followup_mismatch,
            ["Patient_ID", "CKD_stage", "eGFR_D0", "eGFR_FU"]
        ].to_string(index=False)
    )

print(
    "\nInterpretation: These are potential differences from the baseline stage, "
    "not automatic data errors."
)

# ============================================================
# 8. VERIFY CHANGE VARIABLES
# ============================================================

print("\n[7] CHANGE VARIABLE VERIFICATION")

change_pairs = [
    ("Creatinine_D0", "Creatinine_FU", "Creatinine_change"),
    ("eGFR_D0", "eGFR_FU", "eGFR_change"),
    ("BUN_D0", "BUN_FU", "BUN_change")
]

for baseline, followup, change in change_pairs:
    expected = df[followup] - df[baseline]

    if change in df.columns:
        valid = expected.notna() & df[change].notna()
        mismatch = valid & ((df[change] - expected).abs() > 1e-6)
        print(f"{change}: inconsistent values before recalculation = {int(mismatch.sum())}")
    else:
        print(f"{change}: column missing; it will be created.")

    # Recalculate using follow-up minus baseline.
    df[change] = expected

print("Change variables recalculated.")

# ============================================================
# 9. VERIFY NUMERICAL RANGES AND BINARY CODING
# ============================================================

print("\n[8] RANGE AND CODING CHECK")

range_rules = {
    "Age_years": (18, 120),
    "Creatinine_D0": (0, None),
    "Creatinine_FU": (0, None),
    "eGFR_D0": (0, 200),
    "eGFR_FU": (0, 200),
    "BUN_D0": (0, None),
    "BUN_FU": (0, None),
    "Followup_days": (0, 180),
    "Number_rehospitalizations": (0, None)
}

for col, (lower, upper) in range_rules.items():
    invalid = df[col].notna() & (df[col] < lower)

    if upper is not None:
        invalid |= df[col].notna() & (df[col] > upper)

    print(f"{col}: out-of-range values = {int(invalid.sum())}")

for col in ["Cohort", "Rehospitalization_180d", "Death_180d", "Lost_to_followup"]:
    invalid = df[col].notna() & ~df[col].isin([0, 1])
    print(f"{col}: invalid binary values = {int(invalid.sum())}")

invalid_sex = df["Sex"].notna() & ~df["Sex"].isin([1, 2])
print("Sex: invalid codes =", int(invalid_sex.sum()))

# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 75)

print("Final dataset shape:", df.shape)
print("Total records:", len(df))
print("No-readmission records:", int(df["Rehospitalization_180d"].eq(0).sum()))
print("Readmission records:", int(df["Rehospitalization_180d"].eq(1).sum()))
print(
    "No-readmission records with time = 0:",
    int((df["Rehospitalization_180d"].eq(0) & df["Days_to_first_rehospitalization"].eq(0)).sum())
)
print(
    "Readmission records still missing actual event time:",
    int((df["Rehospitalization_180d"].eq(1) & df["Days_to_first_rehospitalization"].isna()).sum())
)
print("Baseline stage/eGFR mismatches:", int(baseline_mismatch.sum()))
print("Follow-up stage/eGFR differences for review:", int(followup_mismatch.sum()))

print("\nCohort distribution:")
print(df["Cohort"].value_counts().sort_index())

print("\nMissing values remaining:")
missing = df.isna().sum()
print(missing[missing > 0].to_string() if missing.any() else "None")

print("\nFirst five preprocessed records:")
print(df.head().to_string(index=False))

print("\nNo files were saved or overwritten.")
print("Note: This is synthetic data; preprocessing does not establish clinical validity.")
print("=" * 75)

# Final in-memory preprocessed dataset:
preprocessed_df = df.copy()
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
REPORT_PATH = ROOT / "reports"


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)


# =========================================================
# LOAD FEATURE NAMES
# =========================================================

with open(META_PATH, "r") as f:
    import json

    metadata = json.load(f)

features = metadata["features"]


# =========================================================
# EXTRACT FEATURE IMPORTANCE
# =========================================================

print("Detecting model structure...")

estimator = model

# Handle sklearn Pipeline
if hasattr(model, "steps"):
    print("Pipeline detected.")

    estimator = model.steps[-1][1]

# Handle GridSearchCV / RandomizedSearchCV
elif hasattr(model, "best_estimator_"):
    print("SearchCV model detected.")

    estimator = model.best_estimator_

    if hasattr(estimator, "steps"):
        estimator = estimator.steps[-1][1]

# Check for feature importance
if hasattr(estimator, "feature_importances_"):

    importances = estimator.feature_importances_

else:

    print(
        "Model structure:",
        type(model)
    )

    print(
        "Final estimator:",
        type(estimator)
    )

    raise AttributeError(
        "Could not locate feature_importances_ in the saved model."
    )


# =========================================================
# CREATE DATAFRAME
# =========================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df = importance_df.reset_index(
    drop=True
)


# =========================================================
# DISPLAY TOP FEATURES
# =========================================================

print("\n")
print("=" * 60)
print("TOP 15 FEATURE IMPORTANCES")
print("=" * 60)

print(
    importance_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =========================================================
# SAVE CSV
# =========================================================

csv_path = (
    REPORT_PATH /
    "feature_importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nSaved feature importance to:\n{csv_path}"
)


# =========================================================
# CREATE PLOT
# =========================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Feature Importance — Top 15 Features"
)

plt.tight_layout()


plot_path = (
    REPORT_PATH /
    "feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=180
)

plt.close()

print(
    f"Saved feature importance plot to:\n{plot_path}"
)

print(
    "\nFeature importance analysis completed successfully! ✅"
)
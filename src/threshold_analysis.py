from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
TEST_PATH = ROOT / "reports" / "test_set.csv"
REPORT_PATH = ROOT / "reports"

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

# =========================================================
# LOAD TEST DATA
# =========================================================

print("Loading test dataset...")

test_data = pd.read_csv(TEST_PATH)

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print(f"Test samples: {len(X_test):,}")
print(f"Fraud samples: {int(y_test.sum()):,}")

# =========================================================
# GENERATE FRAUD PROBABILITIES
# =========================================================

print("\nGenerating fraud probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

results_df = pd.DataFrame(results)

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n")
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# =========================================================
# BEST F1 THRESHOLD
# =========================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n")
print("=" * 70)
print("BEST THRESHOLD BY F1-SCORE")
print("=" * 70)

print(
    f"Threshold : {best_row['threshold']:.2f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1-Score  : {best_row['f1']:.4f}"
)

# =========================================================
# SAVE CSV
# =========================================================

csv_path = REPORT_PATH / "threshold_analysis.csv"

results_df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nSaved results to:\n{csv_path}"
)

# =========================================================
# CREATE PLOT
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["threshold"],
    results_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1"],
    marker="o",
    label="F1-score"
)

plt.xlabel("Decision Threshold")

plt.ylabel("Score")

plt.title(
    "Precision, Recall and F1 vs Decision Threshold"
)

plt.xticks(thresholds)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plot_path = REPORT_PATH / "threshold_analysis.png"

plt.savefig(
    plot_path,
    dpi=180
)

plt.close()

print(
    f"Saved plot to:\n{plot_path}"
)

print("\nThreshold analysis completed successfully! ✅")
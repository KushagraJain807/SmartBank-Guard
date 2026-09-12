from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve
)
from xgboost import XGBClassifier

from data_loader import load_data
from preprocessing import build_preprocessor, TARGET

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def main():
    print("Loading dataset...")
    df = load_data()

    print(f"Rows: {len(df):,}")
    print(f"Fraud cases: {int(df[TARGET].sum()):,}")
    print(f"Fraud rate: {df[TARGET].mean()*100:.4f}%")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Estimate imbalance for XGBoost.
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42
        )
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)

        probabilities = pipe.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        metrics = {
            "model": name,
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities)
        }
        results.append(metrics)
        trained[name] = pipe

        print(pd.DataFrame([metrics]).to_string(index=False))
        print(classification_report(y_test, predictions, digits=4, zero_division=0))

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    joblib.dump(best_model, MODEL_DIR / "fraud_model.joblib")

    # Save test data for reproducible local evaluation.
    test_out = X_test.copy()
    test_out[TARGET] = y_test.values
    test_out.to_csv(REPORT_DIR / "test_set.csv", index=False)

    # Confusion matrix + ROC curve for the selected model.
    best_prob = best_model.predict_proba(X_test)[:, 1]
    best_pred = (best_prob >= 0.50).astype(int)

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, best_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, best_prob):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=180)
    plt.close()

    metadata = {
        "best_model": best_name,
        "threshold": 0.50,
        "features": list(X.columns),
        "test_rows": int(len(X_test)),
        "metrics": results_df.iloc[0].to_dict()
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print("\nBest model:", best_name)
    print("Saved:", MODEL_DIR / "fraud_model.joblib")
    print("Saved metrics:", REPORT_DIR / "model_comparison.csv")

if __name__ == "__main__":
    main()

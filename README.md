# SmartBank Guard 🏦

AI-powered transaction fraud detection and risk scoring system built with Python and machine learning.

## Why this project?
Financial institutions need systems that can identify suspicious transactions while keeping false alarms manageable. SmartBank Guard demonstrates an end-to-end ML workflow: data loading, preprocessing, class-imbalance handling, model comparison, evaluation, model persistence, and an interactive Streamlit dashboard.

> This is an educational project using a public credit-card fraud dataset. It is not an HSBC project and does not use HSBC customer data.

## Features
- Exploratory data analysis
- Stratified train/test split
- Feature scaling
- Logistic Regression baseline
- Random Forest model
- XGBoost model
- Precision, Recall, F1 and ROC-AUC evaluation
- Confusion matrix and ROC curve
- Saved production model
- Transaction-level risk scoring
- Streamlit dashboard

## Project structure
```text
SmartBank-Guard/
├── data/                  # Dataset is downloaded/generated locally; not committed
├── models/                # Trained model is saved here
├── notebooks/
│   └── fraud_detection.ipynb
├── reports/               # Evaluation outputs
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset
The training script uses the public `creditcard` dataset through OpenML. The target column is `Class` where 1 represents fraud and 0 represents a legitimate transaction.

The dataset is highly imbalanced, so accuracy alone is not an appropriate primary metric.

## Run locally
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The first training run may take some time because the dataset is downloaded and the models are trained.

## Model selection
The training script compares Logistic Regression, Random Forest and XGBoost. The final model is selected using ROC-AUC on the held-out test set, while Precision, Recall and F1 are also reported because fraud detection is a rare-event classification problem.

## Interview talking points
1. Why is accuracy misleading for fraud detection?
2. Why use stratification during train/test split?
3. What is class imbalance?
4. Why compare a linear baseline with tree-based models?
5. What does precision vs recall mean in fraud detection?
6. How would you reduce false positives in a real banking system?
7. How would you monitor model drift after deployment?
8. Why should preprocessing be fitted only on training data?
9. How would you protect transaction data in production?
10. How could this system be deployed as an API/microservice?

## Limitations
- Public/anonymized data is used.
- The project is a prototype, not a production banking fraud engine.
- Real banking systems would require stronger security, explainability, latency, monitoring, governance and human-review workflows.

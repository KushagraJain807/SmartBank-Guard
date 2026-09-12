# SmartBank Guard 🏦🛡️

### AI-Powered Transaction Fraud Detection & Risk Scoring System

SmartBank Guard is an end-to-end machine learning project that detects potentially fraudulent credit-card transactions and provides transaction-level risk assessment through an interactive Streamlit dashboard.

The project compares multiple machine learning algorithms, evaluates them using fraud-focused metrics, performs decision-threshold analysis, and provides model explainability through feature importance.

> ⚠️ **Educational Project:** SmartBank Guard uses a public anonymized credit-card fraud dataset. It is not affiliated with HSBC and does not use HSBC customer data.

---

## 🚀 Project Highlights

- 🤖 Machine learning based fraud detection
- 📊 Comparison of Logistic Regression, Random Forest and XGBoost
- ⚖️ Handling of highly imbalanced fraud data
- 🎯 Precision, Recall, F1-score and ROC-AUC evaluation
- 📈 Confusion Matrix and ROC Curve
- 🔍 Decision-threshold analysis
- 🧠 Feature importance and model explainability
- 💳 Transaction-level fraud risk scoring
- 🌐 Interactive Streamlit dashboard
- 💾 Saved trained model and analysis reports

---

## 🧠 Problem Statement

Credit-card fraud detection is a challenging classification problem because fraudulent transactions represent only a very small proportion of all transactions.

A model with high accuracy can still perform poorly at detecting fraud.

Therefore, SmartBank Guard focuses on metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC

The project also analyzes different probability thresholds to identify an operating point that provides a useful balance between detecting fraud and avoiding unnecessary false alerts.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Credit Card Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Loading & EDA  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    │ Scaling & Splitting │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Model Training            │
              │                                │
              │ Logistic Regression            │
              │ Random Forest                   │
              │ XGBoost                        │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │                     │
                    │ Precision           │
                    │ Recall              │
                    │ F1-score            │
                    │ ROC-AUC             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Threshold Analysis  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ XGBoost Model       │
                    │ Risk Prediction     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘
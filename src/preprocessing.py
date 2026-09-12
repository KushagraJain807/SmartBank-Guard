import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

TARGET = "Class"

def build_preprocessor(X: pd.DataFrame):
    numeric_features = list(X.columns)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="drop"
    )

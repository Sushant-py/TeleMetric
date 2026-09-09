"""
Fair final model selection for TeleMetric.

This script tunes Logistic Regression, Random Forest and XGBoost under the
same train/test split, the same 5-fold StratifiedKFold protocol, and the same
primary ROC-AUC objective. It then evaluates the selected estimators on the
same untouched test set.

Run from the repository root:
    python scripts/final_model_selection.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_customer_clean.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Load the already-cleaned / feature-engineered dataset.
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

target = "Churn_Target"
if target not in df.columns:
    df[target] = df["Churn"].map({"Yes": 1, "No": 0})

drop_cols = ["customerID", "Churn", target]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[target].astype(int)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ],
    remainder="drop",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

search_spaces = {
    "Logistic Regression": {
        "model__C": np.logspace(-3, 2, 20),
        "model__class_weight": [None, "balanced"],
        "model__solver": ["lbfgs"],
    },
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 5, 8, 12, 16],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    },
    "XGBoost": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}

rows = []
best_estimators = {}

for name, model in models.items():
    print(f"\nTuning {name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_spaces[name],
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_estimators[name] = search.best_estimator_
    y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    rows.append({
        "Model": name,
        "CV_ROC_AUC": search.best_score_,
        "CV_ROC_AUC_STD": float(search.cv_results_["std_test_score"][search.best_index_]),
        "Test_ROC_AUC": roc_auc_score(y_test, y_prob),
        "Test_PR_AUC": average_precision_score(y_test, y_prob),
        "Test_Accuracy": accuracy_score(y_test, y_pred),
        "Test_Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test_Recall": recall_score(y_test, y_pred, zero_division=0),
        "Test_F1": f1_score(y_test, y_pred, zero_division=0),
        "Best_Params": json.dumps(search.best_params_, default=str),
    })

results = pd.DataFrame(rows).sort_values("CV_ROC_AUC", ascending=False).reset_index(drop=True)
results.to_csv(OUTPUT_DIR / "final_model_selection.csv", index=False)

winner = results.iloc[0]
winner_name = winner["Model"]
joblib.dump(best_estimators[winner_name], OUTPUT_DIR / "final_best_model.pkl")

with open(OUTPUT_DIR / "final_model_selection.json", "w", encoding="utf-8") as f:
    json.dump({
        "selected_model": winner_name,
        "selection_metric": "5-fold CV ROC-AUC",
        "selected_cv_roc_auc": float(winner["CV_ROC_AUC"]),
        "selected_test_roc_auc": float(winner["Test_ROC_AUC"]),
        "selected_test_pr_auc": float(winner["Test_PR_AUC"]),
    }, f, indent=2)

print("\nFINAL MODEL SELECTION")
print(results[["Model", "CV_ROC_AUC", "CV_ROC_AUC_STD", "Test_ROC_AUC", "Test_PR_AUC"]].to_string(index=False))
print(f"\nSelected model: {winner_name}")
print(f"CV ROC-AUC: {winner['CV_ROC_AUC']:.4f}")
print(f"Test ROC-AUC: {winner['Test_ROC_AUC']:.4f}")
print(f"Test PR-AUC: {winner['Test_PR_AUC']:.4f}")
print("\nSaved:")
print("  outputs/final_model_selection.csv")
print("  outputs/final_model_selection.json")
print("  outputs/final_best_model.pkl")

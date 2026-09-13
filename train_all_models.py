"""
train_all_models.py
====================
Reproduces every model built in ITI_Project_V2_.ipynb (HR Analytics /
"Job Change of Data Scientists" dataset) and exports everything the
Streamlit app (app.py) needs to serve live predictions from ALL of them:

    1.  KNN                        (Tuned: SMOTE + PCA + RandomizedSearchCV + threshold tuning)
    2.  LR + SMOTE                 (Logistic Regression on plain encoded features)
    3.  FE LR + SMOTE              (Logistic Regression on feature-engineered features)
    4.  Baseline Random Forest     (feature-engineered features, class_weight="balanced")
    5.  Tuned Random Forest        (RandomizedSearchCV over RF hyperparameters)
    6.  XGBoost + scale_pos_weight (Approach A)
    7.  XGBoost + SMOTE            (Approach B)

USAGE
-----
1. Put `aug_train.csv` (the original Kaggle "HR Analytics: Job Change of
   Data Scientists" training file) in the same folder as this script.
2. Run:  python train_all_models.py
3. This produces an `artifacts/` folder containing:
     - encoder.pkl                 (shared OneHotEncoder)
     - <model_key>.pkl             (one file per trained model)
     - metadata.json               (feature lists, feature order per
                                     feature-family, per-model threshold,
                                     per-model test metrics)
4. Copy (or point) the `artifacts/` folder next to `app.py` and run:
     streamlit run app.py

Notes
-----
- This intentionally mirrors the notebook's logic 1:1 (same cleaning,
  same single shared encoder fit once, same SMOTE usage, same
  hyperparameter search spaces) so the exported models behave exactly
  like the ones in the notebook.
- Hyperparameter search (RF / XGBoost / KNN) can take several minutes.
  Pass --fast to shrink search budgets for a quicker local smoke-test.
"""

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sanitize_columns(cols):
    """XGBoost rejects <, >, [, ], (, ), :, , in feature names."""
    clean = []
    for c in cols:
        c = c.replace("<", "lt").replace(">", "gt")
        c = re.sub(r"[\[\]{}():,]", "_", c)
        clean.append(c)
    return clean


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Step 1-3: Load + clean ----------------
    log("Loading data...")
    df = pd.read_csv(args.data)
    log(f"Shape: {df.shape}")

    df_clean = df.copy()
    mode_columns = ["enrolled_university", "education_level", "experience", "last_new_job"]
    for col in mode_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    unknown_columns = ["gender", "major_discipline", "company_size", "company_type"]
    for col in unknown_columns:
        df_clean[col] = df_clean[col].fillna("Unknown")

    # ---------------- Step 4-5: Split + column typing ----------------
    X = df_clean.drop(columns=["target", "enrollee_id"])
    y = df_clean["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()
    log(f"Categorical columns: {categorical_columns}")
    log(f"Numerical columns:   {numerical_columns}")

    # ---------------- Step 6: Shared encoder (fit ONCE) ----------------
    log("Fitting shared OneHotEncoder...")
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    encoder.fit(X_train[categorical_columns])

    X_train_cat = pd.DataFrame(
        encoder.transform(X_train[categorical_columns]),
        index=X_train.index, columns=encoder.get_feature_names_out(categorical_columns),
    )
    X_test_cat = pd.DataFrame(
        encoder.transform(X_test[categorical_columns]),
        index=X_test.index, columns=encoder.get_feature_names_out(categorical_columns),
    )

    X_train_encoded = pd.concat([X_train_cat, X_train[numerical_columns]], axis=1)
    X_test_encoded = pd.concat([X_test_cat, X_test[numerical_columns]], axis=1)
    X_train_encoded.columns = X_train_encoded.columns.astype(str)
    X_test_encoded.columns = X_test_encoded.columns.astype(str)

    # ---------------- Step 6.1: SMOTE on plain features ----------------
    log("Applying SMOTE (plain features)...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_encoded, y_train)

    metrics = {}

    # ================= Model 1: LR + SMOTE (plain) =================
    log("Training LR + SMOTE...")
    model_bal = LogisticRegression(max_iter=3000)
    model_bal.fit(X_train_bal, y_train_bal)
    y_pred_bal = model_bal.predict(X_test_encoded)
    metrics["LR + SMOTE"] = {
        "Accuracy": accuracy_score(y_test, y_pred_bal),
        "Precision": precision_score(y_test, y_pred_bal),
        "Recall": recall_score(y_test, y_pred_bal),
        "F1-score": f1_score(y_test, y_pred_bal),
    }

    # ---------------- Step 8-9: Feature engineering ----------------
    log("Building feature-engineered matrices...")
    experience_num_train = X_train["experience"].replace({"<1": 0, ">20": 21, "Unknown": 0}).astype(float)
    experience_num_test = X_test["experience"].replace({"<1": 0, ">20": 21, "Unknown": 0}).astype(float)

    X_train_fe_num = X_train[numerical_columns].copy()
    X_test_fe_num = X_test[numerical_columns].copy()

    X_train_fe_num["experience_to_training_ratio"] = experience_num_train / (X_train["training_hours"] + 1)
    X_test_fe_num["experience_to_training_ratio"] = experience_num_test / (X_test["training_hours"] + 1)

    X_train_fe_num["has_relevant_degree"] = (X_train["major_discipline"] == "STEM").astype(int)
    X_test_fe_num["has_relevant_degree"] = (X_test["major_discipline"] == "STEM").astype(int)

    X_train_encoded_fe = pd.concat([X_train_cat, X_train_fe_num], axis=1)
    X_test_encoded_fe = pd.concat([X_test_cat, X_test_fe_num], axis=1)
    X_train_encoded_fe.columns = X_train_encoded_fe.columns.astype(str)
    X_test_encoded_fe.columns = X_test_encoded_fe.columns.astype(str)

    X_train_fe_bal, y_train_fe_bal = smote.fit_resample(X_train_encoded_fe, y_train)

    # ================= Model 2: FE LR + SMOTE (fe) =================
    log("Training FE LR + SMOTE...")
    model_fe_bal = LogisticRegression(max_iter=3000)
    model_fe_bal.fit(X_train_fe_bal, y_train_fe_bal)
    y_pred_fe_bal = model_fe_bal.predict(X_test_encoded_fe)
    metrics["FE LR + SMOTE"] = {
        "Accuracy": accuracy_score(y_test, y_pred_fe_bal),
        "Precision": precision_score(y_test, y_pred_fe_bal),
        "Recall": recall_score(y_test, y_pred_fe_bal),
        "F1-score": f1_score(y_test, y_pred_fe_bal),
    }

    # ================= Model 3: Baseline Random Forest (fe) =================
    log("Training Baseline Random Forest...")
    baseline_rf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    baseline_rf.fit(X_train_encoded_fe, y_train)
    baseline_pred = baseline_rf.predict(X_test_encoded_fe)
    metrics["Baseline Random Forest"] = {
        "Accuracy": accuracy_score(y_test, baseline_pred),
        "Precision": precision_score(y_test, baseline_pred, zero_division=0),
        "Recall": recall_score(y_test, baseline_pred, zero_division=0),
        "F1-score": f1_score(y_test, baseline_pred, zero_division=0),
    }

    # ================= Model 4: Tuned Random Forest (fe) =================
    log("Tuning Random Forest (RandomizedSearchCV)...")
    rf_n_iter = 6 if args.fast else 25
    param_grid = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_random = RandomizedSearchCV(
        estimator=rf, param_distributions=param_grid, n_iter=rf_n_iter, cv=3,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"},
        refit="f1", random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    rf_random.fit(X_train_encoded_fe, y_train)
    best_rf = rf_random.best_estimator_
    tuned_pred = best_rf.predict(X_test_encoded_fe)
    metrics["Tuned Random Forest"] = {
        "Accuracy": accuracy_score(y_test, tuned_pred),
        "Precision": precision_score(y_test, tuned_pred),
        "Recall": recall_score(y_test, tuned_pred),
        "F1-score": f1_score(y_test, tuned_pred),
    }

    # ================= Model 5: Tuned KNN (plain) =================
    log("Tuning KNN (SMOTE + PCA + RandomizedSearchCV)...")
    knn_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("pca", PCA(random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier(n_jobs=-1)),
    ])
    cv_knn = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    knn_param_distributions = {
        "pca": [
            PCA(n_components=0.90, random_state=RANDOM_STATE),
            PCA(n_components=0.95, random_state=RANDOM_STATE),
            PCA(n_components=0.99, random_state=RANDOM_STATE),
            "passthrough",
        ],
        "knn__n_neighbors": list(range(3, 41, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    knn_search = RandomizedSearchCV(
        knn_pipeline, param_distributions=knn_param_distributions,
        n_iter=(3 if args.fast else 5), scoring="f1", cv=cv_knn,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    knn_search.fit(X_train_encoded, y_train)
    best_knn = knn_search.best_estimator_

    y_proba_knn = best_knn.predict_proba(X_test_encoded)[:, 1]

    log("Tuning KNN decision threshold via train CV predictions...")
    y_train_proba_cv = cross_val_predict(
        best_knn, X_train_encoded, y_train, cv=cv_knn, method="predict_proba", n_jobs=-1
    )[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba_cv)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    y_pred_knn = (y_proba_knn >= best_threshold).astype(int)
    metrics["KNN"] = {
        "Accuracy": accuracy_score(y_test, y_pred_knn),
        "Precision": precision_score(y_test, y_pred_knn),
        "Recall": recall_score(y_test, y_pred_knn),
        "F1-score": f1_score(y_test, y_pred_knn),
    }

    # ---------------- Step 13: XGBoost feature prep ----------------
    log("Preparing sanitized features for XGBoost...")
    X_train_xgb = X_train_encoded.copy()
    X_test_xgb = X_test_encoded.copy()
    X_train_xgb.columns = sanitize_columns(X_train_xgb.columns)
    X_test_xgb.columns = sanitize_columns(X_test_xgb.columns)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_param_dist = {
        "n_estimators": [150, 250, 350],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.2, 0.4],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 2, 3],
    }
    xgb_n_iter = 4 if args.fast else 10
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ================= Model 6: XGBoost + scale_pos_weight (xgb) =================
    log("Tuning XGBoost Approach A (scale_pos_weight)...")
    base_model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", scale_pos_weight=scale_pos_weight,
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=1,
    )
    search = RandomizedSearchCV(
        base_model, param_distributions=xgb_param_dist, n_iter=xgb_n_iter, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )
    search.fit(X_train_xgb, y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_xgb, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )
    xgb_model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", scale_pos_weight=scale_pos_weight,
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=30, **search.best_params_,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_xgb = xgb_model.predict(X_test_xgb)
    y_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
    metrics["XGBoost + scale_pos_weight"] = {
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "Precision": precision_score(y_test, y_pred_xgb),
        "Recall": recall_score(y_test, y_pred_xgb),
        "F1-score": f1_score(y_test, y_pred_xgb),
        "ROC-AUC": roc_auc_score(y_test, y_proba_xgb),
    }

    # ================= Model 7: XGBoost + SMOTE (xgb) =================
    log("Tuning XGBoost Approach B (SMOTE)...")
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])
    xgb_param_dist_smote = {f"clf__{k}": v for k, v in xgb_param_dist.items()}
    search_smote = RandomizedSearchCV(
        pipe, param_distributions=xgb_param_dist_smote, n_iter=xgb_n_iter, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )
    search_smote.fit(X_train_xgb, y_train)
    best_params_smote = {k.replace("clf__", ""): v for k, v in search_smote.best_params_.items()}

    X_tr_b, X_val_b, y_tr_b, y_val_b = train_test_split(
        X_train_xgb, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )
    X_tr_b_bal, y_tr_b_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_b, y_tr_b)

    xgb_model_smote = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=1, early_stopping_rounds=30, **best_params_smote,
    )
    xgb_model_smote.fit(X_tr_b_bal, y_tr_b_bal, eval_set=[(X_val_b, y_val_b)], verbose=False)

    y_pred_xgb_smote = xgb_model_smote.predict(X_test_xgb)
    y_proba_xgb_smote = xgb_model_smote.predict_proba(X_test_xgb)[:, 1]
    metrics["XGBoost + SMOTE"] = {
        "Accuracy": accuracy_score(y_test, y_pred_xgb_smote),
        "Precision": precision_score(y_test, y_pred_xgb_smote),
        "Recall": recall_score(y_test, y_pred_xgb_smote),
        "F1-score": f1_score(y_test, y_pred_xgb_smote),
        "ROC-AUC": roc_auc_score(y_test, y_proba_xgb_smote),
    }

    # ---------------- Rank + pick best ----------------
    ranking = pd.DataFrame(metrics).T.sort_values(by="F1-score", ascending=False)
    best_model_name = ranking.index[0]
    log("=== Final Model Ranking (by F1-score) ===")
    log("\n" + ranking.to_string())
    log(f"Best model: {best_model_name}")

    # ---------------- Save everything ----------------
    log("Saving artifacts...")
    joblib.dump(encoder, out_dir / "encoder.pkl")

    registry = {
        "KNN": {"model": best_knn, "feature_type": "plain", "threshold": best_threshold, "file": "knn.pkl"},
        "LR + SMOTE": {"model": model_bal, "feature_type": "plain", "threshold": 0.5, "file": "lr_smote.pkl"},
        "FE LR + SMOTE": {"model": model_fe_bal, "feature_type": "fe", "threshold": 0.5, "file": "fe_lr_smote.pkl"},
        "Baseline Random Forest": {"model": baseline_rf, "feature_type": "fe", "threshold": 0.5, "file": "rf_baseline.pkl"},
        "Tuned Random Forest": {"model": best_rf, "feature_type": "fe", "threshold": 0.5, "file": "rf_tuned.pkl"},
        "XGBoost + scale_pos_weight": {"model": xgb_model, "feature_type": "xgb", "threshold": 0.5, "file": "xgb_scale_pos_weight.pkl"},
        "XGBoost + SMOTE": {"model": xgb_model_smote, "feature_type": "xgb", "threshold": 0.5, "file": "xgb_smote.pkl"},
    }

    feature_orders = {
        "plain": list(X_train_encoded.columns),
        "fe": list(X_train_encoded_fe.columns),
        "xgb": sanitize_columns(list(X_train_encoded.columns)),
    }

    models_meta = {}
    for name, info in registry.items():
        joblib.dump(info["model"], out_dir / info["file"])
        models_meta[name] = {
            "file": info["file"],
            "feature_type": info["feature_type"],
            "threshold": info["threshold"],
            "metrics": metrics[name],
        }

    # Reasonable default numeric ranges/steps for the app's sliders
    numeric_ranges = {
        "city_development_index": {
            "min": float(X["city_development_index"].min()),
            "max": float(X["city_development_index"].max()),
            "default": float(X["city_development_index"].median()),
        },
        "training_hours": {
            "min": float(X["training_hours"].min()),
            "max": float(X["training_hours"].max()),
            "default": float(X["training_hours"].median()),
        },
    }

    metadata = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "numeric_ranges": numeric_ranges,
        "feature_orders": feature_orders,
        "best_model_name": best_model_name,
        "models": models_meta,
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"Done. Artifacts saved to: {out_dir.resolve()}")
    log("Copy this folder next to app.py, then run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all notebook models and export Streamlit artifacts.")
    parser.add_argument("--data", default="aug_train.csv", help="Path to aug_train.csv")
    parser.add_argument("--out_dir", default="artifacts", help="Output folder for artifacts")
    parser.add_argument("--fast", action="store_true", help="Shrink hyperparameter search budgets for a quick smoke-test")
    args = parser.parse_args()
    main(args)

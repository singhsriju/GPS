"""
train_models.py — Train all ML models for CareerGPS Analytics
Runs on first launch or via sidebar retrain button
"""

import pandas as pd
import numpy as np
import json, os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              mean_absolute_error, mean_squared_error,
                              median_absolute_error, r2_score, confusion_matrix)


def encode_df(df, encoders=None, fit=True):
    """Encode categorical columns. Returns encoded df + encoders dict."""
    df = df.copy()

    # Ordinal maps
    income_map = {"Below ₹2L":1,"₹2–5L":2,"₹5–10L":3,"₹10–20L":4,"Above ₹20L":5,"Unknown":0}
    clarity_map = {"Not clear at all":1,"Somewhat unclear":2,"Somewhat clear":3,"Very clear":4}
    urgency_map = {"Within 3 months":5,"3–6 months":4,"6–12 months":3,"1–2 years":2,"More than 2 years":1}
    perf_map = {"Below 60%":1,"60–70%":2,"70–80%":3,"80–90%":4,"Above 90%":5}
    edu_map = {"Class 9-10":1,"Class 11-12":2,"Undergraduate":3,"Postgraduate":4,"Working Professional":5}

    df["income_enc"] = df["family_income"].map(income_map).fillna(0).astype(int)
    df["clarity_enc"] = df["career_clarity"].map(clarity_map).fillna(2).astype(int)
    df["urgency_enc"] = df["urgency"].map(urgency_map).fillna(3).astype(int)
    df["perf_enc"] = df["academic_performance"].map(perf_map).fillna(3).astype(int)
    df["edu_enc"] = df["education_level"].map(edu_map).fillna(2).astype(int)
    df["past_spend_enc"] = (df["past_career_spend"] != "Nothing").astype(int)

    # Label encode nominal
    cat_cols = ["gender","state","location_type","stream","board","decision_maker",
                "primary_info_source","career_domain_interest","preferred_feature","motivation"]

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col+"_enc"] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                df[col+"_enc"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            else:
                df[col+"_enc"] = 0

    return df, encoders


FEATURE_COLS = [
    "age","income_enc","clarity_enc","urgency_enc","perf_enc","edu_enc","past_spend_enc",
    "fear_score","risk_score",
    "gender_enc","state_enc","location_type_enc","stream_enc","board_enc",
    "decision_maker_enc","primary_info_source_enc","career_domain_interest_enc",
    "preferred_feature_enc","motivation_enc"
]


def run():
    print("Loading data...")
    df = pd.read_csv("career_survey_dataset.csv")
    df["family_income"] = df["family_income"].fillna("Unknown")

    print("Encoding features...")
    df_enc, encoders = encode_df(df, fit=True)

    X = df_enc[FEATURE_COLS].values
    y_clf = (df["platform_adoption"] == "Yes").astype(int).values
    y_reg = df["wtp_monthly"].values

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    # ── Classification
    print("Training classifier...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)
    yc_prob = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(yc_test, yc_prob)
    cm = confusion_matrix(yc_test, yc_pred)

    fi_clf = dict(zip(FEATURE_COLS, clf.feature_importances_))
    fi_clf = {k: round(float(v), 4) for k, v in sorted(fi_clf.items(), key=lambda x: x[1], reverse=True)[:12]}

    clf_metrics = {
        "accuracy": round(accuracy_score(yc_test, yc_pred), 4),
        "precision": round(precision_score(yc_test, yc_pred), 4),
        "recall": round(recall_score(yc_test, yc_pred), 4),
        "f1": round(f1_score(yc_test, yc_pred), 4),
        "roc_auc": round(roc_auc_score(yc_test, yc_prob), 4),
        "roc_curve": {"fpr": [round(x,3) for x in fpr.tolist()[::5]],
                      "tpr": [round(x,3) for x in tpr.tolist()[::5]]},
        "confusion_matrix": cm.tolist(),
        "feature_importance": fi_clf
    }

    # ── Regression
    print("Training regressor...")
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    reg.fit(X_train, yr_train)
    yr_pred = reg.predict(X_test)

    fi_reg = dict(zip(FEATURE_COLS, reg.feature_importances_))
    fi_reg = {k: round(float(v), 4) for k, v in sorted(fi_reg.items(), key=lambda x: x[1], reverse=True)[:10]}

    # sample 200 for scatter
    idx = np.random.choice(len(yr_test), min(200, len(yr_test)), replace=False)
    reg_metrics = {
        "mae": round(float(mean_absolute_error(yr_test, yr_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(yr_test, yr_pred))), 2),
        "r2": round(float(r2_score(yr_test, yr_pred)), 4),
        "median_ae": round(float(median_absolute_error(yr_test, yr_pred)), 2),
        "pred_vs_actual": {
            "actual": [round(float(x),1) for x in yr_test[idx]],
            "predicted": [round(float(x),1) for x in yr_pred[idx]]
        },
        "feature_importance": fi_reg
    }

    # ── K-Means
    print("Training clustering...")
    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append({"k": k, "inertia": float(km.inertia_)})
    best_km = KMeans(n_clusters=5, random_state=42, n_init=10)
    best_km.fit(X)

    # ── Save
    print("Saving models...")
    joblib.dump(clf, "model_classifier.pkl")
    joblib.dump(reg, "model_regressor.pkl")
    joblib.dump(best_km, "model_kmeans.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(FEATURE_COLS, "feature_names.pkl")

    all_metrics = {
        "classifier": clf_metrics,
        "regressor": reg_metrics,
        "clustering": {"inertias": inertias, "best_k": 5}
    }
    with open("model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("✅ All models trained and saved.")
    print(f"  Classifier accuracy: {clf_metrics['accuracy']*100:.1f}%")
    print(f"  Regressor R²: {reg_metrics['r2']:.3f}")
    return all_metrics


if __name__ == "__main__":
    run()

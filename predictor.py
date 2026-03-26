"""
predictor.py — Score new survey respondents
"""

import pandas as pd
import numpy as np
import joblib
import os


def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe of new respondents and returns scored results.
    """
    if not os.path.exists("model_classifier.pkl"):
        raise FileNotFoundError("Models not trained. Please train first.")

    clf = joblib.load("model_classifier.pkl")
    reg = joblib.load("model_regressor.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_names.pkl")

    # Import encode function
    import train_models
    df_enc, _ = train_models.encode_df(df.copy(), encoders=encoders, fit=False)
    df_enc["family_income"] = df_enc.get("family_income", pd.Series(["Unknown"]*len(df))).fillna("Unknown")

    # Fill missing encoded cols
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0

    X = df_enc[feature_cols].values

    adopt_prob = clf.predict_proba(X)[:, 1]
    wtp_pred = reg.predict(X)

    # Persona inference
    clarity_map = {"Not clear at all": "Confused Drifter", "Somewhat unclear": "Anxious Achiever",
                   "Somewhat clear": "Career Switcher", "Very clear": "Focused Climber"}
    persona = df["career_clarity"].map(clarity_map).fillna("Passive Explorer")

    # Priority tier
    def priority(p):
        if p >= 0.6: return "Hot"
        elif p >= 0.35: return "Warm"
        else: return "Cold"

    def action(tier, persona_label):
        if tier == "Hot":
            return "Activate Premium trial. WhatsApp parent outreach. Assign mentor."
        elif tier == "Warm":
            return "Free psychometric test CTA. 7-day email nurture sequence."
        else:
            return "Add to freemium pool. Content marketing. B2B school channel."

    results = df.copy()
    results["persona_label"] = persona
    results["adoption_probability"] = np.round(adopt_prob, 3)
    results["predicted_wtp"] = np.round(wtp_pred, 0).astype(int)
    results["priority_tier"] = [priority(p) for p in adopt_prob]
    results["recommended_action"] = [action(priority(p), pn) for p, pn in zip(adopt_prob, persona)]

    return results[["persona_label", "adoption_probability", "predicted_wtp",
                     "priority_tier", "recommended_action"] +
                    [c for c in df.columns if c in results.columns]]

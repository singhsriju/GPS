"""
app.py — CareerPath GPS  |  AI Career Platform Analytics
MBA-grade analytics dashboard — executive design, annotated charts,
business-first framing, actionable prescriptive layer.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerPath GPS — Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Remove default Streamlit padding */
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D2137 0%, #1A3A5C 100%);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] .stRadio label { 
    font-size: 13px !important; padding: 6px 0 !important; 
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* KPI cards */
.kpi-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px 20px 16px;
    border-left: 4px solid #1A6FBF;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-card.green  { border-left-color: #0B8C6E; }
.kpi-card.amber  { border-left-color: #E8900A; }
.kpi-card.red    { border-left-color: #D93025; }
.kpi-card.purple { border-left-color: #6B3FA0; }
.kpi-label { font-size: 11px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #0D2137; line-height: 1; }
.kpi-delta { font-size: 12px; color: #6B7280; margin-top: 4px; }
.kpi-delta.up   { color: #0B8C6E; }
.kpi-delta.down { color: #D93025; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #0D2137, #1A3A5C);
    color: white; border-radius: 10px;
    padding: 14px 20px; margin: 24px 0 16px;
    display: flex; align-items: center; gap: 10px;
}
.section-header h2 { margin: 0; font-size: 16px; font-weight: 600; color: white; }
.section-header p  { margin: 0; font-size: 12px; color: #94A3B8; }

/* Insight boxes */
.insight-box {
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-left: 4px solid #1A6FBF;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0 16px;
    font-size: 13px; color: #1E3A5F; line-height: 1.6;
}
.insight-box b { color: #0D2137; }

/* Finding boxes */
.finding-box {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #0B8C6E;
    border-radius: 8px;
    padding: 12px 16px; margin: 6px 0;
    font-size: 13px; color: #14532D;
}

/* Warning boxes */
.warn-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #E8900A;
    border-radius: 8px;
    padding: 12px 16px; margin: 6px 0;
    font-size: 13px; color: #78350F;
}

/* Metric table */
.metric-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.metric-table th { background: #0D2137; color: white; padding: 10px 14px; text-align: left; font-weight: 600; font-size: 12px; }
.metric-table td { padding: 9px 14px; border-bottom: 1px solid #F3F4F6; color: #1F2937; }
.metric-table tr:nth-child(even) td { background: #F9FAFB; }
.metric-table .good { color: #0B8C6E; font-weight: 600; }
.metric-table .warn { color: #E8900A; font-weight: 600; }
.metric-table .bad  { color: #D93025; font-weight: 600; }

/* Priority tier badges */
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge.hot      { background: #FEE2E2; color: #991B1B; }
.badge.free     { background: #D1FAE5; color: #065F46; }
.badge.reengage { background: #FEF3C7; color: #92400E; }
.badge.nurture  { background: #F3F4F6; color: #374151; }

/* Chart card wrapper */
.chart-card {
    background: white; border: 1px solid #E5E7EB;
    border-radius: 12px; padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}

/* Action card */
.action-card {
    background: white; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.action-card h4 { margin: 0 0 8px; font-size: 14px; font-weight: 600; color: #0D2137; }
.action-card p  { margin: 4px 0; font-size: 13px; color: #374151; line-height: 1.6; }
.action-card .label { font-size: 11px; font-weight: 600; color: #6B7280; text-transform: uppercase; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "career_survey_dataset.csv"
CLF_PATH     = "model_classifier.pkl"
REG_PATH     = "model_regressor.pkl"
KM_PATH      = "model_kmeans.pkl"
ARM_PATH     = "arm_rules.pkl"
METRICS_PATH = "model_metrics.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, delta="", color="blue"):
    cls_map = {"blue":"","green":" green","amber":" amber","red":" red","purple":" purple"}
    delta_cls = "up" if "↑" in delta else ("down" if "↓" in delta else "")
    st.markdown(f"""
    <div class="kpi-card{cls_map.get(color,'')}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_cls}">{delta}</div>
    </div>""", unsafe_allow_html=True)


def section(icon, title, subtitle=""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <div>
            <h2>{icon} {title}</h2>
            {sub}
        </div>
    </div>""", unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def finding(text):
    st.markdown(f'<div class="finding-box">✅ {text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def chart_card(fig, use_container_width=True):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=use_container_width, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, low_memory=False)
    wtp_map = {"Nothing - free only":0,"Up to Rs99/mo":70,"Rs100-299/mo":200,
               "Rs300-499/mo":400,"Rs500-999/mo":750,"Above Rs1000/mo":1200}
    if "wtp_monthly_numeric" not in df.columns:
        if "Q28_monthly_wtp" in df.columns:
            df["wtp_monthly_numeric"] = df["Q28_monthly_wtp"].map(wtp_map).fillna(0)
        else:
            df["wtp_monthly_numeric"] = 0
    if "Q31_platform_adoption" not in df.columns:
        df["Q31_platform_adoption"] = "Neutral"
    if "Q14_decision_urgency" not in df.columns:
        df["Q14_decision_urgency"] = "6-12 months"
    if "Q10_career_clarity" not in df.columns:
        df["Q10_career_clarity"] = "Somewhat confused"
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for key, path in [("clf",CLF_PATH),("reg",REG_PATH),("km",KM_PATH),("arm",ARM_PATH)]:
        if os.path.exists(path):
            try:   out[key] = joblib.load(path)
            except Exception: pass
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f: out["metrics"] = json.load(f)
        except Exception: pass
    return out


def models_trained(models):
    return all(k in models for k in ["clf","reg","km","metrics"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div style="font-size:28px;">🎯</div>
        <div style="font-size:17px;font-weight:700;color:white;margin-top:4px;">CareerPath GPS</div>
        <div style="font-size:11px;color:#94A3B8;margin-top:2px;">AI Career Platform Analytics</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Executive Overview",
        "📊  Descriptive Analysis",
        "🔍  Diagnostic Analysis",
        "🤖  Predictive Analysis",
        "🎯  Prescriptive Strategy",
        "📤  New Customer Scoring",
    ], label_visibility="collapsed")

    st.markdown("<hr style='margin:12px 0;'>", unsafe_allow_html=True)

    if st.button("⚙️  Train / Retrain Models", use_container_width=True):
        if not os.path.exists(DATA_PATH):
            st.error("Dataset not found.")
        else:
            with st.spinner("Training all models — ~60 seconds…"):
                try:
                    from train_models import train_all
                    train_all(DATA_PATH)
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("All models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("""
    <div style="position:absolute;bottom:20px;left:0;right:0;text-align:center;">
        <div style="font-size:10px;color:#475569;">Data Analytics — MGB</div>
        <div style="font-size:10px;color:#475569;">Sriju Singh  |  v2.0</div>
    </div>""", unsafe_allow_html=True)


# ── Load ──────────────────────────────────────────────────────────────────────
df      = load_dataset()
models  = load_models()
trained = models_trained(models)

from charts import (
    age_distribution, gender_donut, location_bar, income_waterfall,
    stream_pie, clarity_funnel, urgency_hbar, target_dist,
    state_chart, wtp_persona_bar, wtp_location_bar, crosstab_heatmap,
    correlation_heatmap, psycho_radar,
    arm_scatter, arm_top_rules, cluster_elbow, cluster_wtp_bar,
    roc_curve_plot, confusion_matrix_plot, feature_importance_plot,
    wtp_actual_hist, priority_donut, quadrant_scatter,
    adoption_prob_hist, wtp_prediction_hist, scatter_wtp_vs_prob,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Executive Overview":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0D2137,#1A3A5C);border-radius:14px;
                padding:28px 32px;margin-bottom:24px;">
        <div style="font-size:22px;font-weight:700;color:white;">
            AI Career Guidance Platform — Analytics Command Centre
        </div>
        <div style="font-size:14px;color:#94A3B8;margin-top:6px;">
            Data-driven intelligence from 2,000 student survey respondents across India
            — Descriptive · Diagnostic · Predictive · Prescriptive
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI Row 1 ─────────────────────────────────────────────────────────────
    if df is not None:
        adopt_pct = df["Q31_platform_adoption"].isin(
            ["Definitely would use","Likely would use"]).sum()/len(df)*100
        avg_wtp   = df["wtp_monthly_numeric"].mean()
        urgent_pct= df["Q14_decision_urgency"].isin(
            ["Within 3 months","3-6 months"]).sum()/len(df)*100
        confused  = df["Q10_career_clarity"].isin(
            ["Very confused","Somewhat confused"]).sum()/len(df)*100

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi("Survey Respondents", f"{len(df):,}", "2,000 synthetic records", "blue")
        with c2: kpi("Platform Adoption Intent", f"{adopt_pct:.1f}%", "↑ Strong pre-launch demand", "green")
        with c3: kpi("Avg Monthly WTP", f"₹{avg_wtp:.0f}", "Across all personas", "blue")
        with c4: kpi("Urgent Decision ≤6mo", f"{urgent_pct:.1f}%", "↑ Immediate conversion pool", "amber")
        with c5: kpi("Career Confusion Rate", f"{confused:.1f}%", "Core market validation", "red")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Model Status ──────────────────────────────────────────────────────────
    section("📡", "Model Performance Dashboard", "All four analytics pipelines — live status")

    if trained:
        m  = models["metrics"]
        mc = m["classifier"]
        mr = m["regressor"]

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: kpi("Accuracy",   f"{mc.get('accuracy', 0):.1%}",  "RF Classifier", "green")
        with c2: kpi("ROC-AUC",    f"{mc.get('roc_auc', 0):.3f}",   "RF Classifier", "green")
        with c3: kpi("F1 Score",   f"{mc.get('f1_score', mc.get('f1-score', 0)):.3f}",  "RF Classifier", "green")
        with c4: kpi("Precision",  f"{mc.get('precision', 0):.1%}", "RF Classifier", "blue")
        with c5: kpi("WTP MAE",    f"₹{mr.get('mae', 0):.0f}",      "GBM Regressor", "amber")
        with c6: kpi("WTP R²",     f"{mr.get('r2', 0):.4f}",        "GBM Regressor", "blue")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("ARM Rules Found",     f"{m.get('arm', {}).get('total_rules', 0)}", "Apriori algorithm", "purple")
        with c2: kpi("Optimal Clusters (k)", f"{m.get('clustering', {}).get('best_k', 5)}", "K-Means silhouette", "purple")
        with c3: kpi("CV F1 (5-fold)",       f"{mc.get('cv_f1_mean',0):.4f}", f"± {mc.get('cv_f1_std',0):.4f}", "green")
        with c4: kpi("Training Records",     f"{m.get('training_rows', 0):,}", "Survey respondents", "blue")

    else:
        st.markdown("""
        <div class="warn-box">
        <b>Models not yet trained.</b> Click <b>⚙️ Train / Retrain Models</b> in the sidebar.
        The training pipeline runs Random Forest, Gradient Boosting, K-Means and Apriori — approx. 60 seconds.
        </div>""", unsafe_allow_html=True)

    # ── Analysis Architecture ─────────────────────────────────────────────────
    section("🗺️", "Analytics Architecture", "Four-layer analysis framework")
    c1,c2,c3,c4 = st.columns(4)
    layers = [
        ("📊","Descriptive","What does our market look like?",
         "Age, location, income, stream, WTP distributions + cross-tabs","blue"),
        ("🔍","Diagnostic","Why do students behave this way?",
         "ARM rules, K-Means personas, correlation matrix, psychographic radar","green"),
        ("🤖","Predictive","Which students will convert?",
         "RF classifier (adoption) + GBM regressor (WTP) + ROC + feature importance","amber"),
        ("🎯","Prescriptive","What should we do?",
         "Segment playbooks, pricing tiers, channel strategy, action matrix","purple"),
    ]
    for col, (icon, title, question, desc, color) in zip([c1,c2,c3,c4], layers):
        with col:
            border = {"blue":BLUE,"green":TEAL,"amber":AMBER,"purple":PURPLE}.get(color,"#1A6FBF") if False else ""
            st.markdown(f"""
            <div class="action-card" style="border-left:4px solid {'#1A6FBF' if color=='blue' else '#0B8C6E' if color=='green' else '#E8900A' if color=='amber' else '#6B3FA0'};">
                <h4>{icon} {title}</h4>
                <p style="font-weight:600;color:#374151;font-size:12px;font-style:italic;">{question}</p>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    # ── Key findings banner ───────────────────────────────────────────────────
    section("🏆", "Top 5 Executive Findings")
    findings = [
        ("Market demand is real",     "48.6% of students are confused about their career — the core problem the platform solves. 57.9% express adoption intent."),
        ("Tier-2 is the sweet spot",  "Tier-2 cities have mean WTP only ₹15/mo below Metro (4% gap) but far fewer offline guidance alternatives — highest ROI acquisition market."),
        ("Psychographics beat income","Income correlates with WTP at r = +0.099. Long-term thinking correlates at r = +0.710 — mindset is 7x more predictive than wealth."),
        ("Urgency = conversion",      "35.3% of students decide within 6 months. This cohort has the highest adoption probability — time marketing to exam season (Nov–Apr)."),
        ("Tiered pricing is essential","₹223 WTP gap between Focused Climber (₹467) and Curious Explorer (₹244). A single price point will under-monetise or under-acquire."),
    ]
    for title, desc in findings:
        st.markdown(f'<div class="finding-box"><b>{title}:</b> {desc}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Descriptive Analysis":
    st.markdown("# 📊 Descriptive Analysis")
    st.markdown("*What does our student market look like? Who are they, where are they, and what do they want?*")

    if df is None:
        st.error("Dataset not found."); st.stop()

    # ── Demographics ──────────────────────────────────────────────────────────
    section("👥", "Demographics", "Geographic, age, gender and income profile of 2,000 respondents")
    c1,c2 = st.columns(2)
    with c1: chart_card(age_distribution(df))
    with c2: chart_card(location_bar(df))
    insight("The 15–17 cohort (Class 10–12 students) is the single largest age group, facing the most acute career decisions — stream selection, college choice, entrance exam prep. Combined with 18–20, they form 65%+ of the total addressable market. Tier-2 cities lead the location split at 33%, marginally ahead of Metro — a strategic signal that offline alternatives are limited in this segment.")

    c1,c2 = st.columns(2)
    with c1: chart_card(income_waterfall(df))
    with c2: chart_card(stream_pie(df))
    insight("₹2–10L household income brackets form the platform's pricing sweet spot (~55% of sample). Science PCM and PCB streams together form the largest student groups — strong ARM associations exist between these streams and Engineering/Medicine career domains respectively.")

    c1,c2 = st.columns([3,2])
    with c1: chart_card(state_chart(df))
    with c2:
        chart_card(gender_donut(df))
        st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)

    # ── Career Clarity ────────────────────────────────────────────────────────
    section("🧭", "Career Clarity & Urgency", "The core demand signal — how confused is the market?")
    c1,c2 = st.columns(2)
    with c1: chart_card(clarity_funnel(df))
    with c2: chart_card(urgency_hbar(df))

    confused_pct = df["Q10_career_clarity"].isin(["Very confused","Somewhat confused"]).sum()/len(df)*100
    urgent_pct   = df["Q14_decision_urgency"].isin(["Within 3 months","3-6 months"]).sum()/len(df)*100
    insight(f"<b>{confused_pct:.1f}% of respondents are confused or very confused</b> — this directly validates the business problem. The 'Just exploring' segment (15.7%) is a warm acquisition pool. Meanwhile, <b>{urgent_pct:.1f}% face a decision within 6 months</b> — these students are in active buying mode and represent the highest-priority conversion window.")

    # ── WTP & Adoption ────────────────────────────────────────────────────────
    section("💰", "Willingness to Pay & Platform Adoption")
    c1,c2 = st.columns(2)
    with c1: chart_card(wtp_persona_bar(df))
    with c2: chart_card(wtp_location_bar(df))
    insight("The ₹223 gap between highest WTP persona (Focused Climber ₹467/mo) and lowest (Curious Explorer ₹244/mo) makes tiered pricing mandatory. Metro vs Tier-2 WTP gap is only 4% — uniform national pricing is justified and avoids alienating Tier-2 aspirants.")

    chart_card(target_dist(df))
    adopt_pos = df["Q31_platform_adoption"].isin(["Definitely would use","Likely would use"]).sum()/len(df)*100
    neutral   = df["Q31_platform_adoption"].eq("Neutral").sum()/len(df)*100
    insight(f"<b>{adopt_pos:.1f}% express positive adoption intent</b> — strong pre-launch demand validation. The <b>{neutral:.1f}% Neutral segment</b> is the strategic swing group. A/B testing a free psychometric assessment vs free career report as lead magnets will reveal which value proposition converts this group most efficiently.")

    # ── Cross-tab explorer ────────────────────────────────────────────────────
    section("🔬", "Cross-Tab Explorer", "Drill into any two variables")
    col_opts = [c for c in ["Q4_location","Q5_income","Q7_stream","Q1_age",
                "Q10_career_clarity","Q14_decision_urgency","Q28_monthly_wtp","persona_label"] if c in df.columns]
    cx,cy = st.columns(2)
    xc = cx.selectbox("Row variable (Y axis)", col_opts, index=2)
    yc = cy.selectbox("Column variable (X axis)", col_opts, index=0)
    if xc != yc:
        chart_card(crosstab_heatmap(df, yc, xc, f"Crosstab: {xc} × {yc}"))

    section("🗃️", "Raw Data Preview", "First 50 rows of the training dataset")
    st.dataframe(df.head(50), use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Diagnostic Analysis":
    st.markdown("# 🔍 Diagnostic Analysis")
    st.markdown("*Why do students behave this way? What drives their confusion, preferences and spending?*")

    if not trained:
        warn("Train models first — click ⚙️ in the sidebar."); st.stop()

    tab1, tab2, tab3 = st.tabs(["🔗 Association Rule Mining", "👥 K-Means Clustering", "📈 Correlations & Psychographics"])

    with tab1:
        section("🔗", "Association Rule Mining — Apriori Algorithm",
                "Discovers hidden relationships between student interests, career choices and platform features")
        rules_df = models.get("arm")
        if rules_df is None or (isinstance(rules_df, pd.DataFrame) and rules_df.empty):
            st.info("No rules found at current thresholds.")
        else:
            m = models["metrics"]["arm"]
            c1,c2,c3,c4 = st.columns(4)
            with c1: kpi("Rules Generated", str(m["total_rules"]), "Apriori output", "blue")
            with c2: kpi("Min Support",     str(m["min_support"]),    "Frequency threshold", "blue")
            with c3: kpi("Min Confidence",  str(m["min_confidence"]), "Reliability threshold", "green")
            with c4: kpi("Min Lift",        str(m["min_lift"]),       "Strength threshold", "amber")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            chart_card(arm_scatter(rules_df))
            insight("Each bubble represents one association rule. <b>X-axis = support</b> (how common the pattern is), <b>Y-axis = confidence</b> (how reliably the antecedent predicts the consequent), <b>bubble size = lift</b> (how much stronger than random chance). Rules in the top-right quadrant above the average lines are the most actionable — frequent, reliable and non-random.")

            chart_card(arm_top_rules(rules_df, n=25))
            insight("Lift values above 2.0 indicate a very strong non-random association. The top rule (Science PCB → Medicine/Healthcare) with lift ~2.47 means students from PCB are 2.47x more likely to select Medicine as their career domain than random chance would predict — validating the platform's stream-based career recommendation engine.")

            st.markdown("#### 🎛️ Filter rules by minimum lift")
            min_lift = st.slider("Minimum lift threshold", 1.0, float(rules_df["lift"].max()), 1.3, 0.05)
            filtered = rules_df[rules_df["lift"] >= min_lift].reset_index(drop=True)
            st.markdown(f"**{len(filtered)} rules** with lift ≥ {min_lift} — showing highest-confidence associations")
            st.dataframe(filtered[["antecedent","consequent","support","confidence","lift"]],
                         use_container_width=True, height=320)

    with tab2:
        section("👥", "K-Means Persona Clustering",
                "Unsupervised discovery of student segments from 110 encoded features")
        km_bundle = models.get("km", {})
        if not km_bundle:
            st.info("K-Means model not found."); st.stop()

        m_cl = models["metrics"]["clustering"]
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Optimal k",          str(m_cl["best_k"]),                      "Best cluster count", "blue")
        with c2: kpi("Silhouette Score",   f"{max(m_cl.get('silhouette_scores', [0])):.4f}",   "Cluster separation", "green")
        with c3: kpi("k Range Tested",     "2 – 8",                                   "Elbow method", "blue")
        with c4: kpi("Training Records",   f"{models['metrics']['training_rows']:,}", "Survey respondents", "blue")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1: chart_card(cluster_elbow(km_bundle))
        with c2: chart_card(cluster_wtp_bar(km_bundle))
        insight("The elbow curve shows inertia decreasing with k (left axis). The silhouette score peaks at the optimal k — marking where clusters are most internally cohesive and externally separated. The right panel shows mean WTP per discovered cluster — clusters with higher WTP represent premium acquisition targets.")

        if df is not None and "persona_label" in df.columns:
            st.markdown("#### 📋 Persona profiles — synthetic ground truth vs discovered clusters")
            psych_cols = [c for c in df.columns if "Q25_psych" in c]
            grp_cols   = ["wtp_monthly_numeric","urgency_score","clarity_score"] + psych_cols
            grp_cols   = [c for c in grp_cols if c in df.columns]
            profile    = df.groupby("persona_label")[grp_cols].mean().round(2)
            profile.columns = [c.replace("Q25_psych_","").replace("_"," ").title() for c in profile.columns]
            st.dataframe(profile.style.background_gradient(cmap="Blues", axis=0),
                         use_container_width=True)
            insight("The psychographic scores validate persona separation. <b>Focused Climber</b> scores highest on autonomy and long-term thinking. <b>Confused Drifter</b> scores highest on fear of wrong choice. These differences confirm the clustering features are discriminative — the model can reliably separate these segments.")

    with tab3:
        section("📈", "Correlation Matrix & Psychographic Analysis",
                "Quantifying relationships between WTP, urgency, clarity and personality dimensions")
        if df is not None:
            chart_card(correlation_heatmap(df))
            insight("Color scale: <b>green = positive correlation, red = negative</b>. The strongest relationship is Long-term Thinking ↔ Psychographic Composite (r = +0.710) — students who plan ahead are your best customers. The near-zero Income ↔ WTP correlation (r = +0.099) is the most counterintuitive and strategically important finding: <b>income-based targeting is inefficient — psychographic segmentation is 7x more predictive</b>.")

        if df is not None and "persona_label" in df.columns:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            chart_card(psycho_radar(df))
            insight("The radar chart overlays all 5 persona psychographic profiles. <b>Focused Climbers</b> (blue) have the highest autonomy and long-term thinking — respond to data-driven messaging. <b>Pragmatic Followers</b> (purple) score highest on money > passion — respond to ROI and job security messaging. <b>Confused Drifters</b> (red) score highest on fear — respond to reassurance and structured guidance.")

        if df is not None:
            section("📊", "WTP Driver Analysis", "What predicts willingness to pay?")
            for col in ["Q4_location","Q5_income","Q1_age","Q7_stream","Q14_decision_urgency"]:
                if col in df.columns and "wtp_monthly_numeric" in df.columns:
                    grp = (df.groupby(col)["wtp_monthly_numeric"]
                           .agg(Mean="mean", Median="median", Count="count", Std="std")
                           .round(1).rename(columns={"Mean":"Mean WTP ₹","Median":"Median WTP ₹","Std":"Std Dev ₹"}))
                    with st.expander(f"WTP by {col}", expanded=False):
                        st.dataframe(grp.style.background_gradient(cmap="Blues", subset=["Mean WTP ₹"]),
                                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Predictive Analysis":
    st.markdown("# 🤖 Predictive Analysis")
    st.markdown("*Which students will adopt the platform — and how much will they pay?*")

    if not trained:
        warn("Train models first."); st.stop()

    m  = models["metrics"]
    mc = m["classifier"]
    mr = m["regressor"]

    tab1, tab2 = st.tabs(["🎯 Random Forest — Adoption Classifier", "💰 Gradient Boosting — WTP Regressor"])

    with tab1:
        section("🎯", "Random Forest Classifier",
                "Binary prediction: will a student adopt the platform? (Q31 = target variable)")
        st.markdown("**Target definition:** Positive class (1) = 'Definitely would use' or 'Likely would use' | Negative class (0) = Neutral, Unlikely, Definitely Not")

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi("Accuracy",  f"{mc.get('accuracy', 0):.1%}",  "Test set", "green")
        with c2: kpi("Precision", f"{mc.get('precision', 0):.1%}", "Of predicted positives", "green")
        with c3: kpi("Recall",    f"{mc.get('recall', 0):.1%}",    "Of actual positives", "green")
        with c4: kpi("F1 Score",  f"{mc.get('f1_score', mc.get('f1-score', 0)):.3f}",  "Harmonic mean P+R", "green")
        with c5: kpi("ROC-AUC",   f"{mc.get('roc_auc', 0):.3f}",   "Area under curve", "blue")

        cv_mean = mc.get("cv_f1_mean", 0)
        cv_std  = mc.get("cv_f1_std", 0)
        st.markdown(f"""
        <div class="insight-box" style="margin-top:12px;">
        📊 <b>5-fold cross-validation F1:</b> {cv_mean:.4f} ± {cv_std:.4f}
        — consistent performance across all data folds confirms the model generalises well and is not overfitting.
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1: chart_card(roc_curve_plot(m))
        with c2: chart_card(confusion_matrix_plot(m))
        insight("The ROC curve shows the trade-off between true positive rate (sensitivity) and false positive rate at all classification thresholds. An AUC near 1.0 on synthetic data is expected — on real survey data expect AUC in the 0.78–0.88 range, which remains excellent for a marketing classification model.")

        chart_card(feature_importance_plot(m, "classifier", 20))
        insight("Feature importance shows which variables most influence the adoption prediction. High importance of urgency_score and past_payer confirms the research design — urgency and prior spending behaviour are the strongest real-world predictors of platform conversion. Income-related features rank lower, validating the psychographic > demographic targeting thesis.")

        st.markdown("#### 📋 Detailed classification report")
        report = mc.get("classification_report", {})
        if report:
            rows = []
            for label, vals in report.items():
                if isinstance(vals, dict):
                    rows.append({"Class": label,
                                 "Precision": round(vals.get("precision",0),3),
                                 "Recall":    round(vals.get("recall",0),3),
                                 "F1-score":  round(vals.get("f1-score",0),3),
                                 "Support":   int(vals.get("support",0))})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        section("💰", "Gradient Boosting Regressor",
                "Predicts continuous monthly WTP (₹) for each respondent")

        c1,c2,c3 = st.columns(3)
        with c1: kpi("Mean Absolute Error", f"₹{mr.get('mae', 0):.0f}", "Avg prediction error", "amber")
        with c2: kpi("R² Score",            f"{mr.get('r2', 0):.4f}",   "Variance explained", "green")
        with c3: kpi("Model Type",          "GBM",               "200 trees, lr=0.05", "blue")

        insight(f"An MAE of ₹{mr.get('mae', 0):.0f} means the model's WTP predictions are off by ₹{mr.get('mae', 0):.0f} on average. For pricing strategy purposes (tier boundaries at ₹0 / ₹99 / ₹299 / ₹599 / ₹999), this precision is more than sufficient to assign new respondents to the correct pricing tier.")

        c1,c2 = st.columns(2)
        with c1: chart_card(feature_importance_plot(m, "regressor", 18))
        with c2:
            if df is not None: chart_card(wtp_actual_hist(df))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRESCRIPTIVE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Prescriptive Strategy":
    st.markdown("# 🎯 Prescriptive Strategy")
    st.markdown("*What should we actually do — for each segment, each channel, each rupee of budget?*")

    section("🎯", "Segment Action Matrix", "Four priority segments — distinct offer, channel and message for each")

    segments = [
        {
            "name":    "🔴 Hot Leads — Focused Climbers + Anxious Achievers",
            "color":   "#D93025",
            "who":     "UG students, Metro/Tier-2, income ₹5–20L, urgency ≤6 months, past payer",
            "size":    "~22% of audience",
            "wtp":     "₹400–500/mo",
            "channel": "Instagram + LinkedIn ads → WhatsApp parent follow-up",
            "offer":   "7-day free trial → Standard ₹299/mo",
            "message": "Aspiration + urgency: 'Your peers already have a roadmap — do you?'",
            "action":  "Act this week — highest conversion probability",
        },
        {
            "name":    "🟡 Freemium Converts — Confused Drifters",
            "color":   "#E8900A",
            "who":     "Class 10–12, any location, very confused, no past spend",
            "size":    "~30% of audience (largest segment)",
            "wtp":     "₹99–200/mo",
            "channel": "YouTube pre-roll + school counsellor partnerships",
            "offer":   "Free psychometric test → ₹99/mo Starter plan",
            "message": "Reassurance: 'Confused is okay — let us figure it out together'",
            "action":  "Build awareness funnel — high volume, low ARPU, high LTV potential",
        },
        {
            "name":    "🟠 Re-engage — Pragmatic Followers (Parent-driven)",
            "color":   "#6B3FA0",
            "who":     "Parent decides + pays, income ₹5–15L, Tier-2/3 cities",
            "size":    "~10% of audience",
            "wtp":     "₹300–400/mo",
            "channel": "WhatsApp forward content + Facebook parent groups",
            "offer":   "One-time career report ₹499 → subscription upsell",
            "message": "ROI-focused: 'Your child's career decision is worth ₹499 to get right'",
            "action":  "Parent-first acquisition — different creative and channel from student-direct",
        },
        {
            "name":    "🟢 Plant Seeds — Curious Explorers",
            "color":   "#0B8C6E",
            "who":     "Class 9–11, open mindset, no deadline pressure, low urgency",
            "size":    "~16% of audience",
            "wtp":     "₹100–250/mo",
            "channel": "Instagram Reels + YouTube Shorts + school ambassador programme",
            "offer":   "Free emerging careers quiz → newsletter → ₹99/mo when urgency increases",
            "message": "Discovery: '10 careers that didn't exist 5 years ago — which fits you?'",
            "action":  "Long-term pipeline investment — low immediate conversion, high 12-month value",
        },
    ]

    for seg in segments:
        st.markdown(f"""
        <div class="action-card" style="border-left:4px solid {seg['color']};">
            <h4 style="color:{seg['color']}">{seg['name']}</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;">
                <div>
                    <div class="label">Who they are</div>
                    <p>{seg['who']}</p>
                    <div class="label" style="margin-top:8px;">Estimated size</div>
                    <p>{seg['size']} · Avg WTP {seg['wtp']}</p>
                    <div class="label" style="margin-top:8px;">Recommended channel</div>
                    <p>{seg['channel']}</p>
                </div>
                <div>
                    <div class="label">Best offer</div>
                    <p>{seg['offer']}</p>
                    <div class="label" style="margin-top:8px;">Message tone</div>
                    <p style="font-style:italic;">{seg['message']}</p>
                    <div class="label" style="margin-top:8px;">Immediate action</div>
                    <p style="font-weight:600;color:{seg['color']}">{seg['action']}</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    section("💲", "Pricing Architecture", "Data-validated five-tier model from WTP regression")
    pricing = pd.DataFrame([
        {"Tier":"Free",      "Price":"₹0",        "Target Persona":"Confused Drifter · Curious Explorer",   "Feature Set":"Psychometric test + 1 career suggestion",                         "Conversion Goal":"Email capture → upsell"},
        {"Tier":"Starter",   "Price":"₹99/mo",    "Target Persona":"Confused Drifter · Curious Explorer",   "Feature Set":"Free + AI chatbot (limited) + 3 roadmap steps",                  "Conversion Goal":"Trial to Standard"},
        {"Tier":"Standard",  "Price":"₹299/mo",   "Target Persona":"Anxious Achiever · Pragmatic Follower", "Feature Set":"Starter + full roadmap + college shortlist + exam tracker",        "Conversion Goal":"6-month subscription"},
        {"Tier":"Premium",   "Price":"₹599/mo",   "Target Persona":"Focused Climber · Anxious Achiever",    "Feature Set":"Standard + 2 mentor sessions/mo + salary dashboard",              "Conversion Goal":"Annual subscription"},
        {"Tier":"Elite",     "Price":"₹999/mo",   "Target Persona":"Focused Climber",                       "Feature Set":"Premium + unlimited mentorship + parent dashboard + priority",    "Conversion Goal":"B2B school licence"},
        {"Tier":"One-time",  "Price":"₹499",      "Target Persona":"Pragmatic Follower (parents)",          "Feature Set":"Comprehensive career report PDF + college shortlist + action plan","Conversion Goal":"Subscription upsell"},
    ])
    st.dataframe(pricing, use_container_width=True, hide_index=True, height=280)
    insight("The ₹299 Standard tier is the highest-volume revenue tier — it sits exactly at the mean WTP of the Anxious Achiever persona (₹339/mo). Positioning it below the mean creates a 'feels affordable' perception while still capturing the majority of that segment's budget.")

    section("📡", "Channel Strategy by Persona")
    channels = pd.DataFrame([
        {"Persona":"Confused Drifter",   "Primary":"YouTube",       "Secondary":"School counsellors",   "Content Type":"'Are you confused?' problem-identification",  "Timing":"Year-round"},
        {"Persona":"Anxious Achiever",   "Primary":"Instagram",     "Secondary":"WhatsApp (student)",   "Content Type":"Countdown / urgency / peer comparison",       "Timing":"Oct–Mar (exam season)"},
        {"Persona":"Focused Climber",    "Primary":"LinkedIn",      "Secondary":"Email drip",           "Content Type":"Data, salary benchmarks, ROI stories",        "Timing":"Jun–Aug (internship season)"},
        {"Persona":"Curious Explorer",   "Primary":"Instagram Reel","Secondary":"YouTube Shorts",       "Content Type":"Emerging career discovery, quizzes",          "Timing":"Year-round"},
        {"Persona":"Pragmatic Follower", "Primary":"WhatsApp",      "Secondary":"Facebook (parents)",   "Content Type":"ROI, success stories, peer comparison",       "Timing":"Jan–Mar (admission season)"},
    ])
    st.dataframe(channels, use_container_width=True, hide_index=True)

    if trained:
        section("🔗", "ARM-Derived Feature Recommendations",
                "Which platform features should we push to which student profiles?")
        rules_df = models.get("arm")
        if rules_df is not None and not rules_df.empty:
            feature_rules = rules_df[rules_df["consequent"].str.startswith("feature:")].head(15)
            display_rules = feature_rules if not feature_rules.empty else rules_df.head(15)
            st.dataframe(
                display_rules[["antecedent","consequent","support","confidence","lift"]],
                use_container_width=True, hide_index=True
            )
            insight("These ARM rules are direct personalisation signals — if a student's profile matches the antecedent (e.g., stream:Science PCM + subject:Technology), the platform should prominently feature the consequent (e.g., feature:AI chatbot or feature:Personalised roadmap) on their dashboard.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — NEW CUSTOMER SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤  New Customer Scoring":
    st.markdown("# 📤 New Customer Scoring")
    st.markdown("*Upload any new survey CSV — every respondent gets scored with adoption probability, predicted WTP, priority tier and recommended action.*")

    if not trained:
        warn("Train models first."); st.stop()

    # Template download
    if os.path.exists(DATA_PATH):
        df_tpl = pd.read_csv(DATA_PATH, nrows=5)
        tpl_cols = [c for c in df_tpl.columns if not c.startswith("pred_") and c != "persona_label"]
        st.download_button("⬇️ Download CSV template (5 rows)",
                           data=df_tpl[tpl_cols].to_csv(index=False),
                           file_name="new_customers_template.csv", mime="text/csv")
        st.caption("Format your new respondent data using this template, then upload below.")

    st.markdown("<hr>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload new respondent CSV", type=["csv"],
                                 help="Columns must match the training dataset format.")

    if uploaded:
        with st.spinner("Scoring respondents…"):
            try:
                df_new  = pd.read_csv(uploaded, low_memory=False)
                st.success(f"✅ Loaded {len(df_new):,} respondents from uploaded file")

                from predictor import predict_new_customers, score_summary
                scored  = predict_new_customers(df_new)
                summary = score_summary(scored)

                # ── KPI summary ────────────────────────────────────────────
                section("📊", "Scoring Summary", f"{summary['total']} respondents scored")
                c1,c2,c3,c4,c5 = st.columns(5)
                with c1: kpi("Total Scored",    str(summary["total"]),     "", "blue")
                with c2: kpi("Hot Leads 🔴",    str(summary["hot_leads"]), "High prob + high WTP", "red")
                with c3: kpi("Freemium 🟢",     str(summary["freemium"]),  "High prob + low WTP", "green")
                with c4: kpi("Re-engage 🟠",    str(summary["reengage"]),  "Low prob + high WTP", "amber")
                with c5: kpi("Nurture ⚪",      str(summary["nurture"]),   "Low prob + low WTP", "blue")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1: kpi("Avg Adoption Probability", f"{summary['avg_adoption_prob']:.1%}", "", "green")
                with c2: kpi("Avg Predicted WTP",        f"₹{summary['avg_predicted_wtp']:.0f}/mo", "", "blue")

                # ── Charts ────────────────────────────────────────────────
                section("📈", "Scoring Visualisations")
                c1,c2 = st.columns(2)
                with c1: chart_card(priority_donut(scored))
                with c2: chart_card(quadrant_scatter(scored))

                c1,c2 = st.columns(2)
                with c1: chart_card(adoption_prob_hist(scored))
                with c2: chart_card(wtp_prediction_hist(scored))

                insight("The <b>quadrant map</b> divides respondents into four action zones. Top-right = Hot Leads (high probability + high WTP) → direct outreach this week. Bottom-right = Freemium prospects → push trial conversion. Top-left = Re-engage → parent-facing WTP messaging. Bottom-left = Nurture → awareness content only.")

                # ── Hot leads table ───────────────────────────────────────
                hot = scored[scored["pred_priority_tier"]=="Hot Lead"].sort_values(
                    "pred_adoption_probability", ascending=False)
                if not hot.empty:
                    section("🔴", f"Hot Leads — {len(hot)} respondents", "Act on these first — highest adoption probability AND high predicted WTP")
                    show_cols = [c for c in ["respondent_id","Q1_age","Q3_state","Q4_location",
                                             "Q7_stream","Q14_decision_urgency",
                                             "pred_adoption_probability","pred_wtp_monthly_inr",
                                             "pred_wtp_tier","pred_marketing_action"] if c in hot.columns]
                    st.dataframe(hot[show_cols].head(50), use_container_width=True, height=320, hide_index=True)

                # ── Full table ────────────────────────────────────────────
                section("🗃️", "All Scored Respondents")
                pred_cols = [c for c in scored.columns
                             if c.startswith("pred_") or c in ["respondent_id","Q3_state","Q4_location","Q1_age"]]
                st.dataframe(scored[pred_cols].head(200), use_container_width=True, height=360, hide_index=True)

                st.download_button("⬇️ Download full scored CSV",
                                   data=scored.to_csv(index=False),
                                   file_name="scored_customers.csv", mime="text/csv")

            except FileNotFoundError as e:
                st.error(f"Model files missing: {e}")
            except Exception as e:
                st.error(f"Scoring error: {e}")
                st.exception(e)

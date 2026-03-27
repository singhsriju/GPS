"""
charts.py — Professional MBA-grade chart library
Every chart carries executive headlines, annotated thresholds,
and business-context formatting. No raw student-style plots.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY   = "#0D2137"
BLUE   = "#1A6FBF"
LBLUE  = "#4FA3E0"
TEAL   = "#0B8C6E"
LTEAL  = "#4ABFA3"
AMBER  = "#E8900A"
RED    = "#D93025"
PURPLE = "#6B3FA0"
GRAY   = "#6C757D"

PERSONA_COLORS = {
    "Confused Drifter":   "#D93025",
    "Anxious Achiever":   "#E8900A",
    "Focused Climber":    "#1A6FBF",
    "Curious Explorer":   "#0B8C6E",
    "Pragmatic Follower": "#6B3FA0",
}
PRIORITY_COLORS = {
    "Hot Lead":         "#D93025",
    "Freemium Convert": "#0B8C6E",
    "Re-engage":        "#E8900A",
    "Nurture":          "#6C757D",
}


def _base(fig, height=380, margin_t=52):
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=margin_t, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#1F2937"),
        title=dict(font=dict(size=14, color=NAVY, family="Inter, Arial, sans-serif")),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E5E7EB",
                    borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", bordercolor="#E5E7EB",
                        font_size=12, font_family="Inter, Arial, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", gridwidth=1,
                     showline=True, linecolor="#E5E7EB", linewidth=1,
                     tickfont=dict(size=11, color="#6B7280"))
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", gridwidth=1,
                     showline=False, tickfont=dict(size=11, color="#6B7280"))
    return fig


def age_distribution(df):
    order = ["Under 15","15-17","18-20","21-23","24 or above"]
    vc = df["Q1_age"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Age","Count"]
    vc["Pct"] = (vc["Count"]/vc["Count"].sum()*100).round(1)
    colors = [BLUE if a in ["15-17","18-20"] else LBLUE for a in vc["Age"]]
    fig = go.Figure(go.Bar(x=vc["Age"], y=vc["Count"], marker_color=colors,
        text=[f"{p}%" for p in vc["Pct"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} respondents (%{text})<extra></extra>"))
    peak_val = vc.loc[vc["Age"]=="15-17","Count"].values
    if len(peak_val):
        fig.add_annotation(x="15-17", y=peak_val[0],
            text="<b>Primary target cohort</b>", showarrow=True,
            arrowhead=2, ax=70, ay=-38, font=dict(size=10, color=BLUE),
            bgcolor="white", bordercolor=BLUE, borderwidth=1)
    fig.update_layout(title="Age distribution of respondents", showlegend=False)
    return _base(fig, 340)


def gender_donut(df):
    vc = df["Q2_gender"].value_counts(dropna=True).reset_index()
    vc.columns = ["Gender","Count"]
    fig = go.Figure(go.Pie(labels=vc["Gender"], values=vc["Count"], hole=0.55,
        marker=dict(colors=[BLUE,LBLUE,GRAY,"#D1D5DB"], line=dict(color="white",width=2)),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>"))
    fig.update_layout(title="Gender split", showlegend=False,
        annotations=[dict(text="Gender", x=0.5, y=0.5,
                          font=dict(size=13,color=NAVY), showarrow=False)])
    return _base(fig, 300, 44)


def location_bar(df):
    order = ["Metro city","Tier-2 city","Tier-3 city","Rural / Village"]
    vc = df["Q4_location"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Location","Count"]
    vc["Pct"] = (vc["Count"]/vc["Count"].sum()*100).round(1)
    colors = [BLUE, TEAL, LBLUE, GRAY]
    fig = go.Figure(go.Bar(x=vc["Location"], y=vc["Count"], marker_color=colors,
        text=[f"{p}%" for p in vc["Pct"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} respondents (%{text})<extra></extra>"))
    t2 = vc.loc[vc["Location"]=="Tier-2 city","Count"].values
    if len(t2):
        fig.add_annotation(x="Tier-2 city", y=t2[0],
            text="<b>Highest ROI market</b><br>WTP only 4% below Metro",
            showarrow=True, arrowhead=2, ax=80, ay=-46,
            font=dict(size=10, color=TEAL), bgcolor="white", bordercolor=TEAL, borderwidth=1)
    fig.update_layout(title="Location tier distribution", showlegend=False)
    return _base(fig, 340)


def income_waterfall(df):
    order = ["Below Rs2L","Rs2-5L","Rs5-10L","Rs10-20L","Above Rs20L","Prefer not to say"]
    vc = df["Q5_income"].value_counts(dropna=True).reindex(order, fill_value=0).reset_index()
    vc.columns = ["Income","Count"]
    vc["Pct"] = (vc["Count"]/vc["Count"].sum()*100).round(1)
    colors = [BLUE if i in [1,2] else LBLUE for i in range(len(vc))]
    fig = go.Figure(go.Bar(y=vc["Income"], x=vc["Count"], orientation="h",
        marker_color=colors, text=[f"{p}%" for p in vc["Pct"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} respondents (%{text})<extra></extra>"))
    fig.add_hrect(y0=0.5, y1=2.5, fillcolor=BLUE, opacity=0.06,
        annotation_text="Platform pricing sweet spot",
        annotation_position="top right",
        annotation_font=dict(size=10, color=BLUE))
    fig.update_layout(title="Household income distribution", showlegend=False,
                      yaxis=dict(autorange="reversed"))
    return _base(fig, 340)


def stream_pie(df):
    vc = df["Q7_stream"].value_counts().reset_index()
    vc.columns = ["Stream","Count"]
    fig = go.Figure(go.Pie(labels=vc["Stream"], values=vc["Count"], hole=0.45,
        marker=dict(colors=[BLUE,TEAL,LBLUE,LTEAL,AMBER,PURPLE,GRAY],
                    line=dict(color="white",width=2)),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>"))
    fig.update_layout(title="Stream / field distribution")
    return _base(fig, 340, 44)


def clarity_funnel(df):
    order = ["Completely clear","Mostly clear","Just exploring","Somewhat confused","Very confused"]
    vc = df["Q10_career_clarity"].value_counts().reindex(order, fill_value=0)
    colors = [TEAL, LTEAL, AMBER, "#F97316", RED]
    pcts = [round(v/vc.sum()*100,1) for v in vc.values]
    fig = go.Figure(go.Bar(x=list(order), y=vc.values, marker_color=colors,
        text=[f"{p}%" for p in pcts], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} respondents (%{text})<extra></extra>"))
    confused_pct = round((vc.get("Somewhat confused",0)+vc.get("Very confused",0))/vc.sum()*100,1)
    fig.add_annotation(x=3.5, y=max(vc.values)*0.75,
        text=f"<b>{confused_pct}% of market is confused</b><br>Core demand validated",
        showarrow=False, font=dict(size=11,color=RED),
        bgcolor="rgba(255,255,255,0.92)", bordercolor=RED, borderwidth=1)
    fig.update_layout(title="Career clarity distribution — Q10 (primary clustering variable)", showlegend=False)
    return _base(fig, 360)


def urgency_hbar(df):
    order = ["Within 3 months","3-6 months","6-12 months","1-2 years","More than 2 years"]
    vc = df["Q14_decision_urgency"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Urgency","Count"]
    vc["Pct"] = (vc["Count"]/vc["Count"].sum()*100).round(1)
    colors = [RED, AMBER, BLUE, LBLUE, GRAY]
    fig = go.Figure(go.Bar(y=vc["Urgency"], x=vc["Count"], orientation="h",
        marker_color=colors, text=[f"{p}%" for p in vc["Pct"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} respondents (%{text})<extra></extra>"))
    if len(vc) >= 2:
        urgent_pct = round((vc.iloc[0]["Count"]+vc.iloc[1]["Count"])/vc["Count"].sum()*100,1)
        cutoff = vc.iloc[1]["Count"]
        fig.add_vline(x=cutoff, line_dash="dot", line_color=RED, line_width=1.5,
            annotation_text=f"  {urgent_pct}% decide within 6 months",
            annotation_font=dict(size=10, color=RED))
    fig.update_layout(title="Decision urgency — Q14 (strongest conversion predictor)", showlegend=False,
                      yaxis=dict(autorange="reversed"))
    return _base(fig, 340)


def target_dist(df):
    order = ["Definitely would use","Likely would use","Neutral","Unlikely to use","Definitely would NOT use"]
    vc = df["Q31_platform_adoption"].value_counts().reindex(order, fill_value=0).reset_index()
    vc.columns = ["Response","Count"]
    vc["Pct"] = (vc["Count"]/vc["Count"].sum()*100).round(1)
    ADOPT_COLORS = {
        "Definitely would use":     TEAL,
        "Likely would use":         LTEAL,
        "Neutral":                  GRAY,
        "Unlikely to use":          AMBER,
        "Definitely would NOT use": RED,
    }
    colors = [ADOPT_COLORS.get(r, GRAY) for r in vc["Response"]]
    fig = go.Figure(go.Bar(x=vc["Response"], y=vc["Count"], marker_color=colors,
        text=[f"{p}%" for p in vc["Pct"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} respondents (%{text})<extra></extra>"))
    pos = vc[vc["Response"].isin(["Definitely would use","Likely would use"])]["Count"].sum()
    pos_pct = round(pos/vc["Count"].sum()*100,1)
    fig.add_hrect(y0=0, y1=vc["Count"].max()*1.18, x0=-0.5, x1=1.5,
                  fillcolor=TEAL, opacity=0.04)
    fig.add_annotation(x=0.5, y=vc["Count"].max()*0.82,
        text=f"<b>{pos_pct}% positive intent</b>", showarrow=False,
        font=dict(size=12, color=TEAL),
        bgcolor="rgba(255,255,255,0.92)", bordercolor=TEAL, borderwidth=1)
    fig.update_layout(title="Platform adoption intent — Q31 (classification target variable)", showlegend=False)
    return _base(fig, 360)


def state_chart(df):
    vc = df["Q3_state"].value_counts().head(12).reset_index()
    vc.columns = ["State","Count"]
    fig = go.Figure(go.Bar(y=vc["State"], x=vc["Count"], orientation="h",
        marker=dict(color=vc["Count"], colorscale=[[0,"#BDD7EE"],[1,NAVY]], showscale=False),
        text=vc["Count"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} respondents<extra></extra>"))
    fig.update_layout(title="Top 12 states by respondent count", yaxis=dict(autorange="reversed"))
    return _base(fig, 400)


def wtp_persona_bar(df):
    if "persona_label" not in df.columns or "wtp_monthly_numeric" not in df.columns:
        return go.Figure()
    grp = df.groupby("persona_label")["wtp_monthly_numeric"].mean().reset_index()
    grp.columns = ["Persona","AvgWTP"]
    grp = grp.sort_values("AvgWTP", ascending=True)
    colors = [PERSONA_COLORS.get(p, GRAY) for p in grp["Persona"]]
    fig = go.Figure(go.Bar(y=grp["Persona"], x=grp["AvgWTP"], orientation="h",
        marker_color=colors, text=[f"₹{v:.0f}" for v in grp["AvgWTP"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg WTP: ₹%{x:.0f}<extra></extra>"))
    grand_mean = df["wtp_monthly_numeric"].mean()
    fig.add_vline(x=grand_mean, line_dash="dash", line_color=NAVY, line_width=1.5,
        annotation_text=f"  Grand mean ₹{grand_mean:.0f}",
        annotation_font=dict(size=10, color=NAVY))
    fig.update_layout(title="Mean monthly WTP (₹) by persona", showlegend=False)
    return _base(fig, 340)


def wtp_location_bar(df):
    if "wtp_monthly_numeric" not in df.columns:
        return go.Figure()
    grp = df.groupby("Q4_location")["wtp_monthly_numeric"].mean().reset_index()
    grp.columns = ["Location","AvgWTP"]
    grp = grp.sort_values("AvgWTP", ascending=False)
    colors = [BLUE, TEAL, LBLUE, GRAY][:len(grp)]
    fig = go.Figure(go.Bar(x=grp["Location"], y=grp["AvgWTP"], marker_color=colors,
        text=[f"₹{v:.0f}" for v in grp["AvgWTP"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg WTP: ₹%{y:.0f}<extra></extra>"))
    metro = grp.loc[grp["Location"]=="Metro city","AvgWTP"].values
    t2 = grp.loc[grp["Location"]=="Tier-2 city","AvgWTP"].values
    if len(metro) and len(t2):
        gap = abs(metro[0]-t2[0])/metro[0]*100
        fig.add_annotation(x="Tier-2 city", y=t2[0],
            text=f"<b>Only {gap:.0f}% below Metro</b><br>Highest ROI acquisition target",
            showarrow=True, arrowhead=2, ax=90, ay=-42,
            font=dict(size=10, color=TEAL), bgcolor="white", bordercolor=TEAL, borderwidth=1)
    fig.update_layout(title="Mean monthly WTP (₹) by location", showlegend=False)
    return _base(fig, 340)


def crosstab_heatmap(df, col_x, col_y, title=""):
    try:
        ct = pd.crosstab(df[col_y], df[col_x])
        fig = px.imshow(ct, color_continuous_scale=[[0,"#EBF4FB"],[1,NAVY]],
                        text_auto=True, aspect="auto")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(title=title or f"{col_x} × {col_y}")
        return _base(fig, 420)
    except Exception:
        return go.Figure()


def correlation_heatmap(df):
    num_cols = ["wtp_monthly_numeric","urgency_score","clarity_score",
                "Q25_psych_fear_wrong_choice","Q25_psych_prefer_independent",
                "Q25_psych_financial_over_passion","Q25_psych_risk_tolerance",
                "Q25_psych_long_term_thinking"]
    present = [c for c in num_cols if c in df.columns]
    short   = [c.replace("Q25_psych_","").replace("_"," ").title()[:14] for c in present]
    corr    = df[present].corr().round(3)
    fig = go.Figure(go.Heatmap(z=corr.values, x=short, y=short,
        colorscale=[[0,"#D93025"],[0.5,"white"],[1,"#0B8C6E"]],
        zmid=0, zmin=-1, zmax=1, text=corr.round(2).values,
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Pearson correlation matrix — key numeric variables")
    return _base(fig, 460, 52)


def psycho_radar(df):
    if "persona_label" not in df.columns:
        return go.Figure()
    psych   = ["Q25_psych_fear_wrong_choice","Q25_psych_prefer_independent",
               "Q25_psych_financial_over_passion","Q25_psych_risk_tolerance",
               "Q25_psych_long_term_thinking"]
    labels  = ["Fear of<br>wrong choice","Autonomy","Money ><br>Passion",
               "Risk<br>tolerance","Long-term<br>thinking"]
    present = [c for c in psych if c in df.columns]
    if not present:
        return go.Figure()
    fig = go.Figure()
    for persona, color in PERSONA_COLORS.items():
        sub = df[df["persona_label"]==persona]
        if sub.empty: continue
        vals = [sub[c].mean() for c in present] + [sub[present[0]].mean()]
        lbls = labels[:len(present)] + [labels[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=lbls, fill="toself",
            name=persona, line=dict(color=color, width=2), opacity=0.72))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[1,5], tickfont=dict(size=9))),
        title="Psychographic profile by persona",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    return _base(fig, 460, 52)


def arm_bubble(rules_df):
    if rules_df is None or (hasattr(rules_df,"empty") and rules_df.empty):
        return go.Figure()
    top = rules_df.head(100)
    fig = go.Figure(go.Scatter(x=top["support"], y=top["confidence"], mode="markers",
        marker=dict(size=top["lift"]*6, sizemode="area",
                    color=top["lift"], colorscale=[[0,LBLUE],[0.5,BLUE],[1,NAVY]],
                    showscale=True, colorbar=dict(title="Lift",thickness=12,len=0.7),
                    line=dict(color="white",width=0.5)),
        text=[f"{r['antecedent'][:25]} → {r['consequent'][:25]}" for _,r in top.iterrows()],
        hovertemplate="<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<br>Lift: %{marker.color:.2f}<extra></extra>"))
    fig.add_vline(x=top["support"].mean(), line_dash="dot", line_color=GRAY, line_width=1,
                  annotation_text=" Avg support", annotation_font=dict(size=9,color=GRAY))
    fig.add_hline(y=top["confidence"].mean(), line_dash="dot", line_color=GRAY, line_width=1,
                  annotation_text="Avg confidence", annotation_font=dict(size=9,color=GRAY))
    fig.update_layout(title="Association rules — support vs confidence (bubble size = lift)",
                      xaxis_title="Support", yaxis_title="Confidence")
    return _base(fig, 440)


def arm_top_bar(rules_df, n=20):
    if rules_df is None or (hasattr(rules_df,"empty") and rules_df.empty):
        return go.Figure()
    top = rules_df.head(n).copy()
    top["rule"] = top["antecedent"].str[:26]+" → "+top["consequent"].str[:26]
    fig = go.Figure(go.Bar(x=top["lift"], y=top["rule"], orientation="h",
        marker=dict(color=top["confidence"], colorscale=[[0,LBLUE],[1,NAVY]],
                    showscale=True, colorbar=dict(title="Confidence",thickness=12,len=0.7)),
        text=[f"Lift {v:.2f}" for v in top["lift"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}<extra></extra>"))
    fig.update_layout(title=f"Top {n} association rules ranked by lift",
                      yaxis=dict(autorange="reversed"), xaxis_title="Lift")
    return _base(fig, 560, 52)


def cluster_elbow_fig(km_bundle):
    ks      = list(range(2, 2+len(km_bundle["inertias"])))
    best_k  = km_bundle.get("best_k", 5)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=km_bundle["inertias"], mode="lines+markers",
        name="Inertia", line=dict(color=BLUE, width=2.5), marker=dict(size=8, color=BLUE),
        hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=km_bundle["sil_scores"], mode="lines+markers",
        name="Silhouette", line=dict(color=TEAL, width=2.5, dash="dash"), marker=dict(size=8,color=TEAL),
        hovertemplate="k=%{x}<br>Silhouette: %{y:.4f}<extra></extra>"), secondary_y=True)
    fig.add_vline(x=best_k, line_dash="dot", line_color=AMBER, line_width=2,
                  annotation_text=f"  Optimal k={best_k}", annotation_font=dict(color=AMBER,size=11))
    fig.update_yaxes(title_text="Inertia", secondary_y=False, title_font=dict(color=BLUE))
    fig.update_yaxes(title_text="Silhouette score", secondary_y=True, title_font=dict(color=TEAL))
    fig.update_xaxes(title_text="k (clusters)")
    fig.update_layout(title="K-Means elbow + silhouette curve",
                      legend=dict(orientation="h", y=1.12))
    return _base(fig, 380)


def cluster_profile_fig(km_bundle):
    sizes   = km_bundle.get("cluster_sizes", {})
    wtps    = km_bundle.get("cluster_wtp", {})
    clusters = sorted(sizes.keys(), key=lambda x: int(x))
    labels   = [f"Cluster {c}" for c in clusters]
    counts   = [sizes[c] for c in clusters]
    avgs     = [wtps.get(c, 0) for c in clusters]
    colors   = [BLUE, TEAL, AMBER, PURPLE, RED, LBLUE][:len(clusters)]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Cluster size (respondents)","Mean monthly WTP (₹)"))
    fig.add_trace(go.Bar(x=labels, y=counts, marker_color=colors,
        text=counts, textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} respondents<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=avgs, marker_color=colors,
        text=[f"₹{v:.0f}" for v in avgs], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg WTP: ₹%{y:.0f}<extra></extra>"), row=1, col=2)
    fig.update_layout(title="Discovered cluster profiles", showlegend=False)
    return _base(fig, 360, 64)


def roc_curve_fig(metrics):
    roc = metrics["classifier"]["roc_curve"]
    auc = roc["auc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
        name=f"Classifier (AUC={auc:.3f})",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(26,111,191,0.08)",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        name="Random baseline", line=dict(color=GRAY, width=1.5, dash="dash")))
    quality = "Excellent" if auc > 0.9 else ("Good" if auc > 0.75 else "Fair")
    fig.add_annotation(x=0.6, y=0.38,
        text=f"<b>AUC = {auc:.3f}</b><br>{quality} model discrimination",
        showarrow=False, font=dict(size=12, color=BLUE),
        bgcolor="rgba(255,255,255,0.92)", bordercolor=BLUE, borderwidth=1)
    fig.update_layout(title="ROC curve — Random Forest classifier",
                      xaxis_title="False positive rate",
                      yaxis_title="True positive rate",
                      legend=dict(x=0.55, y=0.15))
    return _base(fig, 400)


def confusion_matrix_fig(metrics):
    cm     = np.array(metrics["classifier"]["confusion_matrix"])
    labels = ["Will not adopt", "Will adopt"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels,
        colorscale=[[0,"#EBF4FB"],[1,NAVY]],
        text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=18),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False))
    fig.update_layout(title="Confusion matrix", xaxis_title="Predicted",
                      yaxis_title="Actual", yaxis=dict(autorange="reversed"))
    return _base(fig, 340)


def feature_importance_fig(metrics, model_key="classifier", n=18):
    feats = metrics[model_key]["top_features"]
    df_fi = pd.DataFrame(list(feats.items()), columns=["Feature","Importance"])
    df_fi = df_fi.sort_values("Importance").tail(n)
    df_fi["Feature"] = (df_fi["Feature"].str.replace("_enc","")
                        .str.replace("__"," — ").str.replace("_"," ").str[:44])
    fig = go.Figure(go.Bar(x=df_fi["Importance"], y=df_fi["Feature"], orientation="h",
        marker=dict(color=df_fi["Importance"], colorscale=[[0,LBLUE],[1,NAVY]], showscale=False),
        text=[f"{v:.4f}" for v in df_fi["Importance"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
    lbl = "classifier" if model_key == "classifier" else "WTP regressor"
    fig.update_layout(title=f"Feature importance — {lbl}",
                      xaxis_title="Importance", yaxis=dict(autorange="reversed"))
    return _base(fig, max(440, n*26), 52)


def wtp_actual_hist(df):
    fig = go.Figure(go.Histogram(x=df["wtp_monthly_numeric"], nbinsx=18,
        marker_color=BLUE, opacity=0.85,
        hovertemplate="WTP: ₹%{x}<br>Count: %{y}<extra></extra>"))
    mean_wtp = df["wtp_monthly_numeric"].mean()
    fig.add_vline(x=mean_wtp, line_dash="dash", line_color=RED, line_width=2,
                  annotation_text=f"  Mean ₹{mean_wtp:.0f}",
                  annotation_font=dict(size=11, color=RED))
    fig.update_layout(title="WTP distribution (training data)",
                      xaxis_title="Monthly WTP (₹)", yaxis_title="Count")
    return _base(fig, 320)


def priority_donut(scored):
    vc = scored["pred_priority_tier"].value_counts().reset_index()
    vc.columns = ["Priority","Count"]
    colors = [PRIORITY_COLORS.get(p, GRAY) for p in vc["Priority"]]
    hot = scored[scored["pred_priority_tier"]=="Hot Lead"].shape[0]
    fig = go.Figure(go.Pie(labels=vc["Priority"], values=vc["Count"], hole=0.6,
        marker=dict(colors=colors, line=dict(color="white",width=2)),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} leads (%{percent})<extra></extra>"))
    fig.update_layout(title="Lead priority distribution",
        annotations=[dict(text=f"<b>{hot}</b><br>Hot Leads", x=0.5, y=0.5,
                          font=dict(size=14, color=RED), showarrow=False)])
    return _base(fig, 340, 44)


def quadrant_scatter(scored):
    fig = go.Figure()
    for priority, color in PRIORITY_COLORS.items():
        sub = scored[scored["pred_priority_tier"]==priority]
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub["pred_adoption_probability"],
            y=sub["pred_wtp_monthly_inr"], mode="markers", name=priority,
            marker=dict(color=color, size=7, opacity=0.7, line=dict(color="white",width=0.5)),
            hovertemplate=f"<b>{priority}</b><br>Prob: %{{x:.2f}}<br>WTP: ₹%{{y}}<extra></extra>"))
    fig.add_vline(x=0.65, line_dash="dot", line_color=NAVY, line_width=1.5,
                  annotation_text="  65% threshold", annotation_font=dict(size=9,color=NAVY))
    fig.add_hline(y=300, line_dash="dot", line_color=NAVY, line_width=1.5,
                  annotation_text="₹300 high WTP", annotation_font=dict(size=9,color=NAVY))
    fig.add_annotation(x=0.83, y=700, text="<b>HOT LEADS</b>", showarrow=False,
        font=dict(size=11,color=RED), bgcolor="rgba(217,48,37,0.08)", bordercolor=RED, borderwidth=1)
    fig.add_annotation(x=0.83, y=100, text="<b>FREEMIUM</b>", showarrow=False,
        font=dict(size=11,color=TEAL), bgcolor="rgba(11,140,110,0.08)", bordercolor=TEAL, borderwidth=1)
    fig.update_layout(title="Lead quadrant map — adoption probability vs predicted WTP",
                      xaxis_title="Adoption probability", yaxis_title="Predicted WTP (₹)",
                      legend=dict(orientation="h", y=-0.2, xanchor="center", x=0.5))
    return _base(fig, 440)


def adoption_prob_dist(scored):
    fig = go.Figure(go.Histogram(x=scored["pred_adoption_probability"], nbinsx=20,
        marker_color=BLUE, opacity=0.85,
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>"))
    mean_p = scored["pred_adoption_probability"].mean()
    fig.add_vline(x=0.65, line_dash="dot", line_color=RED, line_width=2,
                  annotation_text="  Conversion threshold",
                  annotation_font=dict(size=10, color=RED))
    fig.add_vline(x=mean_p, line_dash="dash", line_color=TEAL, line_width=1.5,
                  annotation_text=f"  Mean {mean_p:.2f}",
                  annotation_font=dict(size=10, color=TEAL))
    fig.update_layout(title="Adoption probability distribution — new customers",
                      xaxis_title="Probability", yaxis_title="Count")
    return _base(fig, 300)


def wtp_pred_dist(scored):
    fig = go.Figure(go.Histogram(x=scored["pred_wtp_monthly_inr"], nbinsx=20,
        marker_color=TEAL, opacity=0.85,
        hovertemplate="WTP: ₹%{x}<br>Count: %{y}<extra></extra>"))
    mean_w = scored["pred_wtp_monthly_inr"].mean()
    fig.add_vline(x=300, line_dash="dot", line_color=RED, line_width=2,
                  annotation_text="  High WTP ₹300+",
                  annotation_font=dict(size=10, color=RED))
    fig.add_vline(x=mean_w, line_dash="dash", line_color=NAVY, line_width=1.5,
                  annotation_text=f"  Mean ₹{mean_w:.0f}",
                  annotation_font=dict(size=10, color=NAVY))
    fig.update_layout(title="Predicted WTP distribution — new customers",
                      xaxis_title="Monthly WTP (₹)", yaxis_title="Count")
    return _base(fig, 300)


# ── Aliases to keep app.py imports working unchanged ─────────────────────────
def arm_scatter(r):                    return arm_bubble(r)
def arm_top_rules(r, n=20):           return arm_top_bar(r, n)
def cluster_elbow(k):                  return cluster_elbow_fig(k)
def cluster_wtp_bar(k):               return cluster_profile_fig(k)
def roc_curve_plot(m):                return roc_curve_fig(m)
def confusion_matrix_plot(m):         return confusion_matrix_fig(m)
def feature_importance_plot(m, k="classifier", n=18): return feature_importance_fig(m, k, n)
def wtp_prediction_hist(s):           return wtp_pred_dist(s)
def adoption_prob_hist(s):            return adoption_prob_dist(s)
def scatter_wtp_vs_prob(s):           return quadrant_scatter(s)
def metrics_gauge(*a, **kw):          return go.Figure()

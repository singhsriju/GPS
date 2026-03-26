"""
CareerGPS Analytics Platform — MBA-Level Dashboard
Sriju | AI-Powered Career Guidance for Indian Students
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CareerGPS Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #1a2744 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 8px 12px; border-radius: 8px; 
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }

/* Main area */
.main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }
[data-testid="stAppViewContainer"] { background: #f8fafd; }

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 20px 22px;
    border: 1px solid #e8edf5;
    box-shadow: 0 2px 12px rgba(15,23,41,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(15,23,41,0.10); }
.kpi-label { font-size: 12px; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 700; color: #0f1729; line-height: 1.1; }
.kpi-sub { font-size: 12px; color: #6b7280; margin-top: 4px; }
.kpi-delta-pos { color: #10b981; font-size: 13px; font-weight: 600; }
.kpi-delta-neg { color: #ef4444; font-size: 13px; font-weight: 600; }

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 22px; font-weight: 700; color: #0f1729;
    margin: 28px 0 16px 0; padding-bottom: 10px;
    border-bottom: 2px solid #e8edf5;
}
.section-sub { font-size: 13px; color: #6b7280; margin-top: -12px; margin-bottom: 14px; }

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 100%);
    border-left: 4px solid #3b82f6; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0;
    font-size: 13.5px; color: #374151; line-height: 1.6;
}
.insight-box b { color: #1d4ed8; }
.insight-warn {
    background: linear-gradient(135deg, #fff7ed 0%, #fffbf5 100%);
    border-left: 4px solid #f59e0b;
}
.insight-warn b { color: #d97706; }
.insight-success {
    background: linear-gradient(135deg, #f0fdf4 0%, #fafffe 100%);
    border-left: 4px solid #10b981;
}
.insight-success b { color: #059669; }

/* Page title */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px; font-weight: 700; color: #0f1729; margin-bottom: 2px;
}
.page-tagline { font-size: 14px; color: #6b7280; margin-bottom: 22px; }

/* Table styling */
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th { background: #0f1729; color: white; padding: 10px 14px; text-align: left; font-weight: 500; }
.styled-table td { padding: 9px 14px; border-bottom: 1px solid #e8edf5; color: #374151; }
.styled-table tr:hover td { background: #f8fafd; }
.styled-table tr:nth-child(even) td { background: #fafbfc; }

/* Badge */
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-hot { background: #fee2e2; color: #dc2626; }
.badge-warm { background: #fef3c7; color: #d97706; }
.badge-cold { background: #e0f2fe; color: #0369a1; }
.badge-green { background: #d1fae5; color: #065f46; }

/* Divider */
hr { border: none; border-top: 1px solid #e8edf5; margin: 20px 0; }

/* Upload area */
.upload-zone { background: white; border: 2px dashed #c7d2e0; border-radius: 12px; padding: 30px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("career_survey_dataset.csv")
    df["family_income"] = df["family_income"].fillna("Unknown")
    income_order = ["Below ₹2L","₹2–5L","₹5–10L","₹10–20L","Above ₹20L","Unknown"]
    df["income_cat"] = pd.Categorical(df["family_income"], categories=income_order, ordered=True)
    clarity_order = ["Not clear at all","Somewhat unclear","Somewhat clear","Very clear"]
    df["clarity_cat"] = pd.Categorical(df["career_clarity"], categories=clarity_order, ordered=True)
    urgency_order = ["Within 3 months","3–6 months","6–12 months","1–2 years","More than 2 years"]
    df["urgency_cat"] = pd.Categorical(df["urgency"], categories=urgency_order, ordered=True)
    return df

@st.cache_data
def load_model_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json") as f:
            return json.load(f)
    return {}

# ─── Imports for charts ──────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#1e40af","#3b82f6","#60a5fa","#93c5fd","#bfdbfe","#dbeafe"]
ACCENT  = ["#0f1729","#1e40af","#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#06b6d4","#6366f1"]
PERSONA_COLORS = {
    "Confused Drifter":"#f59e0b","Anxious Achiever":"#3b82f6",
    "Focused Climber":"#10b981","Career Switcher":"#8b5cf6","Passive Explorer":"#94a3b8"
}

def chart_style(fig, height=360, bg="#ffffff", showlegend=True):
    fig.update_layout(
        height=height, plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(family="DM Sans", size=12, color="#374151"),
        margin=dict(l=30,r=20,t=40,b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
                    font=dict(size=11)) if showlegend else dict(visible=False),
        xaxis=dict(showgrid=False, linecolor="#e8edf5", linewidth=1),
        yaxis=dict(showgrid=True, gridcolor="#f0f4f9", linecolor="#e8edf5"),
    )
    return fig

def kpi(label, value, sub="", delta="", delta_type="pos"):
    delta_html = f'<div class="kpi-delta-{"pos" if delta_type=="pos" else "neg"}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {delta_html}
    </div>"""

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 CareerGPS")
    st.markdown("<div style='font-size:11px;color:#94a3b8;margin-bottom:20px;'>AI Career Platform · Analytics Hub</div>", unsafe_allow_html=True)
    
    page = st.radio("Navigation", [
        "🏠  Business Overview",
        "📊  Descriptive EDA",
        "🔬  Diagnostic Analysis",
        "🤖  Predictive Models",
        "🧭  Prescriptive Actions",
        "📤  New Customer Scoring"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#94a3b8;'>Dataset: 2,000 respondents<br>Last updated: Synthetic v2</div>", unsafe_allow_html=True)
    
    if os.path.exists("model_metrics.json"):
        st.markdown("<div style='margin-top:10px;'><span style='background:#d1fae5;color:#065f46;font-size:11px;padding:3px 10px;border-radius:20px;font-weight:600;'>✓ Models Trained</span></div>", unsafe_allow_html=True)
    else:
        if st.button("⚙️ Train Models", use_container_width=True):
            with st.spinner("Training all models…"):
                import train_models
                train_models.run()
            st.success("Done!")
            st.rerun()

df = load_data()
metrics = load_model_metrics()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BUSINESS OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Business Overview":
    st.markdown('<div class="page-title">CareerGPS — Business Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-tagline">AI-Powered Career Guidance Platform · India Market Analysis · 2,000 Student Survey</div>', unsafe_allow_html=True)

    # ── KPI Row 1
    total = len(df)
    adopters = (df["platform_adoption"]=="Yes").sum()
    adopt_rate = adopters/total*100
    avg_wtp = df["wtp_monthly"].mean()
    tam = 45_000_000
    confused = (df["career_clarity"]=="Not clear at all").sum()
    confused_pct = confused/total*100
    
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Total Respondents","2,000","Survey sample","", "pos"), unsafe_allow_html=True)
    c2.markdown(kpi("Platform Adoption Rate",f"{adopt_rate:.1f}%","Would use CareerGPS","▲ vs 28% industry avg","pos"), unsafe_allow_html=True)
    c3.markdown(kpi("Avg Monthly WTP",f"₹{avg_wtp:.0f}","Willingness to pay","", "pos"), unsafe_allow_html=True)
    c4.markdown(kpi("Confused Students",f"{confused_pct:.0f}%","'Not clear at all'","High unmet need","neg"), unsafe_allow_html=True)
    c5.markdown(kpi("Addressable Market","45M+","Class 9–PG India","₹12,000 Cr+ TAM","pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Business Problem & Solution
    col_l, col_r = st.columns([1.1, 0.9])
    with col_l:
        st.markdown('<div class="section-header">The Business Opportunity</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box insight-success">
        <b>Problem:</b> 63% of Indian students are confused or unclear about their career path. 
        Existing guidance is fragmented — offline counsellors (expensive), YouTube (unstructured), family advice (biased). 
        No scalable, AI-driven, India-specific platform exists at an affordable price point.
        </div>
        <div class="insight-box">
        <b>Solution — CareerGPS:</b> An AI-powered platform that combines psychometric assessment, 
        personalised roadmapping, live market intelligence, and 1-on-1 mentorship — 
        designed for Indian students from Class 9 to postgraduate level, with Tier-2/3 city reach.
        </div>
        <div class="insight-box insight-warn">
        <b>Market Timing:</b> India has 350M+ students under 25. NEP 2020 is pushing career awareness earlier. 
        Post-COVID digital adoption among students is at an all-time high. The window to establish category leadership is now.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Platform Feature Stack</div>', unsafe_allow_html=True)
        features = {
            "Feature": ["AI Career Roadmap","1-on-1 Mentorship","College Matchmaking","Entrance Exam Navigator",
                        "Scholarship Finder","Skill Assessment","Market Intelligence","Parent Dashboard"],
            "Demand %": [68,71,65,72,58,53,61,44],
            "Revenue Tier": ["Core","Premium","Core","Core","Freemium","Core","Premium","Add-on"]
        }
        feat_df = pd.DataFrame(features)
        fig = px.bar(feat_df, x="Demand %", y="Feature", orientation="h",
                     color="Demand %", color_continuous_scale=["#bfdbfe","#1e40af"],
                     text="Demand %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=310, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=20,r=40,t=10,b=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          font=dict(family="DM Sans", size=11),
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Persona Overview
    st.markdown('<div class="section-header">Student Persona Landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">5 distinct segments driving distinct product and pricing strategies</div>', unsafe_allow_html=True)

    persona_stats = df.groupby("persona_label").agg(
        Count=("respondent_id","count"),
        Avg_WTP=("wtp_monthly","mean"),
        Adoption_Rate=("platform_adoption", lambda x: (x=="Yes").mean()*100)
    ).reset_index()
    persona_stats["Share_pct"] = persona_stats["Count"]/len(df)*100
    persona_stats["Revenue_Index"] = (persona_stats["Avg_WTP"] * persona_stats["Adoption_Rate"]/100).round(0)

    pc1, pc2 = st.columns(2)
    with pc1:
        fig_donut = go.Figure(go.Pie(
            labels=persona_stats["persona_label"], values=persona_stats["Count"],
            hole=0.55, marker_colors=[PERSONA_COLORS[p] for p in persona_stats["persona_label"]],
            textinfo="label+percent", textfont_size=11
        ))
        fig_donut.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                                 paper_bgcolor="white", title="Segment Distribution",
                                 font=dict(family="DM Sans"), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with pc2:
        fig_scatter = px.scatter(persona_stats, x="Adoption_Rate", y="Avg_WTP",
                                  size="Count", color="persona_label",
                                  color_discrete_map=PERSONA_COLORS,
                                  text="persona_label",
                                  size_max=55, labels={"Adoption_Rate":"Adoption Rate (%)","Avg_WTP":"Avg Monthly WTP (₹)"})
        fig_scatter.update_traces(textposition="top center", textfont_size=9)
        fig_scatter.add_hline(y=persona_stats["Avg_WTP"].mean(), line_dash="dot", line_color="#94a3b8", annotation_text="Avg WTP")
        fig_scatter.add_vline(x=persona_stats["Adoption_Rate"].mean(), line_dash="dot", line_color="#94a3b8", annotation_text="Avg Adoption")
        fig_scatter.update_layout(height=300, showlegend=False, margin=dict(l=30,r=20,t=30,b=30),
                                   paper_bgcolor="white", plot_bgcolor="white",
                                   font=dict(family="DM Sans",size=11),
                                   title="Segment Value Matrix (Size = Volume)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Persona Table
    persona_stats_display = persona_stats.copy()
    persona_stats_display["Avg_WTP"] = persona_stats_display["Avg_WTP"].map("₹{:.0f}".format)
    persona_stats_display["Adoption_Rate"] = persona_stats_display["Adoption_Rate"].map("{:.1f}%".format)
    persona_stats_display["Share_pct"] = persona_stats_display["Share_pct"].map("{:.1f}%".format)
    persona_stats_display["Revenue_Index"] = persona_stats_display["Revenue_Index"].map("₹{:.0f}".format)
    persona_stats_display.columns = ["Persona","Count","Avg Monthly WTP","Adoption Rate","Market Share","Revenue Index (WTP×Adoption)"]

    rows_html = "".join([
        f"<tr><td><b>{r['Persona']}</b></td><td>{r['Count']}</td><td>{r['Avg Monthly WTP']}</td>"
        f"<td>{r['Adoption Rate']}</td><td>{r['Market Share']}</td><td>{r['Revenue Index (WTP×Adoption)']}</td></tr>"
        for _, r in persona_stats_display.iterrows()
    ])
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>{''.join(f'<th>{c}</th>' for c in persona_stats_display.columns)}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Revenue Model Snapshot</div>', unsafe_allow_html=True)

    r1,r2,r3,r4 = st.columns(4)
    r1.markdown(kpi("Freemium Users","~60%","Free psychometric test","Funnel entry","pos"), unsafe_allow_html=True)
    r2.markdown(kpi("Core Plan","₹299/mo","Roadmap + AI chatbot","Primary revenue","pos"), unsafe_allow_html=True)
    r3.markdown(kpi("Premium Plan","₹799/mo","1-on-1 mentorship","High-LTV segment","pos"), unsafe_allow_html=True)
    r4.markdown(kpi("B2B / Schools","₹50K/yr","Per institution","Fastest scale","pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box insight-success">
    <b>Primary Target Segment (Next 6 Months):</b> Anxious Achievers in Tier-2 cities · Class 11–12 · Science/Commerce stream · 
    Income ₹5–20L · Decision urgency within 6 months. 
    Rationale: High adoption rate (52%), strong WTP (₹340/mo), large volume (25% of market), parental financial support. 
    Lowest CAC-to-LTV ratio of any segment.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Descriptive EDA":
    st.markdown('<div class="page-title">Descriptive Analytics — EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-tagline">Who are our 2,000 survey respondents? Distribution, profile, and market structure.</div>', unsafe_allow_html=True)

    # ── KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(kpi("Mean Age",f"{df['age'].mean():.1f} yrs","Range: 15–25","","pos"), unsafe_allow_html=True)
    k2.markdown(kpi("Top State","Uttar Pradesh",f"{(df['state']=='Uttar Pradesh').sum()} resp","Largest segment","pos"), unsafe_allow_html=True)
    k3.markdown(kpi("Top Stream","Science PCM",f"{(df['stream']=='Science PCM').sum()} resp","28% of sample","pos"), unsafe_allow_html=True)
    k4.markdown(kpi("Tier-2 Students",f"{(df['location_type']=='Tier-2').sum()}",f"{(df['location_type']=='Tier-2').mean()*100:.0f}% of sample","Primary market","pos"), unsafe_allow_html=True)
    k5.markdown(kpi("WTP Median",f"₹{df['wtp_monthly'].median():.0f}/mo","50th percentile","","pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Age + Gender
    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)

    with d1:
        age_counts = df["age"].value_counts().sort_index()
        fig = px.bar(x=age_counts.index, y=age_counts.values, labels={"x":"Age","y":"Count"},
                     color=age_counts.values, color_continuous_scale=["#bfdbfe","#1e40af"], title="Age Distribution")
        fig.update_layout(height=290, showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=40,b=10),
                           font=dict(family="DM Sans",size=11))
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        gc = df["gender"].value_counts()
        fig = go.Figure(go.Pie(labels=gc.index, values=gc.values, hole=0.5,
                                marker_colors=["#1e40af","#f472b6","#94a3b8"]))
        fig.update_layout(height=290, paper_bgcolor="white", margin=dict(l=10,r=10,t=40,b=30),
                           font=dict(family="DM Sans",size=11), title="Gender Split",
                           legend=dict(orientation="h",y=-0.1,x=0.5,xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    with d3:
        lc = df["location_type"].value_counts()
        colors = {"Metro":"#1e40af","Tier-2":"#3b82f6","Tier-3":"#93c5fd","Rural":"#dbeafe"}
        fig = go.Figure(go.Bar(x=lc.index, y=lc.values, marker_color=[colors.get(l,"#60a5fa") for l in lc.index],
                                text=lc.values, textposition="outside"))
        fig.update_layout(height=290, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                           title="Location Type", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor="#f0f4f9"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box"><b>Demographic Insight:</b> Peak age is 17–19, aligning with Class 11–12 and first-year undergraduate — 
    the highest-urgency decision window in a student's academic life. Tier-2 cities represent 35% of respondents, 
    confirming the under-served Tier-2 market as the primary growth opportunity.</div>
    """, unsafe_allow_html=True)

    # ── Education & Stream
    st.markdown('<div class="section-header">Education Profile</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)

    with e1:
        ec = df["education_level"].value_counts()
        fig = px.bar(x=ec.values, y=ec.index, orientation="h", title="Education Level",
                     color=ec.values, color_continuous_scale=["#bfdbfe","#1e40af"], text=ec.values)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=290, showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=40,t=40,b=10),
                           font=dict(family="DM Sans",size=11), yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    with e2:
        sc = df["stream"].value_counts()
        fig = px.pie(values=sc.values, names=sc.index, hole=0.4, title="Academic Stream",
                     color_discrete_sequence=ACCENT)
        fig.update_layout(height=290, paper_bgcolor="white", margin=dict(l=10,r=10,t=40,b=30),
                           font=dict(family="DM Sans",size=11),
                           legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center",font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    with e3:
        pc = df["academic_performance"].value_counts()
        order = ["Below 60%","60–70%","70–80%","80–90%","Above 90%"]
        pc = pc.reindex([o for o in order if o in pc.index])
        colors_perf = ["#fee2e2","#fef3c7","#dbeafe","#d1fae5","#a7f3d0"]
        fig = go.Figure(go.Bar(x=pc.index, y=pc.values, marker_color=colors_perf[:len(pc)],
                                text=pc.values, textposition="outside"))
        fig.update_layout(height=290, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=40,b=10), title="Academic Performance",
                           font=dict(family="DM Sans",size=11),
                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor="#f0f4f9"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Income Distribution
    st.markdown('<div class="section-header">Family Income Distribution</div>', unsafe_allow_html=True)
    income_order = ["Below ₹2L","₹2–5L","₹5–10L","₹10–20L","Above ₹20L","Unknown"]
    ic = df["family_income"].value_counts().reindex([o for o in income_order if o in df["family_income"].values])
    ic_pct = ic / ic.sum() * 100
    ic_cum = ic_pct.cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ic.index, y=ic.values, name="Count",
                          marker_color=["#fee2e2","#fef3c7","#bfdbfe","#93c5fd","#1e40af","#e2e8f0"],
                          text=ic.values, textposition="outside"), secondary_y=False)
    fig.add_trace(go.Scatter(x=ic.index, y=ic_cum.values, name="Cumulative %",
                              mode="lines+markers", line=dict(color="#f59e0b",width=2.5),
                              marker=dict(size=8)), secondary_y=True)
    fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                       title="Income Bands with Cumulative Distribution",
                       legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center"))
    fig.update_yaxes(title_text="Count", secondary_y=False, showgrid=True, gridcolor="#f0f4f9")
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True, showgrid=False, range=[0,105])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box"><b>Income Insight:</b> The ₹5–10L income band is the largest segment (30%), 
    and 55% of respondents fall in ₹5–20L — the sweet spot for ₹299–₹799/mo pricing. 
    Only 10% are above ₹20L, confirming a mass-market rather than premium-only positioning.</div>
    """, unsafe_allow_html=True)

    # ── Career Clarity
    st.markdown('<div class="section-header">Career Clarity & Urgency</div>', unsafe_allow_html=True)
    cl1, cl2, cl3 = st.columns(3)

    with cl1:
        cc = df["career_clarity"].value_counts()
        order = ["Not clear at all","Somewhat unclear","Somewhat clear","Very clear"]
        cc = cc.reindex([o for o in order if o in cc.index])
        cols_clarity = ["#ef4444","#f59e0b","#60a5fa","#10b981"]
        fig = go.Figure(go.Bar(x=cc.values, y=cc.index, orientation="h",
                                marker_color=cols_clarity[:len(cc)], text=cc.values, textposition="outside"))
        fig.update_layout(height=260, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=40,t=40,b=10), font=dict(family="DM Sans",size=11),
                           title="Career Clarity Level", yaxis=dict(categoryorder="array", categoryarray=order))
        st.plotly_chart(fig, use_container_width=True)

    with cl2:
        uc = df["urgency"].value_counts()
        order_u = ["Within 3 months","3–6 months","6–12 months","1–2 years","More than 2 years"]
        uc = uc.reindex([o for o in order_u if o in uc.index])
        fig = px.funnel(y=uc.index, x=uc.values, title="Decision Urgency Funnel",
                         color_discrete_sequence=["#1e40af"])
        fig.update_layout(height=260, paper_bgcolor="white", margin=dict(l=10,r=10,t=40,b=10),
                           font=dict(family="DM Sans",size=11))
        st.plotly_chart(fig, use_container_width=True)

    with cl3:
        dm = df["decision_maker"].value_counts()
        fig = px.pie(values=dm.values, names=dm.index, hole=0.45, title="Who Decides Career Choice?",
                     color_discrete_sequence=ACCENT)
        fig.update_layout(height=260, paper_bgcolor="white", margin=dict(l=10,r=10,t=40,b=30),
                           font=dict(family="DM Sans",size=11),
                           legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center",font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box insight-warn"><b>Clarity Gap:</b> 63% of respondents are "Not clear" or "Somewhat unclear" — 
    this is the core product-market fit signal. <b>Decision maker data</b> is critical for India: 
    35% say "Both equally" (student + parent), 30% say parents alone — meaning marketing must reach parents on WhatsApp/Facebook, 
    not just students on Instagram.</div>
    """, unsafe_allow_html=True)

    # ── WTP Distribution
    st.markdown('<div class="section-header">Willingness to Pay (WTP) Analysis</div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)

    with w1:
        fig = px.histogram(df, x="wtp_monthly", nbins=30, title="Monthly WTP Distribution",
                            color_discrete_sequence=["#3b82f6"])
        fig.add_vline(x=df["wtp_monthly"].mean(), line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Mean ₹{df['wtp_monthly'].mean():.0f}")
        fig.add_vline(x=df["wtp_monthly"].median(), line_dash="dot", line_color="#10b981",
                       annotation_text=f"Median ₹{df['wtp_monthly'].median():.0f}")
        fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                           showlegend=False, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with w2:
        wtp_loc = df.groupby("location_type")["wtp_monthly"].agg(["mean","median","std"]).reset_index()
        wtp_loc.columns = ["Location","Mean","Median","Std"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mean", x=wtp_loc["Location"], y=wtp_loc["Mean"],
                              marker_color="#1e40af", text=wtp_loc["Mean"].map("₹{:.0f}".format), textposition="outside"))
        fig.add_trace(go.Bar(name="Median", x=wtp_loc["Location"], y=wtp_loc["Median"],
                              marker_color="#93c5fd", text=wtp_loc["Median"].map("₹{:.0f}".format), textposition="outside"))
        fig.update_layout(height=280, barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                           title="WTP by Location", legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    with w3:
        wtp_p = df.groupby("persona_label")["wtp_monthly"].mean().sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=wtp_p.values, y=wtp_p.index, orientation="h",
                                marker_color=[PERSONA_COLORS[p] for p in wtp_p.index],
                                text=wtp_p.values.astype(int), textposition="outside",
                                texttemplate="₹%{text}"))
        fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=60,t=40,b=10), font=dict(family="DM Sans",size=11),
                           title="Avg WTP by Persona")
        st.plotly_chart(fig, use_container_width=True)

    # WTP Stats Table
    st.markdown('<div class="section-sub">WTP Summary Statistics — mirroring Excel EDA Sheet 4</div>', unsafe_allow_html=True)
    wtp_stats = df["wtp_monthly"].describe().to_frame().T
    wtp_stats.columns = ["Count","Mean","Std Dev","Min","25th %ile","Median","75th %ile","Max"]
    wtp_stats = wtp_stats.applymap(lambda x: f"₹{x:.0f}" if isinstance(x,float) else str(int(x)))
    wtp_stats.at[0,"Count"] = "2,000"
    stat_html = "".join([f"<td>{wtp_stats.iloc[0][c]}</td>" for c in wtp_stats.columns])
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>{''.join(f'<th>{c}</th>' for c in wtp_stats.columns)}</tr></thead>
      <tbody><tr>{stat_html}</tr></tbody>
    </table><br>""", unsafe_allow_html=True)

    # ── State Analysis
    st.markdown('<div class="section-header">Geographic Distribution (Top 10 States)</div>', unsafe_allow_html=True)
    state_df = df.groupby("state").agg(
        Count=("respondent_id","count"),
        Avg_WTP=("wtp_monthly","mean"),
        Adopt_Rate=("platform_adoption", lambda x:(x=="Yes").mean()*100)
    ).reset_index().sort_values("Count", ascending=False).head(10)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=("Respondents by State","Avg Monthly WTP by State (₹)"))
    fig.add_trace(go.Bar(y=state_df["state"], x=state_df["Count"], orientation="h",
                          marker_color="#1e40af", name="Count",
                          text=state_df["Count"], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(y=state_df["state"], x=state_df["Avg_WTP"].round(0), orientation="h",
                          marker_color="#10b981", name="WTP",
                          text=state_df["Avg_WTP"].map("₹{:.0f}".format), textposition="outside"), row=1, col=2)
    fig.update_layout(height=340, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=10,r=40,t=40,b=10), font=dict(family="DM Sans",size=11))
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box"><b>Geographic Insight:</b> UP dominates in volume (18%) but WTP is below average — 
    suggesting a freemium entry strategy. Maharashtra and Karnataka have both high volume and above-average WTP, 
    making them ideal launch markets for the paid tier. Tamil Nadu shows the highest willingness to pay index relative to volume.</div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Diagnostic Analysis":
    st.markdown('<div class="page-title">Diagnostic Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-tagline">Why do students behave the way they do? Correlation, clustering, and association patterns.</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📐 Correlation Analysis","🔗 Association Rules","👥 Clustering","📊 Cross-tabs"])

    # ── TAB 1: Correlations
    with tabs[0]:
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)

        num_df = df[["age","fear_score","risk_score","wtp_monthly","wtp_onetime"]].copy()
        num_df["adoption_binary"] = (df["platform_adoption"]=="Yes").astype(int)
        num_df["urgency_score"] = df["urgency"].map({"Within 3 months":5,"3–6 months":4,"6–12 months":3,"1–2 years":2,"More than 2 years":1})
        num_df["clarity_score"] = df["career_clarity"].map({"Not clear at all":1,"Somewhat unclear":2,"Somewhat clear":3,"Very clear":4})
        
        corr = num_df.corr().round(2)
        
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Pearson Correlation Matrix")
        fig.update_layout(height=420, paper_bgcolor="white", font=dict(family="DM Sans",size=11),
                           margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box"><b>Key Correlation Findings:</b><br>
        • <b>WTP_monthly ↔ WTP_onetime (r≈+0.90)</b>: Strong — students with higher monthly WTP also accept higher one-time pricing. 
        Bundle pricing is viable.<br>
        • <b>Fear ↔ Risk (r≈−0.47)</b>: As designed — fearful students are risk-averse. 
        Different messaging required (reassurance vs. aspiration).<br>
        • <b>Urgency ↔ Adoption (r≈+0.35)</b>: Students with imminent decisions are significantly more likely to convert. 
        Time-triggered marketing campaigns will outperform generic ones.<br>
        • <b>Clarity ↔ Adoption (r≈+0.28)</b>: Slightly clearer students convert better — but confused students 
        are the largest segment. Low-friction entry point (free test) is essential.</div>
        """, unsafe_allow_html=True)

        # Pairwise table
        st.markdown('<div class="section-sub" style="margin-top:20px">Pairwise Correlations with Adoption (Business View)</div>', unsafe_allow_html=True)
        pairs = [(col, corr.loc[col, "adoption_binary"]) for col in corr.columns if col != "adoption_binary"]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        prows = "".join([f"<tr><td>{p}</td><td><b>{v:.3f}</b></td><td>{'Strong' if abs(v)>0.4 else 'Moderate' if abs(v)>0.2 else 'Weak'}</td></tr>" for p,v in pairs])
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Variable</th><th>Correlation with Adoption</th><th>Strength</th></tr></thead>
          <tbody>{prows}</tbody>
        </table>""", unsafe_allow_html=True)

    # ── TAB 2: ARM
    with tabs[1]:
        st.markdown('<div class="section-header">Association Rule Mining — Interest → Career Patterns</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box insert-warn">Association rules reveal natural co-occurrence patterns 
        between stream, domain interest, features, and career choices. 
        These rules directly feed the platform's recommendation engine.</div>
        """, unsafe_allow_html=True)

        # Build rules manually from crosstabs (mlxtend not available)
        rules_data = []
        streams = df["stream"].unique()
        domains = df["career_domain_interest"].unique()
        
        for s in streams:
            for d in domains:
                subset = df[df["stream"]==s]
                if len(subset) < 20: continue
                support = len(df[(df["stream"]==s) & (df["career_domain_interest"]==d)]) / len(df)
                conf = len(df[(df["stream"]==s) & (df["career_domain_interest"]==d)]) / len(subset) if len(subset)>0 else 0
                d_base = len(df[df["career_domain_interest"]==d]) / len(df)
                lift = conf / d_base if d_base > 0 else 1
                if support > 0.02 and conf > 0.15 and lift > 1.2:
                    rules_data.append({"Antecedent (Stream)": s, "Consequent (Career Domain)": d,
                                       "Support": round(support,3), "Confidence": round(conf,3), "Lift": round(lift,2)})

        rules_df = pd.DataFrame(rules_data).sort_values("Lift", ascending=False).head(15)

        if not rules_df.empty:
            fig_rules = px.scatter(rules_df, x="Support", y="Confidence", size="Lift", color="Lift",
                                    color_continuous_scale="Blues", hover_data=["Antecedent (Stream)","Consequent (Career Domain)"],
                                    text="Consequent (Career Domain)", size_max=40,
                                    title="Association Rules: Support vs Confidence (Size = Lift)")
            fig_rules.update_traces(textposition="top center", textfont_size=9)
            fig_rules.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                                     margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11))
            st.plotly_chart(fig_rules, use_container_width=True)

            rules_display = rules_df.copy()
            rules_display["Support"] = rules_display["Support"].map("{:.1%}".format)
            rules_display["Confidence"] = rules_display["Confidence"].map("{:.1%}".format)
            rrows = "".join([
                f"<tr><td>{r['Antecedent (Stream)']}</td><td>{r['Consequent (Career Domain)']}</td>"
                f"<td>{r['Support']}</td><td>{r['Confidence']}</td><td><b>{r['Lift']}</b></td></tr>"
                for _,r in rules_display.iterrows()
            ])
            st.markdown(f"""
            <table class="styled-table">
              <thead><tr><th>Antecedent (Stream)</th><th>Consequent (Career Domain)</th><th>Support</th><th>Confidence</th><th>Lift ↑</th></tr></thead>
              <tbody>{rrows}</tbody>
            </table>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="insight-box insight-success" style="margin-top:12px"><b>Top Rule:</b> Science PCB → Medicine/Healthcare 
            has the highest lift, confirming strong natural association. 
            Science PCM → Engineering/Tech is the second strongest. 
            Commerce → Business/Management and Finance/Banking are both high-confidence rules. 
            These patterns should auto-populate career suggestions in the roadmap engine.</div>
            """, unsafe_allow_html=True)

    # ── TAB 3: Clustering
    with tabs[2]:
        st.markdown('<div class="section-header">Persona Clustering Analysis</div>', unsafe_allow_html=True)

        # Show WTP distribution per persona
        fig_box = px.box(df, x="persona_label", y="wtp_monthly",
                          color="persona_label", color_discrete_map=PERSONA_COLORS,
                          title="WTP Distribution per Persona Cluster",
                          labels={"persona_label":"Persona","wtp_monthly":"Monthly WTP (₹)"})
        fig_box.update_layout(height=360, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11))
        st.plotly_chart(fig_box, use_container_width=True)

        # Persona profiler
        persona_profile = df.groupby("persona_label").agg(
            n=("respondent_id","count"),
            avg_wtp=("wtp_monthly","mean"),
            adopt_rate=("platform_adoption", lambda x:(x=="Yes").mean()*100),
            avg_fear=("fear_score","mean"),
            avg_risk=("risk_score","mean"),
            pct_urgent=("urgency", lambda x:(x.isin(["Within 3 months","3–6 months"])).mean()*100),
            pct_past_spend=("past_career_spend", lambda x:(x!="Nothing").mean()*100)
        ).reset_index().round(1)

        prows = "".join([
            f"<tr><td><b>{r['persona_label']}</b></td><td>{int(r['n'])}</td>"
            f"<td>₹{r['avg_wtp']:.0f}</td><td>{r['adopt_rate']:.1f}%</td>"
            f"<td>{r['avg_fear']:.1f}/5</td><td>{r['avg_risk']:.1f}/5</td>"
            f"<td>{r['pct_urgent']:.0f}%</td><td>{r['pct_past_spend']:.0f}%</td></tr>"
            for _,r in persona_profile.iterrows()
        ])
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Persona</th><th>N</th><th>Avg WTP</th><th>Adoption Rate</th>
          <th>Fear Score</th><th>Risk Score</th><th>% Urgent</th><th>% Past Spender</th></tr></thead>
          <tbody>{prows}</tbody>
        </table>""", unsafe_allow_html=True)

    # ── TAB 4: Cross-tabs
    with tabs[3]:
        st.markdown('<div class="section-header">Cross-Tabulation Analysis</div>', unsafe_allow_html=True)
        
        ct1, ct2 = st.columns(2)
        with ct1:
            ct = pd.crosstab(df["stream"], df["platform_adoption"], normalize="index").round(3)*100
            fig = px.imshow(ct, text_auto=".0f", color_continuous_scale="Blues",
                             title="Adoption Rate by Stream (%)", aspect="auto")
            fig.update_layout(height=320, paper_bgcolor="white", font=dict(family="DM Sans",size=11),
                               margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with ct2:
            ct2_df = pd.crosstab(df["location_type"], df["career_clarity"])
            ct2_pct = ct2_df.div(ct2_df.sum(axis=1), axis=0).round(3)*100
            fig = px.imshow(ct2_pct, text_auto=".0f", color_continuous_scale="RdYlGn",
                             title="Career Clarity by Location (%)", aspect="auto")
            fig.update_layout(height=320, paper_bgcolor="white", font=dict(family="DM Sans",size=11),
                               margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # WTP by Income x Location
        wtp_pivot = df.groupby(["location_type","family_income"])["wtp_monthly"].mean().unstack(fill_value=0).round(0)
        fig = px.imshow(wtp_pivot, text_auto=".0f", color_continuous_scale="Blues",
                         title="Mean Monthly WTP (₹) — Location × Income Band", aspect="auto")
        fig.update_layout(height=300, paper_bgcolor="white", font=dict(family="DM Sans",size=11),
                           margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Predictive Models":
    st.markdown('<div class="page-title">Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-tagline">ML models to predict adoption likelihood and willingness to pay for every new respondent.</div>', unsafe_allow_html=True)

    if not os.path.exists("model_metrics.json"):
        st.warning("⚙️ Models not yet trained. Click **Train Models** in the sidebar first.")
    else:
        metrics = load_model_metrics()

        # Classification metrics
        st.markdown('<div class="section-header">Classification Model — Platform Adoption Prediction</div>', unsafe_allow_html=True)
        clf = metrics.get("classifier", {})
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.markdown(kpi("Accuracy",f"{clf.get('accuracy',0)*100:.1f}%","Overall correctness","","pos"), unsafe_allow_html=True)
        m2.markdown(kpi("Precision",f"{clf.get('precision',0)*100:.1f}%","Of predicted Yes, % correct","","pos"), unsafe_allow_html=True)
        m3.markdown(kpi("Recall",f"{clf.get('recall',0)*100:.1f}%","% of actual Yes caught","","pos"), unsafe_allow_html=True)
        m4.markdown(kpi("F1-Score",f"{clf.get('f1',0)*100:.1f}%","Harmonic mean","","pos"), unsafe_allow_html=True)
        m5.markdown(kpi("ROC-AUC",f"{clf.get('roc_auc',0):.3f}","Discrimination power","","pos"), unsafe_allow_html=True)

        # ROC Curve
        roc_data = clf.get("roc_curve", {})
        if roc_data:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_data.get("fpr",[]), y=roc_data.get("tpr",[]),
                                          mode="lines", name=f"ROC (AUC={clf.get('roc_auc',0):.3f})",
                                          line=dict(color="#1e40af",width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                          line=dict(color="#94a3b8",dash="dash")))
            fig_roc.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                                   title="ROC Curve — Adoption Classifier",
                                   xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                   legend=dict(x=0.6,y=0.1))
            
            fi_data = clf.get("feature_importance", {})
            if fi_data:
                fi_df = pd.DataFrame({"Feature":list(fi_data.keys()), "Importance":list(fi_data.values())})
                fi_df = fi_df.sort_values("Importance",ascending=True).tail(12)
                fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                                           marker_color="#1e40af",
                                           text=fi_df["Importance"].map("{:.3f}".format),
                                           textposition="outside"))
                fig_fi.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                                      margin=dict(l=10,r=60,t=40,b=10), font=dict(family="DM Sans",size=11),
                                      title="Top Feature Importances")
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(fig_roc, use_container_width=True)
                with c2: st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrix
        cm_data = clf.get("confusion_matrix", [])
        if cm_data:
            st.markdown('<div class="section-sub">Confusion Matrix</div>', unsafe_allow_html=True)
            cm_arr = np.array(cm_data)
            fig_cm = px.imshow(cm_arr, text_auto=True, color_continuous_scale="Blues",
                                x=["Pred: No","Pred: Yes"], y=["Actual: No","Actual: Yes"],
                                title="Confusion Matrix")
            fig_cm.update_layout(height=300, paper_bgcolor="white", font=dict(family="DM Sans",size=12),
                                  margin=dict(l=10,r=10,t=40,b=10))
            col_cm, _ = st.columns([0.4,0.6])
            with col_cm: st.plotly_chart(fig_cm, use_container_width=True)

        # Regression
        st.markdown('<div class="section-header">Regression Model — Monthly WTP Prediction</div>', unsafe_allow_html=True)
        reg = metrics.get("regressor",{})
        r1,r2,r3,r4 = st.columns(4)
        r1.markdown(kpi("MAE",f"₹{reg.get('mae',0):.0f}","Mean Absolute Error","","pos"), unsafe_allow_html=True)
        r2.markdown(kpi("RMSE",f"₹{reg.get('rmse',0):.0f}","Root Mean Squared Error","","pos"), unsafe_allow_html=True)
        r3.markdown(kpi("R² Score",f"{reg.get('r2',0):.3f}","Variance explained","","pos"), unsafe_allow_html=True)
        r4.markdown(kpi("Median Error",f"₹{reg.get('median_ae',0):.0f}","50th pctile error","","pos"), unsafe_allow_html=True)

        # Actual vs Predicted
        pred_data = reg.get("pred_vs_actual", {})
        if pred_data:
            fig_pva = go.Figure()
            fig_pva.add_trace(go.Scatter(x=pred_data.get("actual",[]), y=pred_data.get("predicted",[]),
                                          mode="markers", marker=dict(color="#3b82f6",size=4,opacity=0.5), name="Predictions"))
            max_val = max(max(pred_data.get("actual",[1])), max(pred_data.get("predicted",[1])))
            fig_pva.add_trace(go.Scatter(x=[0,max_val],y=[0,max_val],mode="lines",
                                          line=dict(color="#ef4444",dash="dash"), name="Perfect Fit"))
            fig_pva.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                                   margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                                   title="Predicted vs Actual WTP (₹)",
                                   xaxis_title="Actual WTP", yaxis_title="Predicted WTP",
                                   legend=dict(x=0.6,y=0.1))
            
            fi_reg = reg.get("feature_importance",{})
            if fi_reg:
                fi_rdf = pd.DataFrame({"Feature":list(fi_reg.keys()), "Importance":list(fi_reg.values())})
                fi_rdf = fi_rdf.sort_values("Importance",ascending=True).tail(10)
                fig_fi2 = go.Figure(go.Bar(x=fi_rdf["Importance"], y=fi_rdf["Feature"], orientation="h",
                                            marker_color="#10b981",
                                            text=fi_rdf["Importance"].map("{:.3f}".format), textposition="outside"))
                fig_fi2.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                                       margin=dict(l=10,r=60,t=40,b=10), font=dict(family="DM Sans",size=11),
                                       title="WTP Feature Importances")
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(fig_pva, use_container_width=True)
                with c2: st.plotly_chart(fig_fi2, use_container_width=True)
            else:
                st.plotly_chart(fig_pva, use_container_width=True)

        st.markdown("""
        <div class="insight-box"><b>Model Insight:</b> High metrics on synthetic data are expected 
        (clean correlations). On real survey data, expect Accuracy ~78–85% and R² ~0.65–0.80. 
        The feature importance rankings are the most actionable output — <b>urgency, past spending, and persona</b> 
        are consistently the top three predictors of both adoption and WTP across both models.</div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRESCRIPTIVE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧭  Prescriptive Actions":
    st.markdown('<div class="page-title">Prescriptive Analytics — Action Playbooks</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-tagline">Convert insights into concrete actions. Segment-level marketing, pricing, and product playbooks.</div>', unsafe_allow_html=True)

    # Priority segments
    st.markdown('<div class="section-header">Segment Prioritisation Matrix</div>', unsafe_allow_html=True)
    
    seg_data = df.groupby("persona_label").agg(
        Volume=("respondent_id","count"),
        WTP=("wtp_monthly","mean"),
        Adoption_Rate=("platform_adoption", lambda x:(x=="Yes").mean()*100),
        Urgency_Pct=("urgency", lambda x:(x.isin(["Within 3 months","3–6 months"])).mean()*100)
    ).reset_index()
    seg_data["Revenue_Score"] = (seg_data["WTP"]/seg_data["WTP"].max()*0.35 +
                                  seg_data["Adoption_Rate"]/100*0.35 +
                                  seg_data["Urgency_Pct"]/100*0.30) * 100
    seg_data["Priority"] = pd.qcut(seg_data["Revenue_Score"], q=3, labels=["Low","Medium","High"])

    fig = px.scatter(seg_data, x="Adoption_Rate", y="WTP", size="Volume",
                     color="Revenue_Score", text="persona_label",
                     color_continuous_scale="Blues", size_max=60,
                     labels={"Adoption_Rate":"Adoption Rate (%)","WTP":"Avg WTP (₹)","Revenue_Score":"Revenue Score"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                       title="Segment Revenue Score Matrix (Bubble = Volume)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Persona-Level Action Playbooks</div>', unsafe_allow_html=True)

    playbooks = {
        "Focused Climber": {
            "priority": "HOT","badge": "badge-hot",
            "wtp": "₹399–₹799/mo", "channel": "LinkedIn, YouTube, Direct Search",
            "product": "Full roadmap + College matcher + Premium mentorship",
            "pricing": "Standard (₹299) or Premium (₹799) — skip freemium",
            "message": "Aspiration-led: 'Get to your dream college 2x faster'",
            "action": "Activate immediately. Offer 7-day free trial of Premium."
        },
        "Anxious Achiever": {
            "priority": "HOT","badge": "badge-hot",
            "wtp": "₹299–₹499/mo", "channel": "Instagram Stories, WhatsApp parent groups, Google Search",
            "product": "AI roadmap + Exam navigator + Parent dashboard",
            "pricing": "Standard plan ₹299 with parent add-on ₹99",
            "message": "Reassurance-led: 'Don't let confusion cost you a year'",
            "action": "Run urgency-triggered campaigns. Free psychometric test CTA."
        },
        "Career Switcher": {
            "priority": "WARM","badge": "badge-warm",
            "wtp": "₹350–₹600/mo", "channel": "LinkedIn, Quora, YouTube",
            "product": "1-on-1 mentorship + Market intelligence + Skill gap analysis",
            "pricing": "Premium ₹799 — high WTP, outcome-focused",
            "message": "Outcome-led: 'Transition with a clear plan, not a leap of faith'",
            "action": "Target with re-engagement content. Long-form webinars."
        },
        "Passive Explorer": {
            "priority": "WARM","badge": "badge-warm",
            "wtp": "₹100–₹250/mo", "channel": "Instagram Reels, YouTube Shorts",
            "product": "Free psychometric test → freemium funnel",
            "pricing": "Freemium entry, upsell at 30-day mark",
            "message": "Curiosity-led: 'Discover what career is actually right for you'",
            "action": "Viral content seeding. Referral incentives."
        },
        "Confused Drifter": {
            "priority": "COLD","badge": "badge-cold",
            "wtp": "₹100–₹200/mo", "channel": "YouTube, family WhatsApp, school partnerships",
            "product": "Free assessment only — convert over 60–90 days",
            "pricing": "Freemium. Do NOT pitch paid upfront.",
            "message": "Empathy-led: 'You're not alone — 6 in 10 students feel the same way'",
            "action": "B2B school channel. Low-cost brand awareness."
        }
    }

    for persona, pb in playbooks.items():
        with st.expander(f"{'🔴' if pb['priority']=='HOT' else '🟡' if pb['priority']=='WARM' else '🔵'} {persona} — Priority: {pb['priority']}", expanded=(pb['priority']=='HOT')):
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**💰 WTP Range**<br>{pb['wtp']}", unsafe_allow_html=True)
            col2.markdown(f"**📣 Best Channels**<br>{pb['channel']}", unsafe_allow_html=True)
            col3.markdown(f"**🎯 Pricing Strategy**<br>{pb['pricing']}", unsafe_allow_html=True)
            st.markdown(f"**🛒 Product Offer:** {pb['product']}")
            st.markdown(f"**💬 Message Tone:** {pb['message']}")
            st.markdown(f"**✅ Immediate Action:** {pb['action']}")

    st.markdown("---")
    st.markdown('<div class="section-header">Revenue Projections (Conservative Scenario)</div>', unsafe_allow_html=True)

    months = list(range(1,13))
    users_base = [50,120,280,450,620,800,1000,1200,1400,1600,1900,2200]
    rev = [u * 350 for u in users_base]

    fig_rev = make_subplots(specs=[[{"secondary_y":True}]])
    fig_rev.add_trace(go.Bar(x=months, y=users_base, name="Paid Users", marker_color="#bfdbfe"), secondary_y=False)
    fig_rev.add_trace(go.Scatter(x=months, y=rev, name="Monthly Revenue (₹)", mode="lines+markers",
                                  line=dict(color="#1e40af",width=2.5), marker=dict(size=8)), secondary_y=True)
    fig_rev.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=40,b=10), font=dict(family="DM Sans",size=11),
                           title="12-Month Revenue Trajectory",
                           legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center"))
    fig_rev.update_yaxes(title_text="Paid Users", secondary_y=False)
    fig_rev.update_yaxes(title_text="Monthly Revenue (₹)", secondary_y=True)
    st.plotly_chart(fig_rev, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — NEW CUSTOMER SCORING
# ════════════════════════════════════════════════════════════════════════════
elif page == "📤  New Customer Scoring":
    st.markdown('<div class="page-title">New Customer Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-tagline">Upload new survey responses to instantly score adoption likelihood and predicted WTP for each respondent.</div>', unsafe_allow_html=True)

    if not os.path.exists("model_metrics.json"):
        st.warning("⚙️ Train models first via the sidebar before scoring new customers.")
    else:
        st.markdown("""
        <div class="insight-box">
        <b>How it works:</b> Upload a CSV file with the same column structure as the training data. 
        Each respondent will receive: <b>Persona Label</b>, <b>Adoption Probability (0–1)</b>, 
        <b>Predicted Monthly WTP (₹)</b>, <b>Priority Tier</b> (Hot/Warm/Cold), and <b>Recommended Action</b>.
        </div>
        """, unsafe_allow_html=True)

        col_up, col_info = st.columns([0.5, 0.5])
        with col_up:
            uploaded = st.file_uploader("Upload new respondent CSV", type=["csv"], label_visibility="collapsed")
        with col_info:
            st.markdown("""
            **Required columns:** age, gender, state, location_type, family_income,
            education_level, stream, board, academic_performance, career_clarity,
            urgency, decision_maker, primary_info_source, past_career_spend,
            career_domain_interest, preferred_feature, motivation, fear_score, risk_score
            """)
            if st.button("📥 Download Template CSV"):
                template = df[["age","gender","state","location_type","family_income",
                                "education_level","stream","board","academic_performance",
                                "career_clarity","urgency","decision_maker","primary_info_source",
                                "past_career_spend","career_domain_interest","preferred_feature",
                                "motivation","fear_score","risk_score"]].head(5)
                st.download_button("Download", template.to_csv(index=False),
                                   file_name="careergps_template.csv", mime="text/csv")

        if uploaded:
            new_df = pd.read_csv(uploaded)
            try:
                import predictor
                results = predictor.score(new_df)
                st.success(f"✅ Scored {len(results)} respondents")

                hot = (results["priority_tier"]=="Hot").sum()
                warm = (results["priority_tier"]=="Warm").sum()
                cold = (results["priority_tier"]=="Cold").sum()

                c1,c2,c3,c4 = st.columns(4)
                c1.markdown(kpi("Total Scored",f"{len(results)}","New respondents","","pos"), unsafe_allow_html=True)
                c2.markdown(kpi("🔴 Hot Leads",f"{hot}",f"{hot/len(results)*100:.0f}% of batch","High priority","pos"), unsafe_allow_html=True)
                c3.markdown(kpi("🟡 Warm Leads",f"{warm}",f"{warm/len(results)*100:.0f}%","Medium priority","pos"), unsafe_allow_html=True)
                c4.markdown(kpi("🔵 Cold Leads",f"{cold}",f"{cold/len(results)*100:.0f}%","Low priority","neg"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(results.style.background_gradient(subset=["adoption_probability","predicted_wtp"], cmap="Blues"),
                             use_container_width=True)
                st.download_button("📥 Download Scored Results",
                                   results.to_csv(index=False),
                                   file_name="careergps_scored_leads.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Scoring error: {str(e)}")
        else:
            st.markdown("""
            <div class="upload-zone">
            <div style="font-size:40px;margin-bottom:12px;">📤</div>
            <div style="font-size:16px;color:#374151;font-weight:600;">Drop your CSV here or click to upload</div>
            <div style="font-size:13px;color:#6b7280;margin-top:6px;">Max 50MB · CSV format only</div>
            </div>
            """, unsafe_allow_html=True)

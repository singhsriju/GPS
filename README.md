# CareerGPS Analytics Dashboard

**AI-Powered Career Guidance Platform — India Market Analytics**  
MBA-level business intelligence dashboard built with Streamlit.

## Pages
1. 🏠 Business Overview — KPIs, persona map, revenue model
2. 📊 Descriptive EDA — Demographics, education, income, WTP, geography
3. 🔬 Diagnostic Analysis — Correlation, ARM, clustering, cross-tabs
4. 🤖 Predictive Models — Classifier (adoption) + Regressor (WTP) with metrics
5. 🧭 Prescriptive Actions — Segment playbooks, revenue projection
6. 📤 New Customer Scoring — Upload CSV, get scored leads

## Deploy on Streamlit Cloud
1. Push all files to GitHub repo root
2. Create `.streamlit/config.toml` (see below)
3. Deploy via share.streamlit.io → app.py
4. Click "Train Models" in sidebar on first launch

## .streamlit/config.toml
```toml
[theme]
primaryColor = "#1e40af"
backgroundColor = "#f8fafd"
secondaryBackgroundColor = "#ffffff"
textColor = "#0f1729"
font = "sans serif"

[server]
maxUploadSize = 50
```

## Dataset
2,000 synthetic Indian student survey respondents with 24 features.

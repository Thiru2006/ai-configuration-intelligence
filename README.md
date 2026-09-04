# PILOT - Performance & Intelligence from LOGs for Optimal Testing

An AI/ML-powered Streamlit dashboard for analyzing VLSI / semiconductor execution logs and configuration data. The application combines log processing, statistical analysis, machine learning, anomaly detection, configuration stability analysis, failure analysis, early-warning risk scoring, and an AI Configuration Intelligence Copilot.

## Key capabilities

- Configuration and execution-log analysis
- Configuration normalization and feature extraction
- Pass/fail and failure-mode analysis
- Randomization and parameter sensitivity analysis
- Configuration performance scoring and Pareto analysis
- Failure clustering and seed-based repeatability analysis
- Configuration difference and change-impact analysis
- Random-Forest failure-risk prediction
- SHAP-based explainability
- Isolation-Forest anomaly detection
- Early-warning risk scoring for configurations
- Configuration recommendations
- Interactive Plotly visualizations and filters
- AI Configuration Intelligence Copilot using Gemini
- Downloadable analysis report

## Project structure

```text
.
├── app.py
├── log_processor.py
├── CHANGES.md
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

`secrets.toml` is intentionally excluded from GitHub.

## Local setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini locally

Create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example` and put your own Gemini API key in it.

**Never commit `.streamlit/secrets.toml` or expose the API key publicly.**

### 5. Run the dashboard

```bash
streamlit run app.py
```

## Data

The dashboard is designed to work with execution-log CSV data. Use the application's CSV upload/input workflow to provide the dataset you want to analyze.

Local datasets are excluded by `.gitignore` so that private or generated data is not accidentally committed.

## AI / ML architecture

```text
Execution Logs / CSV
        │
        ▼
Data Processing & Normalization
        │
        ├── Feature extraction
        ├── Configuration normalization
        └── Outcome / failure canonicalization
        │
        ▼
Analytics Layer
        │
        ├── Statistics & correlations
        ├── Sensitivity analysis
        ├── Performance / Pareto analysis
        └── Failure fingerprints
        │
        ▼
AI / ML Layer
        │
        ├── Random Forest failure prediction
        ├── SHAP explainability
        ├── Isolation Forest anomaly detection
        ├── K-Means clustering
        ├── Repeatability / seed analysis
        └── Configuration recommendations
        │
        ▼
Streamlit Dashboard
        │
        └── Gemini Configuration Intelligence Copilot
```

## Q1-Q7 coverage

The prototype addresses the seven core analysis questions through influence analysis, performance optimization, randomization impact, failure clustering/repeatability, root-cause/failure fingerprints, configuration change impact, and early-warning failure-risk prediction.

## Important modeling note

Predicted failure risk is based on historical configuration patterns. It is an estimate and is not a guarantee of future failure. Statistical associations should not be interpreted as proof of causation.

## Security

- API keys must be stored in Streamlit secrets.
- `.streamlit/secrets.toml` is ignored by Git.
- Do not paste API keys into source code, README files, screenshots, issues, or commit history.
- If a key is accidentally exposed, revoke/rotate it immediately.

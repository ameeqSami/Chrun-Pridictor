# Customer Churn Intelligence Portal

A full-stack evaluation and analysis portal built for the Telco Customer Churn test cohort, powered directly by the trained `pipeline_dt.pkl` model from `chrunPipe.ipynb`.

## Key Capabilities

- **Test Cohort Explorer**: Browse and filter all 1,409 test-set customers (derived from the exact 80/20 stratified split used during pipeline training).
- **One-Click Customer Dossier**: Inspect any customer's churn risk probability, model verdict, and comparison against ground truth.
- **Explainability & Signal Breakdown**: Understand the exact impact of decisive features (`tenure`, `Contract_Two year`, `InternetService_Fiber optic`, `PaymentMethod_Electronic check`).
- **Decision Tree Path Trace**: Step-by-step audit of the tree traversal from root split down to the exact leaf node and sample class distribution.
- **Interactive "What-If" Simulator**: Real-time counterfactual testing to evaluate how contract adjustments or automated payment adoption reduce churn risk.
- **Actionable Retention Playbook**: Personalized, rule-driven retention interventions for at-risk accounts.

## Non-AI Editorial Design

Built with an editorial financial command console aesthetic:
- Warm charcoal, ivory, terracotta alert, amber watchlist, and juniper retention styling (no generic AI purple/cyan glows).
- Micro-interactions, custom SVG probability arc gauge, and tabular figures.

## Running Locally

```bash
# From the repository root:
python churn-portal/backend/server.py
```

Then open [http://localhost:5050](http://localhost:5050) in your browser.

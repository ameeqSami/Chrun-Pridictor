import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure dataset and libs directory is importable for custom transformers
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIBS_DIR = os.path.join(BASE_DIR, "dataset and other libs")
MODEL_PATH = os.path.join(BASE_DIR, "churn-predictor", "pipeline_dt.pkl")
DATASET_PATH = os.path.join(LIBS_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

if LIBS_DIR not in sys.path:
    sys.path.append(LIBS_DIR)

from cleaningcls import clean_cls
from ctf import CorrelationThresholdFilter
from pridict_thresh import pridict_thresh
from trace_path import get_customer_path_df, _extract_predictor_and_model, _get_feature_names

class ChurnModelService:
    def __init__(self):
        self.pipeline = None
        self.raw_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_records = []
        self.feature_importances = {}
        self.load_model_and_data()

    def load_model_and_data(self):
        print(f"Loading pipeline from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            self.pipeline = pickle.load(f)

        print(f"Loading dataset from {DATASET_PATH}...")
        self.raw_df = pd.read_csv(DATASET_PATH)

        X = self.raw_df.drop(columns=['Churn'])
        y = self.raw_df['Churn'].map({'Yes': 1, 'No': 0})

        # Exact stratified split used during pipeline evaluation (80/20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Reset index for consistent integer indexing
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        print("Computing baseline test set predictions...")
        probs = self.pipeline.predict_proba(self.X_test)[:, 1]
        preds = self.pipeline.predict(self.X_test)

        # Extract model and feature importances
        predictor, model = _extract_predictor_and_model(self.pipeline)
        ctf_features = list(self.pipeline.named_steps['ctf'].get_feature_names_out())
        importances = getattr(model, 'feature_importances_', None)

        if importances is not None:
            self.feature_importances = {
                feat: float(imp)
                for feat, imp in zip(ctf_features, importances)
            }
        else:
            self.feature_importances = {feat: 1.0 / len(ctf_features) for feat in ctf_features}

        # Build fast lookup list of test customers
        self.test_records = []
        for i in range(len(self.X_test)):
            row = self.X_test.iloc[i].to_dict()
            prob = float(probs[i])
            pred = int(preds[i])
            actual = int(self.y_test.iloc[i])

            if prob >= 0.60:
                risk_tier = "High"
            elif prob >= 0.35:
                risk_tier = "Medium"
            else:
                risk_tier = "Low"

            # Parse TotalCharges numeric
            tc = row.get('TotalCharges')
            try:
                tc_val = float(tc) if pd.notnull(tc) and str(tc).strip() != "" else 0.0
            except:
                tc_val = 0.0

            mc_val = float(row.get('MonthlyCharges', 0.0))
            tenure_val = int(row.get('tenure', 0))

            customer_summary = {
                "test_index": i,
                "customerID": row.get('customerID', f"CUST-{i+1:04d}"),
                "gender": row.get('gender'),
                "SeniorCitizen": int(row.get('SeniorCitizen', 0)),
                "Partner": row.get('Partner'),
                "Dependents": row.get('Dependents'),
                "tenure": tenure_val,
                "PhoneService": row.get('PhoneService'),
                "MultipleLines": row.get('MultipleLines'),
                "InternetService": row.get('InternetService'),
                "OnlineSecurity": row.get('OnlineSecurity'),
                "OnlineBackup": row.get('OnlineBackup'),
                "DeviceProtection": row.get('DeviceProtection'),
                "TechSupport": row.get('TechSupport'),
                "StreamingTV": row.get('StreamingTV'),
                "StreamingMovies": row.get('StreamingMovies'),
                "Contract": row.get('Contract'),
                "PaperlessBilling": row.get('PaperlessBilling'),
                "PaymentMethod": row.get('PaymentMethod'),
                "MonthlyCharges": mc_val,
                "TotalCharges": tc_val,
                "churn_prob": round(prob, 4),
                "churn_prob_pct": round(prob * 100, 1),
                "prediction": pred,
                "prediction_label": "Likely to Churn" if pred == 1 else "Likely to Retain",
                "actual_churn": actual,
                "actual_label": "Churned" if actual == 1 else "Retained",
                "risk_tier": risk_tier,
                "is_match": (pred == actual)
            }
            self.test_records.append(customer_summary)

        print(f"Model service ready! {len(self.test_records)} test customers processed.")

    def get_summary_metrics(self):
        total = len(self.test_records)
        high_risk = sum(1 for c in self.test_records if c['risk_tier'] == 'High')
        medium_risk = sum(1 for c in self.test_records if c['risk_tier'] == 'Medium')
        low_risk = sum(1 for c in self.test_records if c['risk_tier'] == 'Low')
        pred_churn = sum(1 for c in self.test_records if c['prediction'] == 1)
        actual_churn = sum(1 for c in self.test_records if c['actual_churn'] == 1)
        matches = sum(1 for c in self.test_records if c['is_match'])
        avg_prob = sum(c['churn_prob'] for c in self.test_records) / total if total > 0 else 0

        return {
            "total_customers": total,
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "predicted_churn_count": pred_churn,
            "actual_churn_count": actual_churn,
            "accuracy_pct": round((matches / total) * 100, 1) if total > 0 else 0,
            "avg_churn_prob_pct": round(avg_prob * 100, 1),
            "model_type": "Decision Tree Classifier (chrunPipe.ipynb)",
            "features_used": list(self.feature_importances.keys()),
            "feature_importances": self.feature_importances
        }

    def search_customers(self, query=None, risk_tier=None, contract=None, internet=None,
                         sort_by="churn_prob", sort_dir="desc", page=1, per_page=25):
        filtered = self.test_records

        if query:
            q = query.strip().lower()
            filtered = [
                c for c in filtered
                if q in c['customerID'].lower()
                or q in c['Contract'].lower()
                or q in c['PaymentMethod'].lower()
                or q in c['InternetService'].lower()
            ]

        if risk_tier and risk_tier.lower() != "all":
            filtered = [c for c in filtered if c['risk_tier'].lower() == risk_tier.lower()]

        if contract and contract.lower() != "all":
            filtered = [c for c in filtered if c['Contract'].lower() == contract.lower()]

        if internet and internet.lower() != "all":
            filtered = [c for c in filtered if c['InternetService'].lower() == internet.lower()]

        # Sorting
        reverse = (sort_dir.lower() == "desc")
        if sort_by in ["churn_prob", "tenure", "MonthlyCharges", "TotalCharges"]:
            filtered = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=reverse)
        elif sort_by in ["customerID", "risk_tier", "Contract", "InternetService"]:
            filtered = sorted(filtered, key=lambda x: str(x.get(sort_by, '')), reverse=reverse)

        total_matching = len(filtered)

        # Pagination
        if per_page <= 0:
            paged = filtered
        else:
            start = (page - 1) * per_page
            end = start + per_page
            paged = filtered[start:end]

        return {
            "total": total_matching,
            "page": page,
            "per_page": per_page,
            "total_pages": int(np.ceil(total_matching / per_page)) if per_page > 0 else 1,
            "customers": paged
        }

    def analyze_customer(self, test_index):
        if test_index < 0 or test_index >= len(self.X_test):
            raise IndexError("Customer index out of bounds")

        summary = self.test_records[test_index]
        raw_row = self.X_test.iloc[[test_index]]

        # Preprocess customer through pipeline steps for path trace
        clean_step = self.pipeline.named_steps['clean']
        ctf_step = self.pipeline.named_steps['ctf']

        X_clean = clean_step.transform(raw_row)
        X_filtered = ctf_step.transform(X_clean)

        # Traced decision path
        path_df = get_customer_path_df(self.pipeline, X_filtered, customer_idx=0)
        decision_steps = []
        for idx, row in path_df.iterrows():
            decision_steps.append({
                "node_id": int(row["Node ID"]),
                "node_type": row["Node Type"],
                "feature": row["Feature Checked"],
                "threshold_rule": row["Threshold Rule"],
                "customer_value": None if row["Customer Value"] == "N/A" else float(row["Customer Value"]),
                "condition_met": None if row["Condition Met"] == "N/A" else bool(row["Condition Met"]),
                "direction": row["Direction"],
                "samples": int(row["Samples"]),
                "churn_prob": float(row["Churn Probability"]),
                "churn_prob_pct": round(float(row["Churn Probability"]) * 100, 1),
                "class_dist": row["Class Distribution"],
                "prediction": None if pd.isna(row.get("Prediction")) else int(row["Prediction"])
            })

        # Feature signals and contributions
        active_features = list(ctf_step.get_feature_names_out())
        feature_signals = []
        for feat in active_features:
            val = float(X_filtered[feat].iloc[0])
            importance = self.feature_importances.get(feat, 0.0)

            # High risk drivers explanation
            risk_contribution = "Neutral"
            if "Contract_Two year" in feat:
                risk_contribution = "Protective (Reduces Risk)" if val == 1 else "Elevated Risk (No 2-Yr Contract)"
            elif "Fiber optic" in feat:
                risk_contribution = "Elevated Risk (Higher Churn Segment)" if val == 1 else "Protective"
            elif "Electronic check" in feat:
                risk_contribution = "Elevated Risk (Manual Billing)" if val == 1 else "Protective (Auto/Direct)"
            elif feat == "tenure":
                if val <= 6:
                    risk_contribution = "Critical Risk (New Customer < 6m)"
                elif val <= 24:
                    risk_contribution = "Moderate Risk (Tenure under 2 yrs)"
                else:
                    risk_contribution = "Strongly Protective (Established Customer)"

            feature_signals.append({
                "feature": feat,
                "importance": round(importance, 4),
                "importance_pct": round(importance * 100, 1),
                "customer_value": val,
                "risk_contribution": risk_contribution
            })

        # Sort features by importance
        feature_signals = sorted(feature_signals, key=lambda x: x['importance'], reverse=True)

        # Domain retention advice
        retention_actions = self._generate_retention_actions(summary)

        return {
            "summary": summary,
            "feature_signals": feature_signals,
            "decision_path": decision_steps,
            "retention_actions": retention_actions,
            "model_metadata": {
                "classifier": "DecisionTreeClassifier",
                "source_notebook": "chrunPipe.ipynb",
                "features_analyzed": len(feature_signals)
            }
        }

    def _generate_retention_actions(self, cust):
        actions = []
        if cust['Contract'] == "Month-to-month":
            actions.append({
                "priority": "Critical",
                "title": "Convert to Long-Term Commitment",
                "action": "Offer a 15% discount on an annual contract or a locked 2-year pricing agreement.",
                "reason": "Month-to-month customers represent over 70% of historical churners."
            })

        if cust['PaymentMethod'] == "Electronic check":
            actions.append({
                "priority": "High",
                "title": "Incentivize Automated Billing",
                "action": "Offer a one-time $10 credit to switch to Automatic Bank Transfer or Credit Card.",
                "reason": "Electronic check users churn at nearly 3x the rate of automated payment methods."
            })

        if cust['InternetService'] == "Fiber optic" and cust['TechSupport'] == "No":
            actions.append({
                "priority": "High",
                "title": "Complimentary Tech Support & Security Bundle",
                "action": "Provide 3 months of complimentary Tech Support and Online Security.",
                "reason": "Fiber customers with zero bundled support experience higher dissatisfaction churn."
            })

        if cust['tenure'] <= 12:
            actions.append({
                "priority": "Medium",
                "title": "First-Year Loyalty Checkpoint",
                "action": "Schedule a dedicated account review with a customer success agent.",
                "reason": "Early-tenure customers require proactive onboarding to survive the critical drop-off window."
            })

        if not actions:
            actions.append({
                "priority": "Standard",
                "title": "Sustained Relationship Care",
                "action": "Maintain current service tier with periodic loyalty perks or speed upgrades.",
                "reason": "Customer profile indicates high retention stability."
            })

        return actions

    def simulate_what_if(self, test_index, contract_override=None, payment_override=None,
                         tech_support_override=None, tenure_months_override=None):
        if test_index < 0 or test_index >= len(self.X_test):
            raise IndexError("Customer index out of bounds")

        raw_row = self.X_test.iloc[[test_index]].copy()

        if contract_override:
            raw_row['Contract'] = contract_override
        if payment_override:
            raw_row['PaymentMethod'] = payment_override
        if tech_support_override:
            raw_row['TechSupport'] = tech_support_override
        if tenure_months_override is not None:
            raw_row['tenure'] = tenure_months_override

        new_prob = float(self.pipeline.predict_proba(raw_row)[:, 1][0])
        new_pred = int(self.pipeline.predict(raw_row)[0])
        old_prob = self.test_records[test_index]['churn_prob']

        return {
            "original_prob": old_prob,
            "original_prob_pct": round(old_prob * 100, 1),
            "simulated_prob": round(new_prob, 4),
            "simulated_prob_pct": round(new_prob * 100, 1),
            "simulated_prediction": new_pred,
            "simulated_label": "Likely to Churn" if new_pred == 1 else "Likely to Retain",
            "delta_pct": round((new_prob - old_prob) * 100, 1),
            "reduced_risk": (new_prob < old_prob)
        }

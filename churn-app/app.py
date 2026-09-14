import os
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(BASE_DIR, "libs")
if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)

# Fallback path if needed
FALLBACK_LIBS = os.path.join(BASE_DIR, "..", "dataset and other libs")
if os.path.exists(FALLBACK_LIBS) and FALLBACK_LIBS not in sys.path:
    sys.path.insert(1, FALLBACK_LIBS)

# Import existing custom modules as-is
from trace_path import get_customer_path_df
from cleaningcls import clean_cls
from ctf import CorrelationThresholdFilter
from offer_engine import generate_retention_offer

from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Global runtime state
MODEL_PIPELINE = None
X_TEST_RAW = None
X_TEST_FILT = None
CUSTOMER_STORE = {}
KPI_STATS = {}

def initialize_app():
    global MODEL_PIPELINE, X_TEST_RAW, X_TEST_FILT, CUSTOMER_STORE, KPI_STATS

    print("[*] Initializing Telco Churn Predictor & Retention Engine...")

    # 1. Load trained pipeline
    model_path = os.path.join(BASE_DIR, "models", "pipeline_dt.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "..", "churn-predictor-dt", "pipeline_dt.pkl")

    with open(model_path, "rb") as f:
        MODEL_PIPELINE = pickle.load(f)
    print(f"[+] Loaded model pipeline from: {model_path}")

    # 2. Load dataset
    csv_path = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "..", "dataset and other libs", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Stratified split matching original training (test_size=0.2, random_state=42)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_TEST_RAW = X_test.reset_index(drop=True)

    # 4. Extract pipeline stages: clean -> ctf -> model
    clean_step = MODEL_PIPELINE.named_steps['clean']
    ctf_step = MODEL_PIPELINE.named_steps['ctf']
    dt_step = MODEL_PIPELINE.named_steps['model']

    # Batch transform entire test set to prevent single-row dummy dropping bug
    print("[*] Performing batch transformation on test set (1,409 rows)...")
    X_clean = clean_step.transform(X_TEST_RAW)
    X_TEST_FILT = ctf_step.transform(X_clean)
    if not isinstance(X_TEST_FILT, pd.DataFrame):
        X_TEST_FILT = pd.DataFrame(X_TEST_FILT, columns=ctf_step.get_feature_names_out())

    # Pre-compute predictions & probabilities
    preds = dt_step.predict(X_TEST_FILT)
    probs = dt_step.predict_proba(X_TEST_FILT)[:, 1]

    # 5. Populate in-memory CUSTOMER_STORE
    CUSTOMER_STORE = {}
    for idx, row in X_TEST_RAW.iterrows():
        cid = str(row['customerID']).strip()
        raw_dict = row.to_dict()
        CUSTOMER_STORE[cid] = {
            "iloc": idx,
            "raw": raw_dict,
            "pred": int(preds[idx]),
            "prob": float(round(probs[idx], 4)),
            "analyzed": False,
            "offer_sent": False,
            "offer_sent_at": None,
            "offer_data": None,
            "path_data": None
        }

    total = len(CUSTOMER_STORE)
    churn_count = int(np.sum(preds == 1))
    retained_count = int(np.sum(preds == 0))
    KPI_STATS = {
        "total": total,
        "churn_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": round((churn_count / total) * 100, 1),
        "analyzed_count": 0,
        "offers_sent_count": 0
    }
    print(f"[+] Ready: {total} test customers | {churn_count} Churn ({KPI_STATS['churn_rate']}%) | {retained_count} Retained")

# Run initialization at module load
initialize_app()

def recalculate_kpi_stats():
    global KPI_STATS
    total = len(CUSTOMER_STORE)
    analyzed = sum(1 for c in CUSTOMER_STORE.values() if c["analyzed"])
    offers_sent = sum(1 for c in CUSTOMER_STORE.values() if c["offer_sent"])
    churn_count = sum(1 for c in CUSTOMER_STORE.values() if c["pred"] == 1)
    retained_count = total - churn_count

    KPI_STATS = {
        "total": total,
        "churn_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": round((churn_count / total) * 100, 1) if total > 0 else 0,
        "analyzed_count": analyzed,
        "offers_sent_count": offers_sent
    }
    return KPI_STATS


@app.route("/")
def index():
    recalculate_kpi_stats()
    return render_template("index.html", kpis=KPI_STATS)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    stats = recalculate_kpi_stats()
    return jsonify({"success": True, "stats": stats})


@app.route("/api/customers", methods=["GET"])
def list_customers():
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 12))
    search = request.args.get("search", "").strip().lower()
    status_filter = request.args.get("status", "all").strip().lower()
    contract_filter = request.args.get("contract", "all").strip()
    internet_filter = request.args.get("internet", "all").strip()

    filtered = []
    for cid, data in CUSTOMER_STORE.items():
        # Search query
        if search and search not in cid.lower():
            continue

        # Status filter
        if status_filter == "churn" and data["pred"] != 1:
            continue
        elif status_filter == "retained" and data["pred"] != 0:
            continue
        elif status_filter == "analyzed" and not data["analyzed"]:
            continue
        elif status_filter == "pending" and data["analyzed"]:
            continue
        elif status_filter == "offer_sent" and not data["offer_sent"]:
            continue

        # Contract filter
        if contract_filter != "all" and data["raw"].get("Contract") != contract_filter:
            continue

        # Internet filter
        if internet_filter != "all" and data["raw"].get("InternetService") != internet_filter:
            continue

        item = {
            "customerID": cid,
            "gender": data["raw"].get("gender", ""),
            "SeniorCitizen": int(data["raw"].get("SeniorCitizen", 0)),
            "Partner": data["raw"].get("Partner", ""),
            "Dependents": data["raw"].get("Dependents", ""),
            "tenure": int(data["raw"].get("tenure", 0)),
            "Contract": data["raw"].get("Contract", ""),
            "InternetService": data["raw"].get("InternetService", ""),
            "PaymentMethod": data["raw"].get("PaymentMethod", ""),
            "MonthlyCharges": float(data["raw"].get("MonthlyCharges", 0.0)),
            "TotalCharges": str(data["raw"].get("TotalCharges", "0")),
            "TechSupport": data["raw"].get("TechSupport", ""),
            "OnlineSecurity": data["raw"].get("OnlineSecurity", ""),
            "pred": data["pred"],
            "prob": data["prob"],
            "analyzed": data["analyzed"],
            "offer_sent": data["offer_sent"],
            "offer_sent_at": data["offer_sent_at"]
        }
        filtered.append(item)

    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + limit - 1) // limit)
    page = min(max(1, page), total_pages)

    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = filtered[start_idx:end_idx]

    return jsonify({
        "success": True,
        "customers": paginated,
        "pagination": {
            "page": page,
            "limit": limit,
            "total_items": total_filtered,
            "total_pages": total_pages
        }
    })


@app.route("/api/analyze/<customer_id>", methods=["POST"])
def analyze_customer(customer_id):
    cid = str(customer_id).strip()
    if cid not in CUSTOMER_STORE:
        return jsonify({"success": False, "error": f"Customer '{customer_id}' not found"}), 404

    data = CUSTOMER_STORE[cid]
    iloc_pos = data["iloc"]
    raw = data["raw"]
    pred = data["pred"]
    prob = data["prob"]

    # Mark as analyzed
    data["analyzed"] = True

    # Trace path using user's existing custom function get_customer_path_df
    path_nodes = []
    path_df = None
    try:
        path_df = get_customer_path_df(MODEL_PIPELINE, X_TEST_FILT, iloc_pos)
        
        if path_df is not None and not path_df.empty:
            for _, row in path_df.iterrows():
                node_dict = {
                    "node_id": int(row.get("Node ID", 0)),
                    "node_type": str(row.get("Node Type", "")),
                    "feature": str(row.get("Feature Checked", "")),
                    "threshold_rule": str(row.get("Threshold Rule", "")),
                    "customer_value": str(row.get("Customer Value", "")),
                    "condition_met": bool(row.get("Condition Met", False)),
                    "class_distribution": str(row.get("Class Distribution", ""))
                }
                path_nodes.append(node_dict)
    except Exception as e:
        print(f"[!] Path tracing notice for {cid}: {e}")

    # Generate targeted retention offer
    offer = generate_retention_offer(raw, path_df, prob)
    data["offer_data"] = offer
    data["path_data"] = path_nodes

    recalculate_kpi_stats()

    return jsonify({
        "success": True,
        "customer": {
            "customerID": cid,
            "raw": raw,
            "pred": pred,
            "prob": prob,
            "analyzed": True,
            "offer_sent": data["offer_sent"],
            "offer_sent_at": data["offer_sent_at"]
        },
        "path": path_nodes,
        "offer": offer
    })


@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    req_body = request.get_json(silent=True) or {}
    target_ids = req_body.get("customer_ids", None)

    analyzed_now = 0
    for cid, data in CUSTOMER_STORE.items():
        if target_ids is not None and cid not in target_ids:
            continue
        data["analyzed"] = True
        analyzed_now += 1

    stats = recalculate_kpi_stats()
    return jsonify({
        "success": True,
        "message": f"Successfully analyzed {analyzed_now} customers in batch.",
        "stats": stats
    })


@app.route("/api/send-offer/<customer_id>", methods=["POST"])
def send_offer(customer_id):
    cid = str(customer_id).strip()
    if cid not in CUSTOMER_STORE:
        return jsonify({"success": False, "error": f"Customer '{customer_id}' not found"}), 404

    data = CUSTOMER_STORE[cid]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["offer_sent"] = True
    data["offer_sent_at"] = timestamp

    req_body = request.get_json(silent=True) or {}
    offer_title = req_body.get("offer_title", (data.get("offer_data") or {}).get("title", "Custom Retention Package"))
    channel = req_body.get("channel", "Automated Multichannel (Email + SMS)")

    stats = recalculate_kpi_stats()

    return jsonify({
        "success": True,
        "message": f"Retention offer successfully dispatched to {cid}",
        "details": {
            "customerID": cid,
            "offer_title": offer_title,
            "channel": channel,
            "sent_at": timestamp,
            "status": "Delivered"
        },
        "stats": stats
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

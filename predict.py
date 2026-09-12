import pickle
import pandas as pd
from cleaningcls import clean_cls
from ctf import CorrelationThresholdFilter

def load_model(model_path='pipeline_dt.pkl'):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_single(model, customer_dict):
    df_customer = pd.DataFrame([customer_dict])
    pred = model.predict(df_customer)[0]
    prob = model.predict_proba(df_customer)[0]
    return {
        'prediction': 'Churn' if pred == 1 else 'No Churn',
        'churn_probability': round(float(prob[1]), 4),
        'retention_probability': round(float(prob[0]), 4)
    }

if __name__ == '__main__':
    # Load model
    model = load_model('pipeline_dt.pkl')

    # Example raw customer data
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    result = predict_single(model, sample_customer)
    print("=== Single Customer Prediction ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

"""
Targeted Retention Offer Engine
Generates dynamic, customer-specific retention strategies based on the Decision Tree
decision path nodes and failing/risk features identified by the model.
"""

def generate_retention_offer(customer_raw: dict, path_df, churn_prob: float) -> dict:
    """
    Analyzes customer attributes and decision path nodes to formulate a tailored offer.
    
    Parameters:
    - customer_raw: dict of raw customer features (e.g. Contract, InternetService, tenure, MonthlyCharges, etc.)
    - path_df: pandas DataFrame returned by get_customer_path_df() containing traversed nodes
    - churn_prob: float churn probability (0.0 to 1.0)
    
    Returns:
    - dict with structured offer details:
      {
        "title": str,
        "badge": str,
        "discount": str,
        "incentives": list[str],
        "rationale": str,
        "key_risk_factor": str,
        "est_revenue_saved": str
      }
    """
    tenure = float(customer_raw.get('tenure', 0))
    monthly_charges = float(customer_raw.get('MonthlyCharges', 0.0))
    internet_service = str(customer_raw.get('InternetService', '')).strip()
    contract = str(customer_raw.get('Contract', '')).strip()
    payment_method = str(customer_raw.get('PaymentMethod', '')).strip()

    # Inspect path features if available
    path_features = []
    if path_df is not None and hasattr(path_df, 'columns') and not path_df.empty:
        if 'Feature Checked' in path_df.columns:
            path_features = [str(f) for f in path_df['Feature Checked'].dropna().tolist()]

    # Priority 1: High-Speed Fiber Optic Churn Risk with Early/Mid Tenure (very high churn node)
    if 'Fiber optic' in internet_service and tenure <= 26.5:
        monthly_saving = round(monthly_charges * 0.25, 2)
        annual_val = round((monthly_charges - monthly_saving) * 12, 2)
        return {
            "title": "Fiber Premier Retention & Care Suite",
            "badge": "Critical Risk • High Value",
            "discount": "25% Off Fiber Optic for 6 Months",
            "incentives": [
                "Complimentary Priority Tech Support & Wi-Fi 6 Router Upgrade ($120 value)",
                "Free Online Security & Cloud Backup package for 12 months",
                "Direct Dedicated Concierge Helpline with <30 sec wait times",
                "Price-Lock Guarantee for 12 months with no cancellation penalty"
            ],
            "rationale": f"Customer is on high-tier Fiber optic (${monthly_charges}/mo) with early/mid tenure ({int(tenure)} mos), matching the decision tree's highest-churn branch. Fiber churners often defect due to speed-to-price sensitivity and early friction.",
            "key_risk_factor": f"Fiber Optic service combined with early tenure ({int(tenure)} months)",
            "est_revenue_saved": f"${annual_val}/year retained"
        }

    # Priority 2: Established Fiber Optic Customers with Churn Signal
    if 'Fiber optic' in internet_service and tenure > 26.5:
        monthly_saving = round(monthly_charges * 0.20, 2)
        annual_val = round((monthly_charges - monthly_saving) * 12, 2)
        return {
            "title": "VIP Fiber Loyalty Appreciation Bundle",
            "badge": "High Value • Loyalty Risk",
            "discount": "20% Monthly Loyalty Credit for 12 Months",
            "incentives": [
                "Next-Gen Mesh Wi-Fi Extender kit free of charge",
                "Complimentary Streaming Pass (Hulu / Disney+ Bundle) for 6 months",
                "Annual Account Review with customized data & tier tuning",
                "Zero activation or equipment rental fees indefinitely"
            ],
            "rationale": f"Tenured subscriber ({int(tenure)} mos) paying ${monthly_charges}/mo. The tree highlights rate fatigue and lack of recent bundle rewards.",
            "key_risk_factor": f"Fiber Optic tariff fatigue despite {int(tenure)} months tenure",
            "est_revenue_saved": f"${annual_val}/year retained"
        }

    # Priority 3: Month-to-Month Contract Vulnerability (Contract_Two year == 0)
    if 'Month-to-month' in contract:
        monthly_saving = 20.00
        annual_val = round((monthly_charges - 20) * 12, 2)
        return {
            "title": "Annual Commitment Freedom Plan",
            "badge": "Contract Optimization",
            "discount": "$20 Off/Month with 1-Year Price Assurance",
            "incentives": [
                "$100 Bill Credit applied immediately to next statement",
                "No penalty 30-day grace period if dissatisfied",
                "Complimentary Streaming or Family Line Add-on for 3 months",
                "Guaranteed zero rate hikes through next fiscal year"
            ],
            "rationale": "Month-to-month contracts trigger root node branch divergence toward churn. Locking into an incentivized 1-year agreement eliminates contract volatility.",
            "key_risk_factor": "Month-to-Month contract with zero cancellation barrier",
            "est_revenue_saved": f"${annual_val}/year retained"
        }

    # Priority 4: Electronic Check Payment Friction
    if 'Electronic check' in payment_method:
        return {
            "title": "AutoPay & Paperless Seamless Rewards",
            "badge": "Billing Optimization",
            "discount": "$15 Monthly Credit for Switching to AutoPay",
            "incentives": [
                "Instant $25 One-Time Credit upon linking Credit Card / ACH",
                "Automated fraud alerts and transaction protection",
                "Flexible billing date picker (align with payday)",
                "Paperless digital dividend bonus every quarter"
            ],
            "rationale": "Electronic check users exhibit high churn likelihood in the model due to payment friction, manual bill processing, and dispute frequency.",
            "key_risk_factor": "Electronic Check payment method (friction indicator)",
            "est_revenue_saved": f"${round(monthly_charges * 12, 2)}/year retained"
        }

    # Priority 5: Brand New Onboarding Tenure Risk (tenure <= 6)
    if tenure <= 6:
        return {
            "title": "Early Journey Success & Concierge Pack",
            "badge": "Early Onboarding Risk",
            "discount": "50% Off 2nd & 3rd Month Statements",
            "incentives": [
                "Personalized 1-on-1 Onboarding Specialist Check-in call",
                "Free Whole-Home Wi-Fi coverage diagnosis & adjustment",
                "Complimentary Device Protection Plan for 6 months",
                "Satisfaction guarantee with instant speed tier boost"
            ],
            "rationale": f"Customer is in the critical first {int(tenure)} months of relationship where initial friction leads to rapid defection.",
            "key_risk_factor": f"Very low tenure ({int(tenure)} months) - high early defection window",
            "est_revenue_saved": f"${round(monthly_charges * 12, 2)}/year retained"
        }

    # Fallback / General High Risk Offer
    monthly_saving = round(monthly_charges * 0.15, 2)
    return {
        "title": "Preferred Customer Advantage Retainer",
        "badge": "Risk Mitigation",
        "discount": "15% Account Discount for 6 Months",
        "incentives": [
            "Free Speed Boost to next broadband tier at zero added cost",
            "Waived service upgrade & maintenance fees",
            "Free Family Entertainment add-on for 90 days",
            "Direct line to Customer Retention Executive"
        ],
        "rationale": f"Model detected composite churn probability of {round(churn_prob * 100, 1)}% across contract and service parameters.",
        "key_risk_factor": f"Composite Churn Risk ({round(churn_prob * 100, 1)}%)",
        "est_revenue_saved": f"${round((monthly_charges - monthly_saving) * 12, 2)}/year retained"
    }

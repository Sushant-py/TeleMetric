SELECT
    customerID,
    Churn_Probability,
    MonthlyCharges,
    Expected_Revenue_at_Risk
FROM customer_risk_scores
ORDER BY Expected_Revenue_at_Risk DESC
LIMIT 100;
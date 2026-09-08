SELECT
    COUNT(*) AS total_customers,
    AVG(Churn_Target) AS churn_rate,
    AVG(MonthlyCharges) AS average_monthly_charge,
    SUM(MonthlyCharges) AS monthly_revenue
FROM customer_master;
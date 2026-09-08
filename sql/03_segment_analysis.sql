SELECT
    Retention_Segment,
    COUNT(*) AS customers,
    AVG(Churn_Probability) AS average_risk,
    SUM(Expected_Revenue_at_Risk)
        AS expected_revenue_at_risk
FROM customer_risk_scores
GROUP BY Retention_Segment
ORDER BY expected_revenue_at_risk DESC;
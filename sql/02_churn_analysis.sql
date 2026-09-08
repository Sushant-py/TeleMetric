SELECT
    Contract,
    COUNT(*) AS customers,
    AVG(Churn_Target) AS churn_rate
FROM customer_master
GROUP BY Contract
ORDER BY churn_rate DESC;
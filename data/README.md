# Data Directory

Place your CSV data files here:
- `dataset_01_customer_churn_risk.csv` - Raw customer churn data
- `dataset_01_customer_churn_risk_conversion_error.xlsx` - Excel format data (if applicable)

## Dataset Structure

Expected columns:
- `tenure_months` - Customer tenure in months
- `monthly_charges` - Monthly subscription charges
- `support_tickets` - Number of support tickets opened
- `avg_session_minutes` - Average session duration
- `late_payments` - Number of late payments
- `contract_months` - Contract duration in months
- Additional feature columns
- Target variable: `churn` (binary: 0 = No Churn, 1 = Churn)

## Note

Data files are git-ignored to avoid storing large files. Update `.gitignore` if needed.

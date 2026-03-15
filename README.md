# Telecom Customer Churn Analysis

An end-to-end data analytics and machine learning portfolio project built in Python using the IBM Telco Customer Churn dataset.

This project covers the full analytics workflow:
- data loading and cleaning
- exploratory data analysis
- churn pattern visualization
- feature engineering
- predictive modeling
- model evaluation
- business insight generation
- interactive dashboard development

## Project Objective

The goal of this project is to analyze telecom customer behavior, identify churn drivers, and build machine learning models that can predict whether a customer is likely to churn.

This type of analysis helps businesses improve customer retention, reduce revenue loss, and design more targeted retention strategies.

## Dataset

- Source file: `data/Telco_Customer_Churn.csv`
- Records: 7,043 customers
- Target variable: `Churn`

The dataset contains demographic information, service subscriptions, account details, billing information, and churn labels.

## Tools and Libraries

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- plotly

## Project Structure

```text
tele_chrun/
├── dashboard.py
├── churn_analysis.py
├── README.md
├── data/
│   └── Telco_Customer_Churn.csv
├── outputs/
│   ├── cleaned_churn_data.csv
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_churn_by_internet_service.png
│   ├── 04_churn_vs_monthly_charges.png
│   ├── 05_tenure_distribution.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_feature_importance.png
│   ├── cm_logistic_regression.png
│   └── cm_random_forest.png
└── .venv/
```

## Workflow

The script performs the following steps:

1. Loads the dataset using pandas.
2. Explores the dataset using `head()`, `info()`, and `describe()`.
3. Checks for missing values.
4. Converts `TotalCharges` to numeric.
5. Removes duplicate rows.
6. Standardizes column names.
7. Performs exploratory data analysis to understand churn trends.
8. Creates visualizations for key churn patterns.
9. Generates business insights.
10. Engineers features and encodes categorical variables.
11. Splits the data into training and testing sets.
12. Trains Logistic Regression and Random Forest models.
13. Evaluates model performance.
14. Identifies the most important churn drivers.
15. Saves the cleaned dataset for future analysis.

## Interactive Dashboard

This project now includes an interactive Streamlit dashboard in `dashboard.py`.

Dashboard features:
- sidebar filters for contract, internet service, payment method, churn, and tenure
- KPI cards for customer count, churn rate, monthly charge, and tenure
- interactive churn charts
- segment-level churn insights
- model performance view with confusion matrices
- feature importance chart
- filtered customer data preview

## Key Insights

- Overall churn rate is approximately 26.5%.
- Customers on month-to-month contracts churn much more than customers on one-year or two-year contracts.
- Fiber optic customers have a higher churn rate than DSL customers.
- Customers with higher monthly charges are more likely to churn.
- Customers with shorter tenure are at higher churn risk.
- Electronic check users show the highest churn among payment methods.

## Model Performance

Two machine learning models were trained:

### Logistic Regression
- Accuracy: 79.77%

### Random Forest
- Accuracy: 79.84%

Both models performed similarly, with Random Forest slightly ahead in overall accuracy.

## Important Features

The most influential features for churn prediction were:

- tenure
- contract
- monthlycharges
- totalcharges
- charges_per_month
- internetservice
- is_long_tenure
- paymentmethod

These variables strongly influence churn behavior and can be used to support retention decisions.

## Output Files

The script generates the following artifacts in the `outputs/` folder:

- cleaned dataset CSV
- churn distribution chart
- churn by contract chart
- churn by internet service chart
- churn vs monthly charges chart
- tenure distribution chart
- correlation heatmap
- feature importance chart
- confusion matrix for Logistic Regression
- confusion matrix for Random Forest

## How to Run

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python churn_analysis.py
python -m streamlit run dashboard.py
```

### If the virtual environment is not created yet

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
python churn_analysis.py
python -m streamlit run dashboard.py
```

## Portfolio Summary

This project demonstrates practical skills in:

- data cleaning
- exploratory data analysis
- data visualization
- feature engineering
- classification modeling
- business insight extraction
- end-to-end analytics project development
- dashboard building with Streamlit

It is suitable for showcasing on GitHub as a data analyst or junior data scientist portfolio project.

## Next Improvements

- add a Jupyter Notebook version for presentation
- add ROC-AUC and precision-recall metrics
- perform hyperparameter tuning
- build an interactive dashboard in Power BI, Tableau, or Streamlit
- deploy the model as a simple web app

- # Live Demo Link
- https://telecom-customer-churn-dashboard.streamlit.app/

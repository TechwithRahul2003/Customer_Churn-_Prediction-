📊 Customer Churn Prediction

📌 Project Overview
Customer churn prediction helps businesses identify customers who are likely to discontinue their service, enabling proactive retention strategies. Since retaining customers is more cost-effective than acquiring new ones, predicting churn can significantly improve profitability.

This project builds a machine learning pipeline to predict churn using demographic, behavioral, and transactional data. The final model (XGBoost) achieved 100% recall after addressing class imbalance with SMOTE, making it highly effective for capturing potential churners.

📂 Dataset
Source: Kaggle – Customer Churn Dataset
Size: 505,256 rows × 12 features

Key Features:

CustomerID – Unique customer identifier

Age – Customer’s age

Gender – Male/Female

Tenure – Duration of service usage

Usage Frequency – Monthly service usage

Support Calls – Customer service calls made

Payment Delay – Late bill payments

Subscription Type – Basic / Standard / Premium

Contract Length – Monthly / Quarterly / Annual

Total Spend – Lifetime spend of customer

Last Interaction – Months since last activity

Churn – Target label (1 = churned, 0 = active)

🛠 Steps Performed
1. Data Preprocessing
Handled missing values

Encoded categorical features (Label Encoding & One-Hot Encoding)

Scaled numeric features using StandardScaler

Removed duplicates

Train-validation-test split with Stratified Sampling

2. Exploratory Data Analysis
Age vs Churn: Customers <30 or >60 churn more; 40–50 are most loyal

Support Calls: Strong churn indicator; more than 5 calls = higher churn

Payment Delay: Longer delays linked to churn

Subscription & Contract: Premium + Annual contracts → higher retention

Gender: No significant effect

3. Modeling
Baseline Models:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Evaluation Metrics: Accuracy, Precision, Recall, ROC-AUC, MAE, RMSE, MSE

Model	                 ROC-AUC	  Recall	   MAE	     RMSE
Logistic Regression    	0.82	     98%	     0.21	     0.45
Random Forest	          0.91	     100%	     0.16	     0.36
XGBoost	                0.93	     100%	     0.14	     0.32

4. Addressing Class Imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique)

XGBoost & Random Forest maintained 100% recall on balanced data

💡 Key Insights & Recommendations
Support Calls: High churn indicator → improve first-call resolution

Age Group 40–50: Most loyal → offer rewards

Payment Delay: Correlates with churn → send payment reminders

Model Deployment: Use XGBoost in production for real-time churn prediction

🚀 Tech Stack
Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn)

Jupyter Notebook – Interactive development

SMOTE – Class balancing

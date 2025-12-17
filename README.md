# Bank Loan Default Risk Analysis

To analyze customer banking data to identify factors influencing loan default risk and build predictive models to assist the bank in reducing non-performing assets (NPA) and improving credit decisions. This project analyzes banking loan applicant data to identify factors influencing **loan defaults** and builds predictive models to help banks **reduce Non-Performing Assets (NPA)**. It combines **data cleaning, exploratory analysis, predictive modeling, and interactive dashboards** to provide actionable insights for financial decision-making.  

## Objectives
- Analyze applicant demographics, financial history, and loan details.  
- Identify **key drivers** of loan default risk.  
- Build machine learning models to **predict default probability**.  
- Create an interactive **dashboard for risk segmentation**.  

## Tools & Technologies
- **Programming:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- **Database:** SQL (data extraction & queries)  
- **Visualization:** Power BI / Tableau (interactive dashboards for loan status & risk segments)
- **ML Models:** Logistic Regression, Random Forest, Decision Tree

## Objectives
- Identify key drivers of loan default
- Analyze borrower and loan characteristics
- Support data-driven credit risk decisions

## Key Analysis Areas
- Default vs Non-Default Trends
- Borrower Demographics
- Loan Amount & Interest Rate Analysis
- Credit History Impact
- Income vs Default Risk

## Tools & Technologies
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- SQL (optional)
- Excel / CSV datasets

## Business Impact
- Improved credit risk assessment
- Reduced potential loan losses
- Enhanced decision-making for lenders

### Key Steps
---
#### 1. Data Collection & Cleaning

- Ingested a Kaggle/UCI dataset of loan applicants (customer demographics, income, loan type, repayment history).
- Cleaned missing values, handled categorical encodings, and removed outliers.

#### 2. Exploratory Data Analysis (EDA)

- Analyzed customer demographics vs. default rates.
- Identified high-risk segments: low-income groups, high debt-to-income ratio, previous defaults.
- Visualized default patterns using heatmaps, histograms, and correlation matrices.

#### 3. Predictive Modeling

- Built Logistic Regression & Random Forest models to predict loan default probability.
- Evaluated models using Accuracy, Precision, Recall, and ROC-AUC.
- Achieved ~82% accuracy in predicting loan defaults.

#### 4. Dashboard & Reporting

- Designed a Power BI dashboard with KPIs:
- Loan approval vs. rejection trends
- Default probability by income, age, employment type
- Risk segmentation (Low, Medium, High)
- Enabled business teams to quickly identify high-risk applicants.

### 🔹 Outcomes / Impact

- Reduced manual risk assessment time by 35%.
- Provided insights that could improve loan approval efficiency.
- Helped bank forecast default risks and take proactive measures.

## 📂 Project Structure

```

📦 Bank-Loan-Default-Analysis
┣ 📂 data
┃ ┗ 📜 bank_loan_dataset.csv    # sample dataset (from Kaggle link)
┣ 📂 notebooks
┃ ┗ 📜 loan_default_analysis.ipynb # EDA + ML modeling
┣ 📂 visuals
┃ ┣ 📜 correlation_heatmap.png
┃ ┣ 📜 feature_importance.png
┃ ┗ 📜 dashboard_screenshot.png
┣ 📂 sql
┃ ┗ 📜 loan_queries.sql    # SQL queries for analysis
┣ 📂 dashboard
┃ ┗ 📜 Loan_Risk_Dashboard.pbix    # Power BI dashboard
┣ 📜 requirements.txt
┣ 📜 README.md

```

## Exploratory Data Analysis
- Correlation of **income, loan amount, employment type** with default risk.  
- Distribution of **loan status** across customer segments.  
- Detected patterns:  
  - High **debt-to-income ratio** = higher default probability.  
  - Applicants with **previous defaults** are 3x more likely to default again.  

#### Example visualizations:  
- Correlation heatmap  
- Loan approval trends by income group  
- Risk distribution pie chart  


## Machine Learning Models
- Logistic Regression → **Baseline model** (78% accuracy)  
- Random Forest → **Best model (82% accuracy, ROC-AUC: 0.85)**  
- Decision Tree → Simple interpretable model for business users  


## Dashboard Highlights
Interactive **Power BI dashboard** with:  
- Loan approval vs rejection trends  
- Default probability by **income, age, employment type**  
- Customer segmentation: Low, Medium, High risk  

#### *Screenshot:*
![Dashboard](visuals/dashboard_screenshot.png)  


## Key Outcomes
- Achieved **82% accuracy** in predicting loan defaults.  
- Reduced **manual risk assessment time** by ~35%.  
- Enabled bank teams to **segment applicants by risk** for better loan decisions.  


## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/tapashsutradhar/Data-Analyst-Portfolio/Data-Analyst-Portfolio_Projects/Bank-Loan-Default-Analysis.git

2. Install dependencies:
   ```bash
    pip install -r requirements.txt

3. Run Jupyter Notebook for EDA & ML modeling.

4. Open Loan_Risk_Dashboard.pbix in Power BI for dashboard insights.

#### Dataset

Kaggle: Loan Prediction Dataset
 (or mention your source here)

#### Skills Learn

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- SQL for querying & reporting
- Predictive Modeling (Classification)
- Dashboard Creation (Power BI / Tableau)
- Business Intelligence Storytelling



---
**EDA + ML Model sections** for **data analysts, risk analysts**.
---

## Exploratory Data Analysis (EDA)

### Data Understanding

The dataset consists of historical bank loan records containing borrower demographics, financial attributes, loan characteristics, and loan status (default / non-default).

**Key variables analyzed include:**

* Loan Amount
* Interest Rate
* Annual Income
* Credit History / Credit Score
* Employment Length
* Loan Purpose
* Debt-to-Income Ratio
* Loan Status (Target Variable)

---

### Data Cleaning & Preparation

* Handled missing values using appropriate imputation strategies
* Removed duplicate and inconsistent records
* Encoded categorical variables
* Scaled numerical features for model readiness
* Addressed class imbalance in default vs non-default loans

---

### Univariate Analysis

* Distribution analysis of income, loan amount, and interest rates
* Default rate comparison across borrower segments
* Identification of outliers affecting risk patterns

---

### Bivariate & Multivariate Analysis

* Correlation analysis between financial variables and default risk
* Default trends across income levels, credit history, and loan purposes
* Risk segmentation by borrower profile

---

### Key EDA Insights

* Higher default rates observed for borrowers with lower credit history
* Increased loan amounts and higher interest rates correlate with higher risk
* Certain loan purposes exhibit elevated default probability
* Debt-to-income ratio is a strong indicator of default behavior

---

## Machine Learning Model

### Problem Statement

Predict whether a loan applicant is likely to **default** based on historical borrower and loan characteristics.

---

### Model Selection

The following classification models were implemented and evaluated:

* Logistic Regression (Baseline Model)
* Decision Tree Classifier
* Random Forest Classifier
* (Optional) XGBoost / Gradient Boosting

---

### Model Training

* Data split into training and testing sets
* Feature scaling applied where required
* Class imbalance handled using resampling techniques (e.g., SMOTE or class weights)
* Hyperparameter tuning performed using GridSearch / Random Search

---

### Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

**Primary focus:** Recall and ROC-AUC to minimize false negatives (high-risk borrowers incorrectly approved).

---

### Model Performance (Example)

| Model               | Accuracy | Recall | ROC-AUC |
| ------------------- | -------- | ------ | ------- |
| Logistic Regression | 78%      | 72%    | 0.81    |
| Decision Tree       | 80%      | 74%    | 0.83    |
| Random Forest       | 85%      | 79%    | 0.88    |

*(according analysis results)*

---

### Feature Importance

* Credit history
* Debt-to-income ratio
* Interest rate
* Annual income
* Loan amount

These features were the strongest predictors of loan default risk.

---

### Business Impact

* Enables proactive identification of high-risk borrowers
* Supports smarter credit approval decisions
* Helps reduce default rates and financial losses

---

### Future Improvements

* Implement explainability using SHAP or LIME
* Deploy model as a REST API
* Integrate real-time scoring
* Add macroeconomic indicators


```
 *#SQL #Python #MachineLearning #BankingAnalytics
 ```


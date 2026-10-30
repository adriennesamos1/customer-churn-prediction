# Customer Churn Prediction

Logistic regression model to predict telecom customer churn.

## Project Overview

Built a complete machine learning pipeline using the Telco Customer Churn dataset (7,043 records). The goal is to identify customers at risk of leaving so the business can intervene before they go.

## Key Findings from EDA

- Contract type: month-to-month customers churn at the highest rate.
- Internet service: fiber optic customers churn more than DSL or no-internet customers.
- Tenure: new customers with short tenure are the highest-risk group.
- Pricing: higher monthly charges correlate with churn.

## Model Performance

- Accuracy: 0.8038
- Precision (churn): 0.65
- Recall (churn): 0.57
- F1-score (churn): 0.61

The model is better at identifying non-churners than churners. Recall for churn is 57%, so it misses about 43% of actual churners. That is a known trade-off with the default 0.5 threshold.

## Model Interpretation

Top drivers from the logistic regression coefficients:

- Tenure (-1.35): longer tenure strongly reduces churn risk.
- Two-year contract (-0.79): long contracts are protective.
- TotalCharges (+0.64): likely a collinearity effect with tenure, since tenure is already in the model.
- DSL internet (-0.61): DSL customers churn less than fiber optic customers.
- Month-to-month contract (+0.60): increases churn risk.
- Fiber optic internet (+0.56): associated with higher churn.

## Installation

    pip install kagglehub pandas numpy scikit-learn seaborn matplotlib joblib

## Usage

Run the notebook to train and save the model. To predict on new data:

    import joblib
    model = joblib.load('customer_churn_model.pkl')
    # new_customer_df must have the same 19 feature columns used in training
    prediction = model.predict(new_customer_df)

## Dependencies

- Python 3
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- joblib
- kagglehub

## Business Recommendations

- Target month-to-month customers with incentives to switch to annual contracts.
- Investigate the fiber optic experience. High churn there suggests a service or pricing problem.
- Focus retention on the first 6 to 12 months of tenure, where churn concentrates.
- Consider lowering the decision threshold if the business wants to catch more potential churners. That trades precision for recall.

## Deployment Considerations

- The model was trained on the 2020-era Telco dataset. Customer behavior and pricing change over time, so it needs periodic retraining.
- Input data must match the training schema exactly. Missing or extra columns will break prediction.
- Churn probability should be used to rank customers by risk, not as a hard yes/no filter.
- Monitor for data drift. If the churn rate on incoming customers shifts significantly from the original ~26%, the model may be stale.

## Repository Contents

- `Samos_Adrienne_IN503_Unit10_Assignment.ipynb`: notebook with full analysis and pipeline
- `customer_churn_model.pkl`: trained model, loadable with joblib

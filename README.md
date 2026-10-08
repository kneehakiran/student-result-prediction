# Student Result Prediction System

A Streamlit web app that predicts whether a student will pass or fail based on study habits, using Logistic Regression.

## What it does
- **Predict Result:** enter study hours, attendance, previous marks, assignments done and sleep hours, then get a pass/fail prediction with a confidence score.
- **Data Analysis:** charts showing how the features relate to marks, the pass/fail split and the marks distribution.
- **Model Performance:** accuracy, precision, recall, F1 score, confusion matrix and feature importance.

## Tech used
Python, Streamlit, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## About the data and model
- The dataset is **synthetic**: 500 rows generated in code, and the pass/fail label comes from a marks formula plus random noise. It is not real student data.
- Features are scaled with StandardScaler, then split 80/20 into train and test sets.
- Logistic Regression reaches about 96% accuracy on the 100-sample test set. The classes are imbalanced (about 91% pass), so accuracy alone overstates how good the model is.

## Run it locally
```
pip install -r requirements.txt
streamlit run result_prediction.py
```

## Limitations
- Trained on synthetic data, so results do not apply to real students.
- Small test set and imbalanced classes. Next steps would be real data and cross-validation.

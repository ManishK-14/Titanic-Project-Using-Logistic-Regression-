# 🚢 Titanic Survival Prediction using Logistic Regression

This repository contains a machine learning project that predicts the **survival chances of passengers** on the Titanic using **Logistic Regression**. The model not only predicts survival but also provides **insights into which features affect survival probability**, making it both a predictive and interpretive tool.

---

## 📑 Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Key Features](#key-features)  
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)  
- [Modeling](#modeling)  
- [Visualizations](#visualizations)  
- [Results](#results)  
- [Required Packages](#required-packages)  
- [Usage](#usage)  
- [Future Improvements](#future-improvements)  
- [References](#references)  
- [Author](#author)  

---

## 📝 Project Overview
The Titanic dataset is a classic dataset used to practice machine learning and predictive analytics. The main goals of this project are to:

1. Predict whether a passenger survived (`Survived`) or not.  
2. Provide **insights into survival chances** by analyzing feature contributions.  
3. Demonstrate the effect of feature engineering and hyperparameter tuning on model performance.

---

## 📂 Dataset
- **Source:** Kaggle Titanic Dataset ([Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic))  
- **Files used:**  
  - `train.csv` – Contains passenger information along with survival status.  
  - `test.csv` – Contains passenger information without survival status (used for prediction).  

### Columns of interest:
- `Pclass` – Passenger class (1, 2, 3)  
- `Sex` – Male or Female  
- `Age` – Age in years  
- `SibSp` – Number of siblings/spouses aboard  
- `Parch` – Number of parents/children aboard  
- `Fare` – Ticket fare  
- `Embarked` – Port of embarkation (C, Q, S)  
- `Cabin` – Cabin number  
- `Name` – Passenger name (used for feature engineering)  

---

## 🌟 Key Features
- `Sex` – Encoded as 0 (male) and 1 (female)  
- `FamilySize` – Computed as `SibSp + Parch + 1`  
- `Title` – Extracted from passenger name (e.g., Mr, Mrs, Miss, Master, Rare titles)  
- One-hot encoded features: `Embarked`, `Cabin`, `Title`  

---

## ⚙️ Preprocessing & Feature Engineering
1. **Handling Missing Values:**
   - Age → filled with median  
   - Fare → filled with median (test set only)  
   - Cabin → filled with mode  
   - Embarked → filled with mode  

2. **Encoding Categorical Variables:**
   - `Sex` → mapped to 0/1  
   - `Embarked`, `Cabin`, `Title` → one-hot encoding with `drop_first=True`  

3. **Feature Engineering:**
   - `FamilySize` combines `SibSp` and `Parch`  
   - `Title` extracted from passenger name; rare titles grouped as 'Rare'  

4. **Feature Scaling:**
   - `Age` and `Fare` scaled using `StandardScaler`  

---

## 🤖 Modeling
- **Algorithm:** Logistic Regression  
- **Hyperparameter Tuning:** `GridSearchCV` used to find the best `C` (regularization strength) with 5-fold cross-validation.  
- **Training Data:** Full `train.csv`  
- **Predictions:** On `test.csv`, saved as `titanic_predictions.csv`  

**Best Hyperparameters Found:**  

---

## 📊 Visualizations
1. **ROC Curve:** Evaluates model’s ability to distinguish survivors vs non-survivors  
2. **Feature Importance:** Logistic Regression coefficients visualized to understand feature contribution  
3. **Actual vs Predicted (Training):** Scatter plot comparing actual labels with predictions  
4. **Predicted Probability vs Fare (Test):** Shows how survival probability varies with fare  

---

## 🏆 Results
**Training Accuracy:** 0.879 (~88%)  

**Confusion Matrix (Training):**
[[499 50]
[ 58 284]]


**Classification Report (Training):**
          precision    recall  f1-score   support
       0       0.90      0.91      0.90       549
       1       0.85      0.83      0.84       342
accuracy                           0.88       891

   macro avg       0.87      0.87      0.87       891
weighted avg       0.88      0.88      0.88       891



- Improved recall for survivors (class 1) after feature engineering and hyperparameter tuning.  
- Model is interpretable, showing which features most influence survival probability.

---

## 📦 Required Packages

- ![Python](https://img.shields.io/badge/Python-3.x-blue)  
- ![NumPy](https://img.shields.io/badge/NumPy-1.25.0-orange)  
- ![Pandas](https://img.shields.io/badge/Pandas-1.7.0-green)  
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.0-red)  
- ![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-purple)  
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-yellow)

---

## 🚀 Usage

1. Install the required packages (see Required Packages section).  

2. Run the Python file in your terminal or command prompt:
python titanic_project.py 

## Outputs:

Training metrics: Accuracy, Confusion Matrix, and Classification Report printed in console.

Predictions: titanic_predictions.csv – predicted survival for the test set.

Visualizations: Feature importance, ROC curve, Actual vs Predicted scatter plot, and Predicted Probability vs Fare plots displayed.

## 🔮 Future Improvements

Try other models: Random Forest, XGBoost, or SVM for better accuracy

Use cross-validation metrics on training set to ensure generalization

Add interactive dashboard for visualizing survival probabilities

## 📚 References

Kaggle Titanic Competition
(https://www.kaggle.com/competitions/titanic)

Scikit-learn Logistic Regression Documentation
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

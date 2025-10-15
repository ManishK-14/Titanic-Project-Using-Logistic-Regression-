import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

train_data = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)

print(train_data.head())
print(train_data.describe())
print(train_data.isnull().sum())

print(train_data['Age'].value_counts())

# Handling the  Missing Values
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Cabin'] = train_data['Cabin'].fillna(train_data['Cabin'].mode()[0])
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Age'] = test_data['Age'].fillna(train_data['Age'].median()) 
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())
test_data['Cabin'] = test_data['Cabin'].fillna(train_data['Cabin'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

print('-------------------------------')
print('After Handling Missing Values (Train):')
print(train_data.describe())
print(train_data.isnull().sum())


# Encode Categorical Variables
#For using logistic regression i need to make the categorical variables -> numerical
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

#For Encode Embarked column using one-hot encoding
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# Converts categorical column Embarked → numeric columns for ML
## drop_first=True ensures no redundancy for regression models.
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])
train_data = pd.get_dummies(train_data, columns=['Cabin'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Cabin'], drop_first=True)

# Feature Engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# New Feature Engineering: Extract Title from Name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles for better generalization
for data in [train_data, test_data]:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
# Encode Title column using one-hot encoding
train_data = pd.get_dummies(train_data, columns=['Title'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Title'], drop_first=True)

print(train_data.head())

# Selecting my features and target

train_data.reset_index(inplace=True) 
test_data.reset_index(inplace=True)

x_train = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y_train = train_data['Survived']

x_test = test_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# Ensure train and test have same columns after new feature engineering
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

# Feature Scaling
num_cols = ['Age', 'Fare']
scaler = StandardScaler()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])
#                                       #
# Training my Logistic Regression Model #
#                                       #
# 1. Hyperparameter Tuning using Grid Search for Logistic Regression
logreg_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]} # C is the inverse of regularization strength
grid_search_logreg = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy') 
grid_search_logreg.fit(x_train, y_train)

# The best model from the grid search is now used for 'model'
model = grid_search_logreg.best_estimator_ 
print("Best Logistic Regression Parameters:", grid_search_logreg.best_params_)

# 2. Final model selection: Using the tuned Logistic Regression model
final_model = model 

# Predictions
y_pred_train = final_model.predict(x_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Training Classification Report:\n", classification_report(y_train, y_pred_train))

# Predict on test.csv
y_test_pred = final_model.predict(x_test)
y_test_prob = final_model.predict_proba(x_test)[:,1]

# Save predictions
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_test_pred})
submission.to_csv('titanic_predictions.csv', index=False)
print("Predictions saved to titanic_predictions.csv")

# Visualizations
# ROC Curve (using training data as example)
y_prob_train = final_model.predict_proba(x_train)[:,1] 
fpr, tpr, thresholds = roc_curve(y_train, y_prob_train)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Actual vs Predicted (training data)
plt.figure(figsize=(10,5))
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(y_train)), y_pred_train, color='red', label='Predicted', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Survived')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# Predicted probability vs Fare (test data)
plt.figure(figsize=(10,5))
plt.scatter(x_test['Fare'], y_test_prob, color='red', alpha=0.6, label='Predicted Probability')
plt.xlabel('Fare')
plt.ylabel('Survival Probability')
plt.title('Prediction Probabilities vs Fare (Test Data)') 
plt.legend()
plt.show()


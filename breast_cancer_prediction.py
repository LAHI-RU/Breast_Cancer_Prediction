
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt


data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print(df.head())
print(df.isnull().sum())

sns.countplot(df['target'])
plt.title("Distribution of Malignant (1) and Benign (0) Tumors")
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=False)
plt.title("Feature Correlations")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()

# Initialize the model for feature selection
selector = RFE(LogisticRegression(), n_features_to_select=10)  # Select top 10 features (adjustable)
selector.fit(X_train, y_train)

# Transform the training and test sets to keep only selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Print selected features
selected_features = data.feature_names[selector.support_]
print("Selected Features:", selected_features)


cross_val_scores = cross_val_score(model, X_train, y_train, cv=10)
print("Cross-Validation Accuracy Scores:", cross_val_scores)
print("Average Cross-Validation Accuracy:", cross_val_scores.mean())

model.fit(X_train_selected, y_train)

y_pred = model.predict(X_test_selected)


initial_accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy: {initial_accuracy * 100:.2f}%")


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title("Confusion Matrix - Initial Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("Classification Report - Initial Model:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))


param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}


grid_search = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred_best = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Best Model Accuracy after Grid Search: {best_accuracy * 100:.2f}%")

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix_best, annot=True, fmt="d", cmap="Greens", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title("Confusion Matrix - Best Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("Classification Report - Best Model:")
print(classification_report(y_test, y_pred_best, target_names=['Benign', 'Malignant']))

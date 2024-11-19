import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
generation_data = pd.read_csv("C:\\Users\\Hayfa\\Downloads\\Plant_2_Generation_Data.csv\\Plant_2_Generation_Data.csv")
weather_data = pd.read_csv("C:\\Users\\Hayfa\\Downloads\\Plant_2_Weather_Sensor_Data.csv")

# Merge datasets on 'DATE_TIME' column
df_solar = pd.merge(
    generation_data.drop(columns=['PLANT_ID']),
    weather_data.drop(columns=['PLANT_ID', 'SOURCE_KEY']),
    on='DATE_TIME'
)

# Create a sample dataset (replace this with your own dataset)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=20, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost model setup
model = CatBoostClassifier(
    iterations=2000,
    depth=10,
    learning_rate=0.05,
    l2_leaf_reg=3,
    bootstrap_type='Bayesian',
    bagging_temperature=0.8,
    random_seed=42,
    class_weights={0: 1, 1: 1.2},
    task_type='GPU',
    verbose=200,
    early_stopping_rounds=200,
)

# Fit the model on the training data
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("CatBoost Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

y_pred_prob = model.predict_proba(X_test)[:, 1]
mse = mean_squared_error(y_test, y_pred_prob)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred_prob))
r2 = r2_score(y_test, y_pred_prob)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Feature importance visualization
feature_importance = model.get_feature_importance(Pool(X_train, y_train))
sns.barplot(x=feature_importance, y=[f"Feature {i}" for i in range(X_train.shape[1])])
plt.title('Feature Importance')
plt.show()

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

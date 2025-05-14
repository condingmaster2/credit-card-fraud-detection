import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Generate dummy fraud data
np.random.seed(42)
categories = ['Food', 'Bills', 'Personal', 'Entertainment']
data = []

for _ in range(1000):
    category = np.random.choice(categories)
    amount = np.random.uniform(10, 1000)
    fraud = 1 if (category == 'Entertainment' and amount > 800) else 0
    data.append([category, amount, fraud])

df = pd.DataFrame(data, columns=['Category', 'Amount', 'Class'])

# Step 2: Preprocess the data
X = pd.get_dummies(df['Category'])
X['Amount'] = df['Amount']
X.columns = [str(col) for col in X.columns]  # Ensure all column names are strings
y = df['Class']

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Step 4: Convert to DMatrix (optimized XGBoost input)
dtrain = xgb.DMatrix(X_train, label=y_train)

# Step 5: Set parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'eta': 0.1
}

# Step 6: Train the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Step 7: Save the model
joblib.dump(model, 'fraud_model_xgb.pkl')

print("✅ XGBoost model trained and saved.")

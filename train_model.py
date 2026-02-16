import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Starting Model Training Pipeline...")

# 1. Load Data
# Assuming the file is in a 'data' folder or same directory
csv_path = "project_risk_raw_dataset.csv" 
if not os.path.exists(csv_path):
    # Fallback for demonstration if file not found in root
    csv_path = "data/project_risk_raw_dataset.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"Data Loaded: {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find 'project_risk_raw_dataset.csv'. Please make sure it exists.")
    exit()

# 2. EDA & Cleaning
print("Cleaning Data...")

# Missing Values
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Cleaning Strings
if 'Risk_Level' in df.columns:
    df['Risk_Level'] = df['Risk_Level'].str.lower().str.strip()
    
    # Create Target
    df['Risk_Binary'] = df['Risk_Level'].apply(
        lambda x: 1 if x in ['high','critical'] else 0
    )
    df.drop(columns=['Risk_Level'], inplace=True)

if 'Project_ID' in df.columns:
    df.drop(columns=['Project_ID'], inplace=True)

# 3. Feature Engineering
print("Engineering Features...")

# Requirement Stability Mapping
if 'Requirement_Stability' in df.columns:
    df['Requirement_Stability'] = df['Requirement_Stability'].map({
        'Low': 1, 'Medium': 2, 'High': 3
    }).fillna(2) # Fill NaN with Medium (2) just in case

# Calculated Features
# Note: Adding small epsilon 1e-9 to avoid division by zero
df['Budget_per_Month'] = df['Project_Budget_USD'] / (df['Estimated_Timeline_Months'] + 1e-9)
df['Workload_Index'] = df['Complexity_Score'] / (df['Team_Size'] + 1e-9)

df['Experience_Buffer'] = (
    df['Past_Similar_Projects'] *
    df['Previous_Delivery_Success_Rate']
)

df['Org_Strength_Index'] = (
    df['Vendor_Reliability_Score'] +
    df['Resource_Availability'] +
    df['Previous_Delivery_Success_Rate']
) / 3

df['Pressure_x_Complexity'] = (
    df['Schedule_Pressure'] *
    df['Complexity_Score']
)
df['Risk_Intensity'] = (
    df['Schedule_Pressure'] *
    df['Budget_Utilization_Rate']
)

df['Team_Pressure'] = (
    df['Workload_Index'] *
    df['Schedule_Pressure']
)

df['Log_Budget'] = np.log1p(df['Project_Budget_USD'])

# Cleanup Infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# 4. One Hot Encoding
df_model = pd.get_dummies(df, drop_first=True)

# 5. Split Data
X = df_model.drop(columns=['Risk_Binary'])
y = df_model['Risk_Binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Model Training
print("Training Models (This might take a minute)...")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight={0:1, 1:1.5},
    random_state=42
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Stacking
stack_model = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stack_model.fit(X_train, y_train)

# Evaluation
y_pred = stack_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Training Complete into 'stack_model'. Accuracy: {acc:.4f}")

# 7. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save Artifacts
print("Saving Model & Columns...")
joblib.dump(stack_model, "models/risk_model.pkl")

# CRITICAL: Save the column names to ensure the App has the exact same structure
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, "models/train_columns.pkl")

print("DONE! You can now run the app.")

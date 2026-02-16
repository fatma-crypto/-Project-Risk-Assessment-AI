import pandas as pd
import numpy as np

# Load Data
try:
    df = pd.read_csv("data/project_risk_raw_dataset.csv")
except:
    df = pd.read_csv("project_risk_raw_dataset.csv")

# Clean 'Risk_Level' to see balance
df['Risk_Level'] = df['Risk_Level'].str.lower().str.strip()
print("Distribution of Target:")
print(df['Risk_Level'].value_counts())

# Statistics for numeric columns
print("\n Numeric Stats:")
print(df.describe().T[['min', 'mean', '50%', 'max']])

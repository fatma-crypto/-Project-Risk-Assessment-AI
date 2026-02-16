import pandas as pd

try:
    df = pd.read_csv("data/project_risk_raw_dataset.csv")
except:
    df = pd.read_csv("project_risk_raw_dataset.csv")

missing = df.isnull().sum()
print("Missing Values per Column:")
print(missing[missing > 0])
if missing.sum() == 0:
    print("NO missing values found.")

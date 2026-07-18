import pandas as pd

# Load Dataset
df = pd.read_csv("../Dataset/Titanic-Dataset.csv")

print("=" * 60)
print("TASK 1 : DATA COLLECTION")
print("=" * 60)

print("\nFirst 5 Records")
print(df.head())

print("\nLast 5 Records")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

print("\nDataset Information")
df.info()

print("\nTask 1 Completed Successfully")
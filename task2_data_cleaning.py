import pandas as pd

# Load the dataset
df = pd.read_csv("../Dataset/Titanic-Dataset.csv")

print("=" * 60)
print("TASK 2 : DATA CLEANING")
print("=" * 60)

# Dataset shape before cleaning
print("\nDataset Shape Before Cleaning:")
print(df.shape)

# Check duplicate rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Remove duplicate rows
df = df.drop_duplicates()

# Missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the most frequent value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Cabin values with "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Dataset shape after cleaning
print("\nDataset Shape After Cleaning:")
print(df.shape)

# Save cleaned dataset
df.to_csv("../Dataset/Titanic_Cleaned.csv", index=False)

print("\nCleaned dataset saved as Titanic_Cleaned.csv")
print("\nTask 2 Completed Successfully!")
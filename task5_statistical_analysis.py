import pandas as pd
from scipy.stats import ttest_ind

# Load transformed dataset
df = pd.read_csv("../Dataset/Titanic_Transformed.csv")

print("=" * 60)
print("TASK 5 : STATISTICAL ANALYSIS")
print("=" * 60)

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Mean Fare
print("\nMean Fare:", round(df["Fare"].mean(), 2))

# Median Age
print("Median Age:", df["Age"].median())

# Survival Count
print("\nSurvival Count:")
print(df["Survived"].value_counts())

# T-Test: Fare of survivors vs non-survivors
survived = df[df["Survived"] == 1]["Fare"]
not_survived = df[df["Survived"] == 0]["Fare"]

t_stat, p_value = ttest_ind(survived, not_survived)

print("\nT-Test Results")
print("T-Statistic:", round(t_stat, 4))
print("P-Value:", round(p_value, 6))

if p_value < 0.05:
    print("\nResult: The difference in fares is statistically significant.")
else:
    print("\nResult: The difference in fares is NOT statistically significant.")

print("\nTask 5 Completed Successfully!")
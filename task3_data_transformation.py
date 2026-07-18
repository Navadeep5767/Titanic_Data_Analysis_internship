import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("../Dataset/Titanic_Cleaned.csv")

print("=" * 60)
print("TASK 3 : DATA TRANSFORMATION")
print("=" * 60)

# Display original data types
print("\nOriginal Data Types:")
print(df.dtypes)

# Convert Age to integer
df["Age"] = df["Age"].astype(int)

# Create a FamilySize column
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create an IsAlone column
df["IsAlone"] = df["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

# Create a FareCategory column
df["FareCategory"] = pd.cut(
    df["Fare"],
    bins=[0, 10, 30, 100, 600],
    labels=["Low", "Medium", "High", "Very High"],
    include_lowest=True
)

print("\nUpdated Data Types:")
print(df.dtypes)

print("\nFirst 10 Records:")
print(df.head(10))

# Save transformed dataset
df.to_csv("../Dataset/Titanic_Transformed.csv", index=False)

print("\nTransformed dataset saved as Titanic_Transformed.csv")
print("\nTask 3 Completed Successfully!")
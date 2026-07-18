import pandas as pd
import matplotlib.pyplot as plt

# Load transformed dataset
df = pd.read_csv("../Dataset/Titanic_Transformed.csv")

print("=" * 60)
print("TASK 4 : EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Create output folder if it doesn't exist
import os
os.makedirs("../Output", exist_ok=True)

# 1. Survival Count
plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("../Output/survival_count.png")
plt.close()

# 2. Passenger Class Distribution
plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Passengers")
plt.tight_layout()
plt.savefig("../Output/passenger_class.png")
plt.close()

# 3. Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../Output/age_distribution.png")
plt.close()

# 4. Gender Distribution
plt.figure(figsize=(6,4))
df["Sex"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Gender Distribution")
plt.tight_layout()
plt.savefig("../Output/gender_distribution.png")
plt.close()

print("\nEDA Completed Successfully!")
print("Charts saved inside the Output folder.")
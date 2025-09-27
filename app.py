import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Preview of DataSet")
print(df.head())

print("\nDataSet Info")
print(df.info())

print("\nMissing Values Per Column")
print(df.isnull().sum())

# Summary statistics for numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean for each feature
print("\nMean values per species:")
print(df.groupby("species").mean())

# Task 2
print("\nBasic Statistics")
print(df.describe())

print("\nMean Values Per Species")
print(df.groupby("species").mean())

# Task 3
df["petal length (cm)"].plot(kind="line" ,figsize=(8,5))
plt.title("Petal Length Over Samples")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.show()

# Bar Chart
df.groupby("species")["petal length (cm)"].mean().plot(kind="bar", color=["skyblue", "orange", "green"])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram
df["sepal length (cm)"].plot(kind="hist",bins=20,color="purple",alpha=0.7)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.show()

# Scatter Plot
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species",data=df )
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
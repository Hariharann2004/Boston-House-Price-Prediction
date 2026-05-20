import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("boston.csv")

# Display first rows
print("FIRST 5 ROWS:\n")
print(df.head())

# Check missing values
print("\nMISSING VALUES:\n")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Statistical summary
print("\nDATA DESCRIPTION:\n")
print(df.describe())

# Separate features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nSTANDARDIZED DATA:\n")
print(df_scaled.head())

# Normalization
normalizer = MinMaxScaler()

X_normalized = normalizer.fit_transform(X)

df_normalized = pd.DataFrame(X_normalized, columns=X.columns)

print("\nNORMALIZED DATA:\n")
print(df_normalized.head())

# Correlation matrix
correlation_matrix = df.corr()

print("\nCORRELATION MATRIX:\n")
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Correlation Matrix Heatmap")
plt.show()

# Scatterplot RM vs MEDV
plt.figure(figsize=(8,6))

sns.scatterplot(x=df['RM'], y=df['MEDV'])

plt.title("RM vs MEDV")
plt.xlabel("Average Rooms")
plt.ylabel("House Price")

plt.show()

# Scatterplot LSTAT vs MEDV
plt.figure(figsize=(8,6))

sns.scatterplot(x=df['LSTAT'], y=df['MEDV'])

plt.title("LSTAT vs MEDV")
plt.xlabel("Lower Status Population")
plt.ylabel("House Price")

plt.show()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN TEST SPLIT:")

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# PCA
pca = PCA(n_components=5)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA REDUCED SHAPE:")
print(X_pca.shape)

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")
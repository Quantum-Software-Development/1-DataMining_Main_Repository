

# Data Mining Pre-Processing Notebook

This notebook guides you through essential data pre-processing steps applied to datasets similar to those from the UCI Machine Learning Repository, such as Bank Marketing or Breast Cancer datasets. The steps cover cleaning, handling missing values, normalizing, discretizing, and preparing data for mining.

---

## Import Required Libraries

```


# Import libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale, scale

```

---

## Load Dataset

```


# Load dataset from local or online source

# Example: Breast Cancer Wisconsin dataset (adjust path or URL as needed)

url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/cancer.csv'  \# Example URL
dados = pd.read_csv(url)

# Show first rows to inspect data

dados.head()

```

---

## Inspect Columns and Data Types

```


# List columns and data types

print(dados.dtypes)

# Check for 'Unnamed' columns (common from CSV exports)

unnamed_cols = [col for col in dados.columns if "Unnamed" in col]
print("Unnamed columns:", unnamed_cols)

```

---

## Drop Unnamed Columns

```


# Drop unnamed columns if any

dados.drop(columns=unnamed_cols, inplace=True)
print("Columns after dropping unnamed ones:", dados.columns)

```

---

## Handling Missing Values

```


# Check for missing values per column

print(dados.isnull().sum())

# Example: Impute missing values using mean for numerical columns

num_cols = dados.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in num_cols:
mean_value = dados[col].mean()
dados[col].fillna(mean_value, inplace=True)

# For categorical columns, impute mode if needed

cat_cols = dados.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
mode_value = dados[col].mode()
dados[col].fillna(mode_value, inplace=True)

# Verify no missing values left

print(dados.isnull().sum())

```

---

## Remove Rows with Missing Data (Alternative approach)

```


# Alternatively, remove rows with any missing value

dados_clean = dados.dropna()
print(f"Data shape after dropping missing values: {dados_clean.shape}")

```

---

## Normalization: Max-Min Scaling

```


# Select numeric columns for normalization, excluding identifiers or labels

cols = list(dados.columns)

# Example: remove columns if they are IDs or target variables

if 'id' in cols: cols.remove('id')
if 'diagnosis' in cols: cols.remove('diagnosis')

# Apply Min-Max normalization to selected columns

dados_minmax = dados.copy()
dados_minmax[cols] = dados_minmax[cols].apply(minmax_scale)

dados_minmax.head()

```

---

## Normalization: Z-Score Standardization

```


# Apply Z-score standardization: mean=0, std=1

dados_zscore = dados.copy()
dados_zscore[cols] = dados_zscore[cols].apply(scale)

dados_zscore.head()

```

---

## Discretization of Continuous Attributes

```


# Example: Discretizing 'radius_mean' into 10 equal-width bins with labels 0-9

if 'radius_mean' in dados.columns:
dados['radius_mean_binned'] = pd.cut(dados['radius_mean'], bins=10, labels=range(10))
print(dados[['radius_mean', 'radius_mean_binned']].head())
else:
print('Column radius_mean not found in dataset')

```

---

## Handling Categorical Variables

```


# Example: Encoding categorical columns (if needed)

categorical_cols = dados.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_cols)

# Convert categorical columns to category dtype

for col in categorical_cols:
dados[col] = dados[col].astype('category')

# Display categorical columns with their unique values

for col in categorical_cols:
print(f"{col} unique values: {dados[col].cat.categories}")

```

---

## Summary: Data Ready for Mining

```


# Final look at data shape and first few rows after pre-processing

print("Data shape:", dados.shape)
dados.head()

```


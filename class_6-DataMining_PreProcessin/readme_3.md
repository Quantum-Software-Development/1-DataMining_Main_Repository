

```python
# Cell 1 - Import libraries
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.preprocessing import minmax_scale, scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cell 2 - Download and load UCI Bank Marketing dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall('bank_data')

df = pd.read_csv('bank_data/bank-full.csv', sep=';')
print("Loaded dataset shape:", df.shape)
df.head()

# Cell 3 - Check for unnamed or unwanted columns and remove them
unnamed_cols = [col for col in df.columns if "Unnamed" in col]
print("Unnamed columns:", unnamed_cols)
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)
print("Columns after drop:", df.columns.tolist())

# Cell 4 - Explore missing values and replace 'unknown' with NaN
print("Count 'unknown' per column before replacement:")
print((df == 'unknown').sum())

df.replace('unknown', np.nan, inplace=True)  # Mark 'unknown' as NaN

print("Count missing values after replacement:")
print(df.isnull().sum())

# Cell 5 - Imputation of missing values (fill NaN for categorical columns with mode)
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    mode_val = df[col].mode()[^0]
    df[col].fillna(mode_val, inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum())

# Cell 6 - Check unique values for categorical attributes to spot inconsistencies
for col in cat_cols:
    print(f"Unique values in '{col}':")
    print(df[col].unique())
    print('------')

# Cell 7 - Binning the 'age' variable into 5 equal-width intervals
df['age_binned'] = pd.cut(df['age'], bins=5, labels=range(5))
print(df[['age', 'age_binned']].head())

# Cell 8 - Normalize numeric attributes using Max-Min scaling
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numeric columns:", num_cols)

df_minmax = df.copy()
df_minmax[num_cols] = df_minmax[num_cols].apply(minmax_scale)
print(df_minmax[num_cols].head())

# Cell 9 - Normalize numeric attributes using Z-score standardization
df_zscore = df.copy()
df_zscore[num_cols] = df_zscore[num_cols].apply(scale)
print(df_zscore[num_cols].head())

# Cell 10 - Encode categorical variables before PCA
df_encoded = pd.get_dummies(df.drop(columns=['age_binned']))  # drop binned for now
print("Shape after encoding:", df_encoded.shape)

# Cell 11 - Apply PCA for dimensionality reduction (2 components) and visualize
X = df_encoded.select_dtypes(include=[np.number])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio of PCA components:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
plt.title('PCA projection onto 2 components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```


***

### Explanation per cell:

- **Cell 1:** Imports all necessary libraries.
- **Cell 2:** Downloads and extracts the Bank Marketing dataset from UCI and loads it into a DataFrame.
- **Cell 3:** Detects and removes any extraneous unnamed columns.
- **Cell 4:** Identifies `'unknown'` values as missing and replaces with `NaN`.
- **Cell 5:** Imputes missing categorical values with the mode of each column.
- **Cell 6:** Prints unique values of categorical columns to check for inconsistencies.
- **Cell 7:** Demonstrates binning — converting continuous `age` into categorical bins.
- **Cell 8:** Applies Min-Max normalization to numeric columns.
- **Cell 9:** Applies standard Z-score normalization to numeric columns.
- **Cell 10:** One-hot encodes categorical variables preparing for PCA.
- **Cell 11:** Applies PCA to reduce dimensions and plots the result.

***

You can copy this script into Google Colab, run cell-by-cell, and get a complete data pre-processing example using a real UCI dataset.


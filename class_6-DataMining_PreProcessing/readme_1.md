
# Data Mining / Pre-Processing

Explore datasets from the **University of California Irvine (UCI) Machine Learning Repository** such as the **Balloon**, **Bank Marketing**, and **Mammogram** datasets to practice these concepts of data pre-processing and mining: https://archive.ics.uci.edu/ml/index.php

---

## Table of Contents

- [Introduction](#introduction)
- [Common Problems in Raw Data](#common-problems-in-raw-data)
  - [Incompleteness](#incompleteness)
  - [Inconsistency](#inconsistency)
  - [Noise](#noise)
- [Garbage In, Garbage Out (GIGO)](#garbage-in-garbage-out-gigo)
- [Types of Data](#types-of-data)
  - Structured, Semi-Structured, Unstructured
- [Data Attributes and Their Types](#data-attributes-and-their-types)
- [Datasets from University of California - Irvine (UCI)](#datasets-from-university-of-california---irvine-uci)
  - Balloon Dataset
  - Bank Marketing Dataset
  - Mammographic Mass Dataset
- [Steps of Data Pre-Processing](#steps-of-data-pre-processing)
  - Cleaning
  - Integration
  - Reduction
  - Transformation
  - Discretization
- [Data Cleaning Techniques](#data-cleaning-techniques)
  - Handling Missing Values
  - Noise Reduction Techniques
  - Handling Inconsistencies
- [Data Integration Issues](#data-integration-issues)
- [Data Reduction Techniques](#data-reduction-techniques)
- [Data Standardization & Normalization](#data-standardization--normalization)
- [Discretization](#discretization)
- [Python Code Examples](#python-code-examples)
- [ASCII Diagrams](#ascii-diagrams)

---

## Introduction

Real-world data are almost always incomplete, inconsistent, and noisy. These problems must be addressed via pre-processing to ensure clean, reliable data, a prerequisite for successful data mining.

The **pre-processing** step manipulates raw data into a form that enables better and more accurate knowledge extraction.

---

## Common Problems in Raw Data

### Incompleteness

Missing attribute values, records, or features.

Example: "?" in the credit card field or missing rows.

### Inconsistency

Contradictory or conflicting entries within the data, e.g., units mixing kg with lbs.

### Noise

Random variations or errors that obscure real data trends.

---

## Garbage In, Garbage Out (GIGO)

Poor quality input data produce poor quality outputs and insights. Cleaning data beforehand is critical.

---

## Types of Data

| Type            | Description                             | Examples                 |
|-----------------|---------------------------------------|--------------------------|
| Structured      | Fixed fields, clear schema             | CSV, SQL tables          |
| Semi-Structured | Partial structure with markers         | XML, JSON, Emails        |
| Unstructured    | No strict structure or schema          | Text, images, video files|

---

## Data Attributes and Their Types

| Attribute Type | Description                             | Example                     |
|----------------|---------------------------------------|-----------------------------|
| Binary         | Two possible values                    | Yes/No, 0/1                  |
| Nominal        | Categorical, no order                  | Marital Status               |
| Ordinal        | Ordered categories                    | Education Level              |
| Ratio          | Numeric with meaningful zero          | Age, Salary                  |

---

## Datasets from University of California - Irvine (UCI)

### Balloon Dataset

- 20 samples, 5 attributes: color, size, action, age, inflated (True/False).  
- Simple dataset to illustrate basic concepts.

### Bank Marketing Dataset

- 4521 samples, 17 attributes related to direct marketing campaigns.
- Predict whether client will subscribe a term deposit (`application`).

Example attributes:

| Attribute          | Type        | Description                     |
|--------------------|-------------|---------------------------------|
| age                | Numeric     | Client's age                   |
| job                | Categorical | Job type                      |
| marital            | Categorical | Marital Status                |
| education          | Categorical | Education Level               |
| credit             | Binary      | Has credit line (yes/no)      |
| balance            | Numeric     | Account balance               |
| ...                | ...         | ...                          |

### Mammographic Mass Dataset

- 960 samples, 6 attributes related to breast masses.  
- Used for predicting severity (benign/malign).

---

## Steps of Data Pre-Processing

1. **Cleaning:** Handling missing, noisy, and inconsistent data.  
2. **Integration:** Combine data from multiple sources.  
3. **Reduction:** Reduce dimensionality or data volume.  
4. **Transformation:** Normalize and format data.  
5. **Discretization:** Convert continuous attributes into categorical intervals.

---

## Data Cleaning Techniques

### Handling Missing Values

- **Remove rows** with missing data (not recommended if much data lost).  
- **Manual imputation** with domain knowledge.  
- **Global constant imputation** (e.g. zero, -1) — caution advised.  
- **Hot-deck imputation:** Use value from a similar record.  
- **Last observation carried forward:** Use previous valid value.  
- **Mean/mode imputation:** Replace missing with mean (numeric) or mode (categorical).  
- **Predictive models:** Use other attributes to infer missing values.

---

### Noise Reduction Techniques

- **Binning:** Group values into intervals (*equal width* or *equal frequency* bins). Replace each value by bin mean or bin boundaries.  
- **Clustering:** Group similar data points; replace with cluster centroid or medoid.  
- **Approximation:** Fit data to smoothing functions like polynomials.

---

### Handling Inconsistent Data

- Detect out-of-domain or conflicting values.  
- Correct with manual review or automated scripts verifying domain constraints.  
- Use visualization, statistical summaries, and input from domain experts.

---

## Data Integration Issues

- **Redundancy:** Duplicate or derivable data attributes or records.  
- **Duplicity:** Exact copies of objects or attributes.  
- **Conflicts:** Different units or representations of the same attribute.  
- Resolve by normalization and unifying units or standards.

---

## Data Reduction Techniques

- **Attribute selection:** Remove irrelevant or redundant attributes.  
- **Attribute compression:** Use methods like Principal Component Analysis (PCA).  
- **Data volume reduction:** Use sampling, clustering, or parametric models.  
- **Discretization:** Convert continuous data to intervals.

---

## Data Standardization & Normalization

**Normalization** rescales data for algorithm compatibility:

### Max-Min Normalization

Maps value \(a\) to \(a'\) in new range \([new_{min}, new_{max}]\):

\[
a' = \frac{a - min_a}{max_a - min_a} \times (new_{max} - new_{min}) + new_{min}
\]

---

### Z-Score Normalization

Centers attribute around zero and scales by standard deviation:

\[
a' = \frac{a - \bar{a}}{\sigma_a}
\]

---

## Discretization

- Converts numeric attributes into categorical bins.  
- Methods include equal-width, equal-frequency, histogram-based, and entropy-based discretization.

---

## Python Code Examples

```

import pandas as pd
from sklearn.preprocessing import minmax_scale, scale

# Load Bank Marketing dataset from UCI (use your local copy or URL)

# Example URL requires downloading and preprocessing: demonstration uses local CSV

data_path = "bank-additional-full.csv"
df = pd.read_csv(data_path, sep=';')

# Drop unnamed columns (typical from CSV exports)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handling missing values example:

# Replace '?' with NaN for easier manipulation

df.replace('?', pd.NA, inplace=True)

# Remove rows with any missing values (not recommended if too many deleted)

df_clean = df.dropna()

# Or imputing missing values with mode for categorical attribute, e.g. 'job'

df['job'].fillna(df['job'].mode(), inplace=True)

# Max-Min normalization for numeric columns

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df[num_cols] = df[num_cols].apply(minmax_scale)

# Z-score normalization example

df[num_cols] = df[num_cols].apply(scale)

# Discretization example - age into 5 bins

df['age_binned'] = pd.cut(df['age'], bins=5, labels=False)

# Drop columns example: if any 'Unnamed' columns exist

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')

```

---

## ASCII Diagrams

### Data Pre-Processing Flow

```

+-------------------+
| Define Problem    |
+---------+---------+
|
v
+-------------------+
| Select Data       |
+---------+---------+
|
v
+-------------------+     +--------------------------+
| Choose Algorithm  +---->+ Apply Preprocessing Steps |
+---------+---------+     +--------------------------+
|
v
+-------------------+
| Apply Data Mining |
+-------------------+

```

### Binning (Equal Width)

```

Values: 2, 3, 7, 8, 11, 12, 16, 18, 20
Bins: 3 (width = 6)

Bin 1: 2 - 7  -> 2, 3, 7
Bin 2: 8 - 13 -> 8, 11, 12
Bin 3: 14 - 20->16, 18, 20

```

### Clustering Prototypes: Centroid vs Medoid

```

(a) Centroid                     (b) Medoid

*                             *
    
*o*                           ooo
oooo                          o o o

Centroid = mean point (artificial)
Medoid  = actual central object

```

---

## Summary

Data pre-processing ensures the quality of mining outcomes by addressing missing, noisy, and inconsistent data; integrating multiple data sources; reducing data size and dimensionality; normalizing; and discretizing attributes for algorithm compatibility.

UCI datasets like **Balloon**, **Bank Marketing**, and **Mammographic Mass** provide excellent real-world scenarios to practice these techniques using Python.

---

*Prepared by: Data Mining Learning Group*  
*Date: September 2025*


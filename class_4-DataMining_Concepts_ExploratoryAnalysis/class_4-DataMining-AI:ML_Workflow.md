

# Data Mining Concepts and AI Project Workflow

***

## Table of Contents

- [Project Overview](#project-overview)
- [AI Project Workflow](#ai-project-workflow)
- [Installation and Requirements](#installation-and-requirements)
- [Usage Examples](#usage-examples)
- [Knowledge Discovery in Databases (KDD)](#knowledge-discovery-in-databases-kdd)
- [Mathematical Concepts](#mathematical-concepts)
- [Discrete vs Continuous Values](#discrete-vs-continuous-values)
- [Primary Libraries and Tools](#primary-libraries-and-tools)
- [Supervised vs Unsupervised Learning](#supervised-vs-unsupervised-learning)
- [Clustering and Distance Calculations](#clustering-and-distance-calculations)
- [Credit Analysis: Classification Example](#credit-analysis-classification-example)
- [Fruit Clustering Example](#fruit-clustering-example)
- [Anomaly Detection and Association Rules](#anomaly-detection-and-association-rules)
- [Applications](#applications)
- [Repository Structure and Documentation](#repository-structure-and-documentation)
- [Contributing and License](#contributing-and-license)
- [References](#references)

***

## Project Overview

This project provides a comprehensive introduction to **Data Mining and AI**, based on the workbook from PUC-SP by Prof. Dr. Daniel Rodrigues da Silva. It covers:

- Foundations of data mining and KDD
- Relation between data, information, and knowledge concepts
- Mathematical foundations (inflection points, maxima, minima)
- Overview of supervised and unsupervised learning
- Practical coding examples in Python and R
- Real-world cases such as credit classification, fruit clustering, anomaly detection, and industrial applications

***

## AI Project Workflow

Machine learning project development follows an iterative, structured process:

```
+-----------------+      +---------------------+      +----------------------+
|   Dataset/Data  | ---> |   Preprocessing     | ---> |   Cluster Training   |
+-----------------+      +---------------------+      +----------------------+
                                |                          |
                                v                          v
                      +---------------------+      +----------------------+
                      |  Cleaned/Labeled    |      |   Parallel Training  |
                      |      Data           |      +----------------------+
                      +---------------------+                |
                                        |                    v
                                        v          +----------------------+
                                  +---------------------+ | Validation &   |
                                  |     Validation      | |   Tuning       |
                                  +---------------------+ +----------------+
                                        |
                                        v
                               +---------------------+
                               |   Trained Model     |
                               +---------------------+
                                        |
                                        v
                             +------------------------+
                             | Inference in Production|
                             +------------------------+
                                        |
                                        v
                             +------------------------+
                             | Feedback (New Data)    |
                             +------------------------+
```


***

### Detailed AI Project Workflow Explanation

1. **Data Collection (Dataset)**
Everything starts with collecting the dataset used for training.
*Example:* 1 million images for a facial recognition model.
2. **Preprocessing**
Clean, standardize, and organize data.
*Example:* Resize images, remove noise, label properly. Simple machines or servers suffice.
3. **Sending to the Cluster**
Data is sent to clusters (dozens or hundreds of GPUs/CPUs) for parallel processing.
*Example:* Upload to AWS, Google Cloud, or private clusters.
4. **Training on the Cluster**
Workload is split across multiple machines for faster training.
*Example:* GPUs process parts of batches; results combined for final model.
5. **Validation and Tuning**
Test on validation subset to check accuracy; tune hyperparameters until objectives met.
6. **Inference in Production**
Deploy trained model on servers/clusters for real-time predictions.
*Example:* User uploads photo; model recognizes face in seconds.
7. **Feedback and Update**
Collect new data to retrain and improve the model continually.
*Example:* User data expands dataset for next training cycle.

***

## Installation and Requirements

### Python Setup

```bash
pip install pandas numpy scikit-learn seaborn
```

Include `requirements.txt` for ease:

```
pandas
numpy
scikit-learn
seaborn
```


### R Setup

```R
install.packages(c("tidyverse", "caret", "cluster"))
```


***

## Usage Examples

### Python Clustering Example

```python
# Cell 3: Creating a Sample Dataset (Python code with comments)

# Import pandas for data manipulation
import pandas as pd

# Create a simple dataset with categorical and numerical data
data = pd.DataFrame({
    'age': [25, 30, 22, 40],         # Age in years
    'height': [186, 164, 175, 180],  # Height in cm
    'gender': ['male', 'female', 'male', 'female']  # Gender category
})

# Display the dataset
print("Sample Dataset:")
print(data)
```


***

```python
# Cell 4: K-Means Clustering Example (Python code with comments)

from sklearn.cluster import KMeans

# Sample data with two features: shape and color
fruit_data = pd.DataFrame({
    'shape': [5, 4, 1, 2],
    'color': [7, 8, 2, 1],
})

# Create K-Means model with 2 clusters and fixed random state
kmeans = KMeans(n_clusters=2, random_state=42).fit(fruit_data)

# Assign cluster labels to data points
fruit_data['cluster'] = kmeans.labels_

# Show clustered data
print("Clustered Data:")
print(fruit_data)
```


***

```python
# Cell 5: Logistic Regression for Credit Approval (Python code with comments)

from sklearn.linear_model import LogisticRegression

# Sample credit data: income, score, and approval status
credit_data = pd.DataFrame({
    'income': [2300, 4000, 1200, 6000],
    'score': [600, 700, 550, 800],
    'approved': [0, 1, 0, 1]  # 0 - Denied, 1 - Approved
})

# Features and target variable
X = credit_data[['income', 'score']]
y = credit_data['approved']

# Build logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict credit approval for a new client
new_client = [[3000, 650]]
prediction = model.predict(new_client)

# Print prediction result
print(f"Prediction for new client {new_client}: {'Approved' if prediction[^0] == 1 else 'Denied'}")
```


***

```python
# Cell 6: Visualizing Cluster Centers (Python code with comments)

import matplotlib.pyplot as plt

# Get cluster centers
centers = kmeans.cluster_centers_

# Plot data points colored by cluster
plt.scatter(fruit_data['shape'], fruit_data['color'], c=fruit_data['cluster'], cmap='viridis')

# Plot cluster centers as red 'X'
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

# Set labels and title
plt.xlabel("Shape")
plt.ylabel("Color")
plt.title("K-Means Clustering with Centroids")

# Show plot
plt.show()
```

<br><br>


### R Clustering Example

```R
fruit <- data.frame(shape=c(5,4,1,2), color=c(7,8,2,1))
km <- kmeans(fruit, centers=2)
print(km$cluster)
```


### R Classification Example

```R
data <- data.frame(income=c(2300,4000,1200,6000), score=c(600,700,550,800), approved=c(0,1,0,1))
model <- glm(approved ~ income + score, data, family=binomial)
predict(model, newdata=data.frame(income=3000, score=650), type="response")
```


***

## Knowledge Discovery in Databases (KDD)

KDD spans data selection, preprocessing, mining, and validation steps ensuring extracted knowledge is meaningful and valuable.

***

## Mathematical Concepts

| Concept | Description | Formula |
| :-- | :-- | :-- |
| Inflection | Point where curvature changes sign | $f''(x_0) = 0$, concavity change |
| Maximum | Local peak where $f'(x_0)=0$, $f''(x_0)<0$ |  |
| Minimum | Local trough where $f'(x_0)=0$, $f''(x_0)>0$ |  |


***

## Discrete vs Continuous Values

| Type | Example |
| :-- | :-- |
| Discrete | Loan approved: Yes/No |
| Continuous | Loan amount: \$1000 to \$10,000+ |


***

## Primary Libraries and Tools

| Name | Use | Language |
| :-- | :-- | :-- |
| pandas | Data manipulation | Python |
| NumPy | Numerical computations | Python |
| seaborn | Data visualization | Python |
| scikit-learn | Machine learning algorithms | Python |
| tidyverse | Data wrangling and plotting | R |
| caret | Machine learning in R | R |


***

## Supervised vs Unsupervised Learning

| Type | Known Labels | Purpose | Algorithms |
| :-- | :-- | :-- | :-- |
| Supervised | Yes | Predict labels/values | Logistic Regression, SVM, Decision Trees |
| Unsupervised | No | Discover patterns or clusters | K-Means, Hierarchical Clustering, DBSCAN |


***

## Clustering and Distance Calculations

Clusters group similar data by minimizing intra-cluster distances. Euclidean distance calculated as:

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

Objects assigned to clusters with nearest centroid.

***

## Credit Analysis: Classification Example

Predict if credit will be approved based on income and score data.

***

## Fruit Clustering Example

Attributes like shape and color are used to cluster fruits into categories with K-means.

***

## Anomaly Detection and Association Rules

- Detect rare, irregular events (fraud, anomalies) statistically or by distance metrics.
- Association rules discover frequently co-occurring attributes (e.g., smartphone buyers often subscribe to data plans).

***

## Applications

- Finance: fraud detection, credit scoring
- Energy: load forecasting, loss reduction
- Agriculture: crop yield prediction
- Web: sentiment analysis, customer segmentation

***

## Repository Structure and Documentation

```
/docs/       # Extended documentation and guides
/notebooks/  # Jupyter notebooks for exploration and tutorials
/src/        # Source code, modules
/tests/      # Automated tests
README.md    # This file
requirements.txt # Python dependencies
LICENSE      # License details
CONTRIBUTING.md # Contribution guidelines
```


***

## Contributing and License

- Contributions welcome via pull requests
- See `CONTRIBUTING.md` for details
- Licensed under the MIT License

***

## References

- Leandro Nunes de Castro \& Daniel Gomes Ferrari. *Introdução à mineração de dados*, Saraiva, 2016.
- André Carlos Ponce de Leon Ferreira et al. *Inteligência Artificial – Uma Abordagem de Aprendizado de Máquina*, LTC, 2024.
- Larson \& Farber. *Estatística Aplicada*, Pearson, 2015.

***

This README offers a complete, clear, and accessible guide for learners and practitioners working on data mining and AI projects.



# Data Cleaning And Preprocessing

## 📌 Project Overview

This project demonstrates the complete data preprocessing workflow on the Titanic dataset. The objective is to clean, transform, and prepare the dataset for machine learning by handling missing values, encoding categorical features, standardizing numerical data, and detecting/removing outliers.

---

## 📂 Dataset

**Dataset:** Titanic Dataset

The dataset contains information about passengers aboard the Titanic, including demographic details, ticket information, fare, cabin, and survival status.

---

## 🎯 Objectives

* Import and explore the dataset.
* Identify missing values and data types.
* Handle missing values using appropriate imputation techniques.
* Convert categorical features into numerical values.
* Normalize/standardize numerical features.
* Detect and remove outliers using the IQR method.
* Prepare a clean dataset for further analysis or machine learning.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📁 Project Structure

```text
EDA_elevate_labs/
│── Titanic-Dataset.csv
│── eda.py
│── README.md
```

---

## ⚙️ Steps Performed

### 1. Import Dataset

* Loaded the Titanic dataset using Pandas.
* Displayed the first few rows.
* Explored the dataset using:

  * `head()`
  * `info()`
  * `describe()`
  * `isnull().sum()`

---

### 2. Handle Missing Values

* Filled missing values in the **Age** column using the **Median**.
* Filled missing values in the **Embarked** column using the **Most Frequent** value.
* Removed the **Cabin** column due to a large number of missing values.

---

### 3. Encode Categorical Features

Used **LabelEncoder** to convert categorical variables into numerical values.

Encoded columns:

* Sex
* Embarked

---

### 4. Standardize Numerical Features

Applied **StandardScaler** to:

* Age
* Fare

This ensures the features have:

* Mean = 0
* Standard Deviation = 1

---

### 5. Detect Outliers

Visualized outliers using **Seaborn Boxplots** for:

* Age
* Fare

---

### 6. Remove Outliers

Removed outliers using the **Interquartile Range (IQR)** method.

---

## 📊 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

---

## 🚀 How to Run

1. Clone the repository.

```bash
git clone https://github.com/VigneshMallapuram/EDA_elevate_labs.git
```

2. Navigate to the project directory.

```bash
cd EDA_elevate_labs
```

3. Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the Python program.

```bash
python eda.py
```

---

## 📈 Expected Outcome

After completing the preprocessing steps:

* Missing values are handled.
* Categorical variables are encoded.
* Numerical features are standardized.
* Outliers are identified and removed.
* The dataset is clean and ready for machine learning or further analysis.

---

## 📚 Learning Outcomes

Through this project, I learned how to:

* Explore datasets using Pandas.
* Handle missing values effectively.
* Encode categorical variables.
* Standardize numerical features.
* Visualize data distributions and outliers.
* Clean data for machine learning applications.

---

## 👨‍💻 Author

**Vignesh Mallapuram**

GitHub: https://github.com/VigneshMallapuram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler
df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())

print("\ninformation")
print(df.info())
print("\nmissing values")
print(df.isnull().sum())
print("\nsummary")
print(df.describe())

print(df.columns)

age_imputer=SimpleImputer(strategy="median")
df["Age"]=age_imputer.fit_transform(df[["Age"]])
embarked_imputer=SimpleImputer(strategy="most_frequent")
df["Embarked"]=embarked_imputer.fit_transform(df[["Embarked"]]).ravel()
df.drop("Cabin",axis=1,inplace=True)
print(df.isnull().sum())

encoder=LabelEncoder()
df['Sex']=encoder.fit_transform(df['Sex'])
df['Embarked']=encoder.fit_transform(df["Embarked"])
print(df.head())

scaler=StandardScaler()
numerical_columns=["Age", "Fare"]
df[numerical_columns]=scaler.fit_transform(df[numerical_columns])
print(df.head())

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.boxplot(y=df["Age"])
plt.title("Age")

plt.subplot(1,2,2)
sns.boxplot(y=df["Fare"])
plt.title("Fare")

plt.tight_layout()
plt.show()
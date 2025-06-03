import pandas as pd 
import os
import matplotlib.pyplot as plt

file_path=os.path.join("datasets","housing.csv")
housing=pd.read_csv(file_path)

print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist()
housing.hist(bins=50,figsize=(20,15))
plt.show()

file_path=os.path.join("datasets","housing.xlsx")
housing=pd.read_excel(file_path)

print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist()
housing.hist(bins=50,figsize=(20,15))
plt.show()
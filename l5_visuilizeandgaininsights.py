import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

file_path = os.path.join("datasets","housing.csv")
housing = pd.read_csv(file_path)
housing["median_income"].hist()
print("contents of housing.cv:\n")
print(housing)
housing.plot(kind="scatter",x="longitude",y="latitude")
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
plt.show()
housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.4,s=housing["population"]/100,label ="population",figsize=(10,7)
    ,c="median_house_value",cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.show()
print("corelation value of 9 colums in dataset")
print("*********************************************\n")
corr_matrix = housing.corr(method="pearson",numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
pd.plotting.scatter_matrix(housing)
plt.show()
attribute =  ["median_house_value","median_income","total_rooms","housing_median_age"]
plt.show()

# lets focus zoom in on two attribute that seems to have some corelation 

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
plt.show()

# experimenting with attribute combinations(data engineering)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr(method="pearson",numeric_only=True)
print("corelation value of 12 colums in dataset after engineering 3 new column data")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("*********************************************\n")
pd.plotting.scatter_matrix(housing)
plt.show()
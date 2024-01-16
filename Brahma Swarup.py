#!/usr/bin/env python
# coding: utf-8

# In[1]:


# dataset : # https://data.worldbank.org/indicator/EN.ATM.CO2E.PC


# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt
import errors


# In[3]:


def import_data(path):
    """
    Function to read the world bank data using a path.
    Returns:
    - Two data frames - original and transposed data
    """
    df = pd.read_csv(path, skiprows=4)
    dfT = pd.read_csv(path, skiprows=4).set_index(['Country Name']).T
    return df, dfT


# In[4]:


df, dfT = import_data('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_6299644.csv')


# In[5]:


df.head()


# In[15]:


def filter_yearly_data(data_frame):
    """
    This function filters the DataFrame to include only data within the specified range of years.
    
    Args:
    - data_frame (DataFrame): The input DataFrame containing data for various years.
    
    Returns:
    - DataFrame: A subset of the input DataFrame containing data for the specified range of years.
    """
    selected_years = [str(year) for year in range(2000, 2020 + 1)]
    selected_columns = ['Country Name', 'Indicator Name'] + selected_years
    filtered_data = data_frame[selected_columns]
    return filtered_data

data = filter_yearly_data(df)


# In[16]:


data = data.dropna()


# In[17]:


data.head()


# In[19]:


datax = data[["Country Name", "2020"]].copy()
datax.head()


# In[36]:


def percentage_change(data, data2):
    """
    Calculate the percentage change between two specified years.
    
    Args:
    - data 1: The original DataFrame containing data for the initial year.
    - data 2: The DataFrame containing data for the final year.

    Returns:
    - DataFrame: The DataFrame with an additional 'Change' column.
    """
    percentage_change = 100.0 * (data["2020"] - data["2000"]) / data["2000"]
    data2 = data2.assign(Change=percentage_change.replace([np.inf, -np.inf], np.nan))
    return data2

datax = percentage_change(data, datax)


# In[40]:


datax = datax.dropna()


# In[41]:


datax.describe()


# In[43]:


def drop_outliers(data_frame, column_names):
    """
    Drop outliers from specified columns in the DataFrame.

    Args:
    - data_frame (DataFrame): The input DataFrame.
    - column_names (list): List of column names to consider for outlier removal.

    Returns:
    - DataFrame: The DataFrame with outliers removed.
    """
    Q1 = data_frame[column_names].quantile(0.25)
    Q3 = data_frame[column_names].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_data = data_frame[~((data_frame[column_names] < lower_bound) | (data_frame[column_names] > upper_bound)).any(axis=1)]

    return cleaned_data

datax = drop_outliers(datax, ["2020", "Change"])


# In[44]:


datax.describe()


# In[45]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=datax, x="2020", y="Change", label="Different Countries", color='red')
plt.xlabel("CO2 Emissions (kt) in 2020")
plt.ylabel("Percentage Change (2001 to 2020)")
plt.title("CO2 Emissions (kt) in 2020 vs. Percentage Change (2001 to 2020)")
plt.show()


# In[46]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[48]:


scaler = pp.RobustScaler()
x_norm = scaler.fit(datax[["2020", "Change"]]).transform(datax[["2020", "Change"]])


# In[49]:


silhouette_scores = []

for i in range(2, 12):
    score = one_silhouette(x_norm, i)
    silhouette_scores.append(score)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[50]:


kmeans = KMeans(n_clusters=2, n_init=10)
kmeans.fit(x_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[54]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=datax["2020"], y=datax["Change"], hue=labels, palette="Set1", s=30)
sns.scatterplot(x=xkmeans, y=ykmeans, color="black", s=70, label='Centroids', marker="x")
plt.xlabel("CO2 Emissions (kt) in 2020")
plt.ylabel("Percentage Change (2001 to 2020)")
plt.title("CO2 Emissions (kt) in 2020 vs. Percentage Change (2001 to 2020)")
plt.show()


# In[66]:


uk = dfT.loc['2000':'2022', ['United Kingdom']].reset_index()
uk = uk.rename(columns={'index': 'Year', 'United Kingdom': 'CO2 Emissions'})
uk = uk.apply(pd.to_numeric, errors='coerce')
uk = uk.dropna(subset=['CO2 Emissions']).replace([np.inf, -np.inf], np.nan).dropna()
uk.describe()


# In[67]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=uk, x='Year', y='CO2 Emissions')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.title('CO2 Emissions in the United Kingdom')
plt.show()


# In[68]:


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


# In[69]:


param, covar = opt.curve_fit(poly, uk["Year"], uk["CO2 Emissions"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(2001, 2031)
forecast = poly(year, *param)
sigma = errors.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
uk["fit"] = poly(uk["Year"], *param)


# In[73]:


plt.figure(figsize=(8, 5))
sns.lineplot(data=uk, x='Year', y='CO2 Emissions', label="CO2 Emissions in the UK")
sns.lineplot(x=year, y=forecast, label="Prediction")
plt.fill_between(year, low, up, color="yellow", alpha=0.7, label="Confidence Intervals")
plt.title("CO2 Emissions Forecasting of United Kingdom")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.legend()
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


df = pd.read_csv(r"C:\Users\DELL\Desktop\laptop.csv")
df = df.drop('Unnamed: 0', axis=1)


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.isna().sum()


# In[14]:


df[df['Screen_Size_cm'].isna() == True]


# In[15]:


avg_screen_size = df['Screen_Size_cm'].mean()
df['Screen_Size_cm'] = df['Screen_Size_cm'].fillna(avg_screen_size)


# In[16]:


avg_Weight_kg = df['Weight_kg'].mean()
df['Weight_kg'] = df['Weight_kg'].fillna(avg_Weight_kg)


# In[17]:


df.isna().sum()


# In[18]:


plt.figure(figsize=(10,6))
sns.countplot(x='Manufacturer', data=df, color='skyblue')
plt.xlabel('Manufacturer')
plt.ylabel('Amount')
plt.title('Manufacturer Distribution')

plt.show()


# In[33]:


numeric_columns = df.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)


# In[19]:


plt.figure(figsize=(10,6))
sns.histplot(df['Price'], color='skyblue')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution Distribution')

plt.show()


# In[20]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Manufacturer', y='Price', data=df)
plt.xlabel('Manufacturer')
plt.ylabel('Price')

plt.show()


# In[21]:


category_mapping = {1: 'Gaming', 2: 'Netbook', 3: 'Notebook', 4: 'Ultrabook', 5: 'Workstation'}
df['Category_Label'] = df['Category'].map(category_mapping)


plt.figure(figsize=(10,6))
sns.boxplot(x='Category_Label', y='Price', data=df)
plt.xlabel('Category')
plt.ylabel('Price')
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
sns.countplot(x='OS', data=df, color='skyblue')
plt.xlabel('OS')
plt.ylabel('Amount')
plt.title('OS Distribution')

plt.show()


# In[23]:


linux_laptops = df[df['OS'] == 2].sort_values(by='Price',ascending=True)
linux_laptops


# In[24]:


windows_laptops = df[df['OS'] == 1]
windows_laptops.head()


# In[25]:


plt.figure(figsize=(10, 6))
sns.histplot(windows_laptops['Price'], label='Windows', color='skyblue', alpha=0.7)
sns.histplot(linux_laptops['Price'], label='Linux', color='lightgreen', alpha=0.7)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Price for Windows and Linux Laptops')
plt.legend()
plt.show()


# In[26]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Screen_Size_cm', y='Price',data=df, color='skyblue')
plt.xlabel('Screen Size in cm')
plt.ylabel('Price')
plt.title('Correlation between price and Screen Size')

plt.show()


# In[27]:


plt.figure(figsize=(10,8))
sns.lmplot(x='CPU_frequency', y='Price', data=df)
plt.xlabel('CPU Frequency')
plt.ylabel('Price')
plt.show()


# In[28]:


plt.figure(figsize=(10,8))
sns.boxplot(x='RAM_GB', y='Price', data=df)
plt.xlabel('RAM GB')
plt.ylabel('Price')
plt.show()


# In[29]:


df.dtypes


# In[30]:


laptop_features = ['Category', 'GPU', 'OS', 'CPU_core', 'Screen_Size_cm', 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'Weight_kg']
y=df.Price

X = df[laptop_features]
X.describe()


# In[31]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

laptop_model = DecisionTreeRegressor(random_state=1)

laptop_model.fit(train_X, train_y)

val_predictions = laptop_model.predict(val_X)

mean_absolute_error(val_y, val_predictions)


# In[32]:


print('Making predictions for the following laptops:')
print(X.tail())

print("The predictions are")
print(laptop_model.predict(X.tail()))


# In[ ]:





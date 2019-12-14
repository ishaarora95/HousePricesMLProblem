#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=pd.read_csv('trainHousePrices.csv')


# In[3]:


sns.regplot(x="OverallCond", y="SalePrice", data=df);


# In[ ]:



fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(df.isnull(),cbar=False,ax=ax,cmap="YlGnBu")


# In[ ]:


df.info()


# In[7]:


df.drop('MiscFeature',axis=1,inplace=True)


# In[8]:


df.drop('PoolQC',axis=1,inplace=True)


# In[9]:


df.drop('Alley',axis=1,inplace=True)


# In[10]:


df.drop('MiscVal',axis=1,inplace=True)


# In[11]:


df.drop('FireplaceQu',axis=1,inplace=True)


# In[12]:


df.drop('Fence',axis=1,inplace=True)


# In[13]:


fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(df.isnull(),cbar=False,ax=ax)


# In[ ]:


df.info()


# In[14]:


x=df['LotFrontage'].mean()
df['LotFrontage'].fillna(x, inplace=True)


# In[15]:


df['LotFrontage']


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(df.isnull(),cbar=False,ax=ax)


# In[16]:


df.dropna(axis=0,how='any',inplace=True)


# In[ ]:


df.info()


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.info()


# In[ ]:


df


# In[17]:


Street = pd.get_dummies(df['Street'],drop_first=True)


# In[18]:


MSZoning = pd.get_dummies(df['MSZoning'],drop_first=True)


# In[19]:


LotShape = pd.get_dummies(df['LotShape'],drop_first=True)


# In[20]:


LandContour = pd.get_dummies(df['LandContour'],drop_first=True)


# In[23]:


Utilities=pd.get_dummies(df['Utilities'],drop_first=True)


# In[24]:


LotConfig=pd.get_dummies(df['LotConfig'],drop_first=True)


# In[25]:


LandSlope=pd.get_dummies(df['LandSlope'],drop_first=True)


# In[26]:


Neighborhood=pd.get_dummies(df['Neighborhood'],drop_first=True)


# In[27]:


Condition1=pd.get_dummies(df['Condition1'],drop_first=True)


# In[28]:


Condition2=pd.get_dummies(df['Condition2'],drop_first=True)


# In[29]:


BldgType=pd.get_dummies(df['BldgType'],drop_first=True)


# In[30]:


HouseStyle=pd.get_dummies(df['HouseStyle'],drop_first=True)


# In[31]:


RoofStyle=pd.get_dummies(df['RoofStyle'],drop_first=True)


# In[32]:


RoofMatl=pd.get_dummies(df['RoofMatl'],drop_first=True)


# In[33]:


Exterior1st=pd.get_dummies(df['Exterior1st'],drop_first=True)


# In[34]:


Exterior2nd=pd.get_dummies(df['Exterior2nd'],drop_first=True)


# In[35]:


MasVnrType=pd.get_dummies(df['MasVnrType'],drop_first=True) 
MasVnrArea=pd.get_dummies(df['MasVnrArea'],drop_first=True)
ExterQual=pd.get_dummies(df['ExterQual'],drop_first=True)
ExterCond =pd.get_dummies(df['ExterCond'],drop_first=True)
Foundation=pd.get_dummies(df['Foundation'],drop_first=True)
BsmtQual=pd.get_dummies(df['BsmtQual'],drop_first=True)
BsmtCond=pd.get_dummies(df['BsmtCond'],drop_first=True)
BsmtExposure=pd.get_dummies(df['BsmtExposure'],drop_first=True)
BsmtFinType1=pd.get_dummies(df['BsmtFinType1'],drop_first=True)


# In[36]:


BsmtFinType2=pd.get_dummies(df['BsmtFinType2'],drop_first=True)
Heating=pd.get_dummies(df['Heating'],drop_first=True)
HeatingQC=pd.get_dummies(df['HeatingQC'],drop_first=True)
CentralAir=pd.get_dummies(df['CentralAir'],drop_first=True)
Electrical=pd.get_dummies(df['Electrical'],drop_first=True)
KitchenQual=pd.get_dummies(df['KitchenQual'],drop_first=True)
Functional=pd.get_dummies(df['Functional'],drop_first=True)
GarageType=pd.get_dummies(df['GarageType'],drop_first=True)
GarageFinish=pd.get_dummies(df['GarageFinish'],drop_first=True)
GarageQual=pd.get_dummies(df['GarageQual'],drop_first=True)
GarageCond=pd.get_dummies(df['GarageCond'],drop_first=True)
PavedDrive =pd.get_dummies(df['PavedDrive'],drop_first=True)
SaleType=pd.get_dummies(df['SaleType'],drop_first=True)
SaleCondition=pd.get_dummies(df['SaleCondition'],drop_first=True)


# In[37]:


df=pd.concat([df,MSZoning, Street,
       LotShape, LandContour, Utilities, LotConfig, LandSlope,
       Neighborhood, Condition1, Condition2, BldgType, HouseStyle,
       RoofStyle,RoofMatl, Exterior1st, Exterior2nd, MasVnrType,
       ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond,
       BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC,
       CentralAir, Electrical, KitchenQual,Functional,GarageType, GarageFinish,
       GarageQual, GarageCond, PavedDrive,SaleType,
       SaleCondition],axis=1)


# In[51]:


df.drop(['MSZoning', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', 'KitchenQual','Functional','GarageType', 'GarageFinish',
       'GarageQual', 'GarageCond', 'PavedDrive','SaleType',
       'SaleCondition'],axis=1,inplace=True)


# In[52]:


df.head()
fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(df.isnull(),cbar=False,ax=ax)


# In[53]:


df.info()


# In[49]:


df.columns


# In[54]:


X=df.drop('SalePrice',axis=1)


# In[73]:


y=df['SalePrice']


# In[56]:


from sklearn.model_selection import train_test_split


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[75]:


from sklearn.linear_model import LinearRegression


# In[76]:


ln=LinearRegression()


# In[77]:


ln.fit(X_train,y_train)


# In[79]:


pred = ln.predict(X_test)


# In[63]:


from sklearn.metrics import mean_squared_error, r2_score


# In[1]:


# model evaluation
rmse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

# printing values
print('Slope:' ,ln.coef_)
print('Intercept:', ln.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# In[66]:


from sklearn import preprocessing


# In[67]:


normalized_X = preprocessing.normalize(df)


# In[69]:


newdf = pd.DataFrame(normalized_X)


# In[70]:


newdf.head()


# In[ ]:


accuracy = ln.score(X_test,y_test)


# In[ ]:


accuracy


# In[ ]:


sns.pairplot(df)


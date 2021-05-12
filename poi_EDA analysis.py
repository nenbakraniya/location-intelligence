#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


poi=pd.read_csv(r"C:\Users\nenba\Downloads\poi_sbu.csv")


# In[3]:


poi.head()


# In[4]:


poi.info()


# In[5]:


len(poi)


# In[6]:


poi2 = poi[[column for column in poi if poi[column].count() / len(poi) >= 0.3]]
del poi['ro_place_id']
print("List of dropped columns:", end=" ")
for i in poi.columns:
    if i not in poi2.columns:
        print(i, end=", ")
print('\n')
poi = poi2


# In[7]:


print(poi["postal_code"].describe())
plt.figure(figsize=(9, 8))
sns.distplot(poi["postal_code"], color='g', bins=100, hist_kws={'alpha': 0.4});


# In[8]:


list(set(poi.dtypes.tolist()))


# In[9]:


poi_num = poi.select_dtypes(include = ['float64', 'int64'])
poi_num.head()


# In[10]:


poi_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# In[11]:


poi_num_corr = poi_num.corr()['postal_code'][:-1] 
imp_features_list = poi_num_corr[abs(poi_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with postal_code:\n{}".format(len(imp_features_list), imp_features_list))


# In[ ]:


# for i in range(0, len(poi_num.columns),3):
sns.pairplot(data=poi_num,
                x_vars=poi_num,
                y_vars=['postal_Code'])


# In[26]:


data


# In[14]:


corr = poi_num.drop('postal_code', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# In[18]:


poi_num.columns


# In[19]:


features_to_analyse=poi_num.columns


# In[21]:


fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))

for i, ax in enumerate(fig.axes):
    if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='postal_code', data=poi[features_to_analyse], ax=ax)


# In[26]:


poi.columns


# In[33]:


categorical_features =['location_name', 'ro_brand_ids', 'brands',
       'top_category', 'sub_category', 'street_address', 'city',
       'region', 'postal_code', 'iso_country_code', 'phone_number',
       'open_hours', 'category_tags', 'poi_source', 'timestamp', 'ds_name']
poi_categ = poi[categorical_features]
poi_categ.head()


# In[34]:


poi_not_num = poi_categ.select_dtypes(include = ['O'])
print('There is {} non numerical features including:\n{}'.format(len(poi_not_num.columns), poi_not_num.columns.tolist()))


# In[35]:


poi_categ


# In[43]:


plt.figure(figsize = (30,30))
ax = sns.boxplot(x='top_category', y='postal_code', data=poi_categ)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# In[45]:


fig, axes = plt.subplots(round(len(poi_not_num.columns) / 3), 3, figsize=(100,100))

for i, ax in enumerate(fig.axes):
    if i < len(poi_not_num.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=poi_not_num.columns[i], alpha=0.7, data=poi_not_num, ax=ax)
fig.canvas.draw()
fig.tight_layout()


# In[ ]:





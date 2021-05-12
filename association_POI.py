#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[2]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[4]:


import folium
from folium import plugins
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
poi=pd.read_csv(r"C:\Users\nenba\Downloads\aug_poi_2020.csv")
poi.head()


# In[5]:


poi.describe()


# In[6]:


X=poi.drop([21937])
X


# In[7]:


X.reset_index()


# In[8]:


X[X['brands']=="{}"]


# In[9]:


X['brands'].nunique()


# In[18]:


X.drop(X[X['brands'] =="{}"].index, inplace = True) 


# In[19]:


X


# In[20]:


X.reset_index()


# In[21]:


X.describe()


# In[22]:


poi[poi["lat"]>=12023]


# In[14]:


id_n=150
kmeans = KMeans(n_clusters=id_n, random_state=0).fit(X[["lat",'long']])
id_label=kmeans.labels_


# In[15]:


print(kmeans.cluster_centers_)


# In[16]:


X['label']=id_label


# In[17]:


plt.figure(figsize=(50,50))
plt.scatter(X["lat"],X["long"],s=500, c=kmeans.labels_);
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, color="red"); # Show the centres
plt.title("Clusters")
plt.show   


# In[23]:


X['brands'].nunique()


# In[24]:


X=X.reset_index()


# In[25]:


X[X['brands']=="{}"]


# In[21]:


import geopy.distance


# In[59]:


for k in set(X['label']):
    for i in (X[X['label']==k]['brands'].index):
        for j in X[X['label']==k]['brands'].index[X[X['label']==k]['brands'].index>i]:
            if X[X['label']==k]['brands'][i]==X[X['label']==k]['brands'][j]:
                coord_1=X[['lat','long']].iloc[i]
                coord_2=X[['lat','long']].iloc[j]
                distance=geopy.distance.geodesic(coord_1, coord_2).km
                if distance<1:
                    print(distance)
                else:
                    continue


# In[26]:


X['brands'] = X['brands'].str.strip()
X.dropna(axis=0, subset=['label'], inplace=True)
X['label'] = X['label'].astype('str')


# In[79]:


for i in X['label']:
    if i==0:
        print(X['label'][i])


# In[22]:


X[X['label']=='99']['brands'].value_counts()


# In[26]:


POI_basket = (X.groupby(['label','brands'])['label']
              .sum().unstack().reset_index().fillna(0)
          .set_index('label'))
POI_basket


# In[27]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

POI_basket_sets = POI_basket.applymap(encode_units)


# In[28]:


POI_basket_sets


# In[32]:


freq_itemsets = apriori(POI_basket_sets, min_support=0.15, use_colnames=True)


# In[ ]:





# In[ ]:


rules = association_rules(freq_itemsets, metric="lift", min_threshold=3)
rules.head()


# In[ ]:


rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]


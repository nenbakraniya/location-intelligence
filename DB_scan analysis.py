#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[2]:


from sklearn.cluster import DBSCAN 
from sklearn import metrics 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 


# In[3]:


from mlxtend.preprocessing import TransactionEncoder


# In[4]:


poi=pd.read_csv(r"C:\Users\nenba\Downloads\aug_poi_2020.csv")
poi.head()


# In[5]:


X=poi.drop([21937])
X


# In[6]:


X.drop(X[X['brands'] =="{}"].index, inplace = True) 


# In[7]:


X.reset_index()


# In[8]:


Y=X[X["top_category"]=='Restaurants and Other Eating Places']


# In[9]:


Y.reset_index()


# In[11]:


Y[['lat','long']]


# In[62]:


z= StandardScaler().fit_transform(Y[['lat','long']])


# In[63]:


db = DBSCAN(eps=0.1, min_samples=20).fit(z)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# In[65]:


Y["clus_db"]=labels


# In[66]:


core_samples_mask


# In[67]:


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
# n_noise_ = list(labels).count(-1)


# In[68]:


print('Estimated number of clusters: %d' % n_clusters_)


# In[69]:


# Plot result
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = z[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = z[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
plt.show()


# In[73]:


Y['clus_db'].unique()


# In[76]:


POI_basket = (Y.groupby(['clus_db','brands'])['clus_db']
              .sum().unstack().reset_index().fillna(0)
          .set_index('clus_db'))
POI_basket


# In[77]:


LIST=[]
for i in set(Y['clus_db']):
    LIST.append(list(set(Y[Y['clus_db']==i]['brands'])))
    
print(LIST)


# In[18]:


def encode_units(Y):
    if Y <= 0:
        return 0
    if Y >= 1:
        return 1

POI_basket_sets = POI_basket.applymap(encode_units)


# In[78]:


traneco = TransactionEncoder()
traneco_ary = traneco.fit(LIST).transform(LIST)
fPOI = pd.DataFrame(traneco_ary, columns=traneco.columns_)
fPOI


# In[79]:


from mlxtend.frequent_patterns import fpgrowth

fpgrowth(fPOI, min_support=0.50)


# In[81]:


output=fpgrowth(fPOI, min_support=0.50, use_colnames=True)


# In[83]:


output.sort_values(by=['support'])


# In[ ]:





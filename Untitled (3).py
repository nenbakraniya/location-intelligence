#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#create clusutes of pois based on lat/long
#remove null brands


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[4]:


import folium
from folium import plugins


# In[5]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[6]:


poi=pd.read_csv(r"C:\Users\nenba\Downloads\poi_sbu.csv")
poi.head()


# In[7]:


poi[['long','lat']].isnull().sum()


# In[8]:


X=poi[['lat','long']]
X


# In[41]:


K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = X[['long']]
X_axis = X[['lat']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[33]:


loss = []

K = range(1,20)
for k in K:
    kmean = KMeans(n_clusters=k, random_state=0, n_init = 50, max_iter = 500)
    kmean.fit(X)
    loss.append(kmean.inertia_)


# In[34]:


plt.figure(figsize=(10,5))
plt.plot(K, loss, 'bx-')
plt.xlabel('k')
plt.ylabel('loss')
plt.title('The Elbow Method')
plt.show()


# In[32]:


id_n=10
kmeans = KMeans(n_clusters=id_n, random_state=0).fit(X)
id_label=kmeans.labels_


# In[54]:


id_label


# In[40]:


#plot result
ptsymb = np.array(['b.','r.','m.','g.','c.','k.','b*','r*','m*','r^']);
plt.figure(figsize=(12,12))
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
for i in range(id_n):
    cluster=np.where(id_label==i)[0]
    plt.plot(X.lat[cluster].values,X.long[cluster].values,ptsymb[i])
plt.show()


# In[77]:


# X=poi.loc[:,['lat','long']]
# x.head(10)
x.reshape(-1,1)


# In[80]:


K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = x[:,0]
X_axis = X[:,1]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[11]:


kmeans = KMeans(n_clusters = 2, init ='k-means++')
kmeans.fit(X[X.columns[1:2]]) # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:2]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(X[X.columns[1:2]]) # Labels of each point
X.head(10)


# In[13]:


centers


# In[ ]:





# In[12]:


X.plot.scatter(x = 'lat', y = 'long', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[109]:


import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn import metrics 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets 

 


# In[110]:


X 


# In[111]:


x= StandardScaler().fit_transform(X)


# In[112]:


x.dtype


# In[113]:


db = DBSCAN(eps=0.3, min_samples=10).fit(x)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


# In[114]:


X["clus_db"]=labels


# In[115]:


core_samples_mask


# In[ ]:





# In[116]:


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
m_clusters=len(set(labels))
n_noise_ = list(labels).count(-1)


# In[117]:


print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# In[119]:


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

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
# # Plot result 
# import matplotlib.pyplot as plt 

# # Black removed and is used for noise instead. 
# unique_labels = set(labels) 
# colors = ['y', 'b', 'g', 'r'] 
# print(colors)


# In[105]:


for k, col in zip(unique_labels, colors): 
    if k == -1: 
        # Black used for noise. 
        col = 'k'

    class_member_mask = (labels == k) 

xy = X[class_member_mask & core_samples_mask] 
 


# In[106]:


core_samples_mask


# In[107]:


class_member_mask


# In[132]:


xy=X[class_member_mask & core_samples_mask]
xy.iloc[:,0]
xy


# In[ ]:





# In[130]:


plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6) 

xy = X[class_member_mask & ~core_samples_mask] 
plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=col, 
markeredgecolor='k', 
markersize=6) 

plt.title('number of clusters: %d' %n_clusters_) 
plt.show()


# In[26]:


xy


# In[ ]:


plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6) 

xy = X[class_member_mask & ~core_samples_mask] 
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
markeredgecolor='k', 
markersize=6) 

plt.title('number of clusters: %d' %n_clusters_) 
plt.show()


# In[93]:


from mpl_toolkits.basemap import Basemap
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)


# In[ ]:





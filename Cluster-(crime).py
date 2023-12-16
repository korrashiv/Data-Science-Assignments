#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import seaborn as sns


# In[45]:


crime_data = pd.read_csv(r"C:\Users\SHIVA KUMAR\Desktop\Assigmnt\cluster\crime_data.csv")


# In[46]:


crime_data.head()


# In[47]:


crime_data.shape


# In[48]:


crime_data.info()


# In[49]:


crime_data.isna().sum()


# In[50]:


crime_data.columns


# In[51]:


X = crime_data.iloc[:,1:]
X.head()


# In[52]:


array = X.values
array


# In[53]:


X1 = pd.DataFrame(array,columns=crime_data.columns[:-1])


# In[54]:


plt.figure(figsize=(10,7))
plt.title("crime Dendrograms")
dend = sch.dendrogram(sch.linkage(X1, method='complete'))


# In[55]:


import pandas as pd
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
Y2 = cluster.fit_predict(X1)



# In[56]:


Y2_df = pd.DataFrame({'Cluster': Y2})


# In[57]:


counts = Y2_df['Cluster'].value_counts()
counts


# In[58]:


crime_data['Target'] = Y


# In[59]:


crime_data


# In[60]:


X = crime_data.iloc[:, 1:5].values 
X


# In[61]:


get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])


# In[62]:


# Initializing KMeans
from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=4,n_init=20)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
Y = kmeans.predict(X)


# In[63]:


Y = pd.DataFrame(Y)
Y.value_counts()


# In[64]:


crime_data['Target2'] = Y


# In[65]:


crime_data.head()


# In[66]:


kmeans.inertia_


# In[67]:


clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)
    


# In[73]:


clust


# In[ ]:


plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()



#!/usr/bin/env python
# coding: utf-8

# In[72]:


#import hierarchical clustering libraries

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[224]:


Airline_da = pd.read_csv(r"C:\Users\SHIVA KUMAR\Desktop\Assigmnt\EastAirline.csv")
Airline_da.head()


# In[226]:


Airline_da.drop(['ID'],axis=1,inplace=True)


# In[227]:


Airline_da.info()


# In[237]:


Airline_da.isna().sum()


# In[228]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[230]:


d1_norm = norm_func(Airline_da.iloc[:,:])
d1_norm.head()


# In[233]:


plt.figure(figsize=(10,7))
dendrogram = sch.dendrogram(sch.linkage(d1_norm, method='complete'))


# In[234]:


hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'complete')


# In[235]:


y_hc = hc.fit_predict(d1_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[236]:


d1_norm['h_clusterid'] = Clusters
d1_norm


# # DBSCAN

# In[208]:


Airline_data = pd.read_csv(r"C:\Users\SHIVA KUMAR\Desktop\Assigmnt\EastAirline.csv")
Airline_data.head(50)


# In[74]:


Airline_data.shape


# In[75]:


Airline_data.info()


# In[48]:


Airline_data.isna().sum()


# In[49]:


Airline_data.columns


# In[80]:


Airline_data.drop(["ID"],axis=1,inplace=True)


# In[81]:


for i in Airline_data.columns:
    print(i)
    print(Airline_data[i].nunique())


# In[213]:


Airline_data.drop(['ID'],axis=1,inplace=True)


# In[214]:


array = Airline_data.values
array


# In[87]:


for i in Airline_data.columns:
    print(i)
    print(Airline_data[i].unique())


# In[89]:


Stscaler = StandardScaler().fit(array)
X = Stscaler.transform(array)


# In[91]:


X #these are Z-scores which are stamdardized


# In[122]:


dbscan = DBSCAN(eps=1, min_samples=3)
dbscan.fit(X)


# In[123]:


#Noicey data points samples are given the label
""" -1 means outliers and positive are 1,2,3 any number is clusters"""
dbscan.labels_


# In[124]:


C1 = pd.DataFrame(dbscan.labels_,columns=["cluster"])
C1.head(50)


# In[128]:


Airline_data["Clusters"] = pd.DataFrame(dbscan.labels_) #adding CLusters column with all variable


# In[131]:


Airline_data.head(50)


# In[133]:


Airline_data[Airline_data["Clusters"] == -1] #shows the outliers


# In[137]:


#to remove this outliers
final_data = Airline_data[Airline_data["Clusters"] == 0]


# In[140]:


final_data.shape


# # K-means  

# In[171]:


from sklearn.cluster import KMeans


# In[196]:


Airline_data2 = Airline_data.copy()


# In[197]:


#Normaliztion

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(Airline_data2.iloc[:,1:])


# In[198]:


scaled_data


# In[199]:


#finding the best K value
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[215]:


#building the Algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(5, random_state=42)
clusters_new.fit(scaled_data)


# In[216]:


clusters_new.labels_


# In[217]:


#Assigning the clusters to the data set
Airline_data2["clustersid_new"] = clusters_new.labels_


# In[218]:


clusters_new.cluster_centers_


# In[219]:


Airline_data2


# In[220]:


Airline_data2.groupby('clustersid_new').agg(['mean']).reset_index()


# In[221]:


Airline_data2.head(25)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





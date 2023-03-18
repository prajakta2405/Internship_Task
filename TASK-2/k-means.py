
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

iris=datasets.load_iris()
iris_df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head())
print(iris_df.info())

#find the optimal no of k
x= iris_df.iloc[:,[0,1,2,3]].values
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters= i ,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#absorb the elbow point
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#obtained value of k is 3

kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=kmeans.fit_predict(x)

#visualising the cluster
plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='blue',label='Iris-Versicolour')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='green',label='Iris-virginica')

#plotting centroids of clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.legend()
plt.show()
# %%

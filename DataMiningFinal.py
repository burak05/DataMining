import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from tqdm import tqdm
import plotly.graph_objects as go





file = pd.read_csv('20191218-30min-PassengerCount.csv')
df = pd.DataFrame(file)

df['BoardingTime'] = df['BoardingTime'].astype(str)

df['BoardingTime'] = df.BoardingTime.str.replace('T',' ')

#df = df.drop(['BoardingTime'],axis=1)

#df['BoardingTime'] = df['BoardingTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M'))
df[['Day','Time']] = df.BoardingTime.str.split(" ",expand = True)


dfnew = df
dfnew = dfnew.drop(['BoardingTime'],axis = 1)
dfnew['Time'] = dfnew.Time.str.replace(':','')
dfnew['Day'] = dfnew.Day.str.replace('-','')
#dfnew['Time']= dfnew['Time'].astype(str).astype(int)
#dfnew['Day']= dfnew['Day'].astype(str).astype(int)
uniquelines = dfnew.Line.unique()


print(dfnew['Line'].value_counts())
print("Number of different bus lines in the dataset: ",len(uniquelines))
print("Number of total passengers: ", sum(dfnew['PassengerCount']))

sn.catplot(x="Line",y = "Time",kind="swarm",data = dfnew)
plt.show()

sn.catplot(x="PassengerCount",y = "Line",kind="swarm",data = dfnew)
plt.show()



#Elbow Method for finding optimal number of clusters
cost = []
K = range(1,15)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "Huang", n_init = 1, verbose=1)
    kmode.fit_predict(dfnew,categorical = [0,2,3])
    cost.append(kmode.cost_)

y = np.array([i for i in range(1,15,1)])
plt.title("Huang")
plt.scatter(y,cost)
plt.show()
cost = []
#Choosing number of cluster = 10 according to the plot
km_huang = KModes(n_clusters=10, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(dfnew,categorical = [0,2,3])

#making the clusters a dataframe
clustersdf = pd.DataFrame(fitClusters_huang)
clustersdf.columns = ['clusterspredicted']
combineddf = pd.concat([dfnew,clustersdf],axis = 1).reset_index()
combineddf = combineddf.drop(['index'], axis = 1)
print(combineddf.head())


sn.catplot(x="clusterspredicted",y = "Time",hue = "PassengerCount",data = combineddf)
plt.show()


sn.catplot(x="clusterspredicted",y = "Time",kind="swarm",data = combineddf)
plt.show()

sn.catplot(x="clusterspredicted",y = "Line",kind="swarm",data = combineddf)
plt.show()

plt.scatter(combineddf.Time, combineddf.PassengerCount, c=combineddf.clusterspredicted, alpha = 0.6, s=10)
plt.show()

print(combineddf)




"""
km = KModes(n_clusters=20, init='Huang', n_init=10, verbose=1)
clusters = KPrototypes().fit_predict(dfnew, categorical=[0,2,3])
clusters = km.fit_predict(dfnew)
print(km.cluster_centroids_)
print(km.n_iter_)
print(km.cost_)




# Choose optimal K using Elbow method
cost = []
for cluster in range(1, 20):
    try:
        kprototype = KPrototypes(n_jobs = 4, n_clusters = cluster, init = 'Huang', random_state = 42)
        kprototype.fit_predict(dfnew, categorical = [0,2,3])
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break

df_cost = pd.DataFrame({'Cluster':range(1, 20), 'Cost':cost})
df_cost.plot(x='Cluster', y='Cost', style='o')


plt.plot(cost)
plt.show()
"""
#print(dfnew.dtypes)
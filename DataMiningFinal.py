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
from plotnine import *
import plotnine



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
dfnew['Time']= dfnew['Time'].astype(str).astype(int)
dfnew['Day']= dfnew['Day'].astype(str).astype(int)
"""
pd.crosstab(dfnew['Line'],[dfnew['Time']]).plot.bar()
plt.title("Line vs Time and ethnicity")
plt.legend()
plt.show()
"""

"""
kmeans = KMeans(n_clusters = 3).fit(dfnew)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(dfnew['Time'], dfnew['PassengerCount'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
dfnew.plot(x= 'Time', y= 'PassengerCount', kind='scatter')
plt.show()
"""
"""
km = KModes(n_clusters=4, init='Huang', n_init=10, verbose=1)
clusters = KPrototypes().fit_predict(dfnew, categorical=[0])
clusters = km.fit_predict(dfnew)
print(km.cluster_centroids_)
print(km.n_iter_)
print(km.cost_)
"""


kproto = KPrototypes(n_clusters= 15, init='Huang', n_jobs = 4)
clusters = kproto.fit_predict(dfnew, categorical=[0])
print(pd.Series(clusters).value_counts())

#OPTIONAL: Elbow plot with cost (will take a LONG time)
costs = []
n_clusters = []
clusters_assigned = []

for i in tqdm(range(2, 5)):
    try:
        kproto = KPrototypes(n_clusters= i, init='Cao', verbose=2)
        clusters = kproto.fit_predict(dfnew, categorical=[0])
        costs.append(kproto.cost_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
    except:
        print(f"Can't cluster with {i} clusters")
        
#fig = go.Figure(data=go.Scatter(x=n_clusters, y=costs ))
#fig.show()
plt.scatter(n_clusters,costs)
plt.show()


#print(dfnew.dtypes)
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


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
dfnew.plot(x= 'Time', y= 'PassengerCount', kind='scatter')
plt.show()


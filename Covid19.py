import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

dataPasien = pd.read_csv ("patient.csv")
dataTerkonfirmasi = pd.read_csv ("confirmed_acc.csv")
Provinsi = pd.read_csv ("province.csv",encoding = 'unicode_escape')

print (dataPasien.head())
print (dataTerkonfirmasi.head())
print (Provinsi.head())

dataTerkonfirmasi.head()

(dataTerkonfirmasi[dataTerkonfirmasi['cases'] == 0].sort_values(by=['date'], ascending=False)).head()

df = dataTerkonfirmasi.iloc[39:]
df['days']= df['date'].map(lambda x : (datetime.strptime(x, '%m/%d/%Y') - datetime.strptime("3/1/2020", '%m/%d/%Y')).days)
df[['date','days','cases']] #reorder column

def gompertz(a, c, t, t_0):
    Q = a * np.exp(-np.exp(-c*(t-t_0)))
    return Q

x = list(df['days'])
y = list(df['cases'])

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, test_size=0.1, shuffle=False)
x_test_added = x_test + list(range((max(x_test)+1), 60))
popt, pcov = curve_fit(gompertz, x_train, y_train, method='trf', bounds=([100, 0, 0],[6*max(y_train),0.15, 70]))
a, estimated_c, estimated_t_0 = popt
y_pred = gompertz(a, estimated_c, x_train+x_test_added, estimated_t_0)

y_pred

plt.plot(x_train+x_test_added, y_pred, linewidth=2, label='predict data') 
plt.plot(x, y, linewidth=2, color='r', linestyle='dotted', label='train data')
# plt.plot(x_test, y_test, linewidth=2, color='g', linestyle='dotted', label='test data')
plt.title('prediction vs trained data on covid-19 cases in indonesia\n')
plt.xlabel('days since March 1st 2020')
plt.ylabel('confirmed positive')
plt.legend(loc='upper left')

prediksi = pd.DataFrame({'day_pred': x_test_added, 'cases_pred':np.around(y_pred[36:])})
prediksi

dataPasien.head()

dataPasien.info()
dataPasien['current_state'].value_counts()

male = dataPasien.loc[dataPasien['gender']=='male','age'].mean()
female = dataPasien.loc[dataPasien['gender']=='female','age'].mean()
print('Distribusi rata-rata umur pasien laki-laki: %i' %male, 'tahun')
print('Distribusi rata-rata umur pasien perempuan: %i' %female, 'tahun')

dataPasien.current_state.value_counts().plot.bar().grid()

sns.countplot(x='gender', hue='current_state', data=dataPasien)

dataPasien.province.value_counts().plot.bar()

plt.figure(figsize=(15,5))
dataPasien.confirmed_date.value_counts().plot.bar()




#ANALISIS DATA DAN VISUALISASI PROVINSI
Provinsi.head(5)

print("Total Data : ", Provinsi.shape,"\n")
Provinsi.info()

Provinsi['island'].value_counts()

#----------Group By---------------
def FungsiGroup(column, ds):
  dataset = ds
  print((dataset.groupby(column).sum()[['confirmed']]).sort_values(by=column, ascending=False))

FungsiGroup('island', Provinsi)

df =  pd.DataFrame((Provinsi.groupby('island').sum()[['confirmed']]).sort_values(by='island', ascending=False))
df.head()

df['island']=df.index
df.reset_index(drop=True, inplace=True)

df = df[['island', 'confirmed']]
df.head()


dataset = Provinsi[['province_name', 'capital_city', 'population_kmsquare', 'confirmed', 'deceased']]
dataset.head()

dataset.isnull().sum()



#LINEAR REGRESI DATASET PROVINSI
dtLR = dataset.copy()
print(dtLR.shape)
dtLR.head()

dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
numeric_data = dataset.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(12,10))
p = sns.heatmap(numeric_data.corr(), annot=True, cmap='RdYlGn')
plt.show()

dtLR = dtLR.drop(['province_name','capital_city','population_kmsquare'], axis=1)
dtLR.head(5)

#-----Proses linear regression
X_LR=dtLR.iloc[:, :-1].values
Y_LR=dtLR.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_LR, Y_LR, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_1, y_train_1)

pred = regressor.predict(X_test_1)

plt.scatter(X_test_1, y_test_1, color = 'red')
plt.plot(X_train_1, regressor.predict(X_train_1), color = 'blue' )
plt.title('confirmed vs deceased')
plt.xlabel('confirmed')
plt.ylabel('deceased')
plt.show()

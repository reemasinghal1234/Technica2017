%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

dateparse = lambda dates:pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv("D:/capital_one_nessie_data.csv", parse_dates=['month'], index_col=['month'], date_parser=dateparse)

decomposition = seasonal_decompose(data.shopping, freq=12)
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(15, 8)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

test_stationarity(data.shopping)

data.shopping_log= data.shopping.apply(lambda x: np.log(x))
test_stationarity(data.shopping_log)

data['log_first_difference'] = data.shopping_log - data.shopping_log.shift(1)
test_stationarity(data.log_first_difference.dropna(inplace=False))

data['seasonal_difference'] = data.shopping - data.shopping.shift(12)
test_stationarity(data.seasonal_difference.dropna(inplace=False))

data['log_seasonal_difference'] = data.shopping_log - data.shopping_log.shift(12)
test_stationarity(data.log_seasonal_difference.dropna(inplace=False))

mod = sm.tsa.statespace.SARIMAX(data.shopping, trend='n', order=(0,1,1), seasonal_order=(1,0,0,12))
results = mod.fit()
print results.summary()

data['forecast'] = results.predict(start = 10, end= 35, dynamic= True)
#print data['forecast']
data[['shopping', 'forecast']].plot(figsize=(12, 8))
plt.savefig('ts_data_predict.png', bbox_inches='tight')

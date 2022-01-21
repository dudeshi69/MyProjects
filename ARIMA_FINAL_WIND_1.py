#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pmdarima.arima import *
import statsmodels.tsa.arima.model as tsa
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import *
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)
import numpy as np
import sklearn.metrics as metrics

#from Models.Arima import *
#from Models.Misc import *
#from Data import *
#from Models.MLP import*
#from Models.Hybrid import *


# In[4]:


wind=pd.read_csv(r'C:\Users\Dhyey\Documents\wind speed data\freshwind.csv',index_col='Combined name')
wind['WS10M'].plot(figsize=(12,5))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Speed')
plt.title('Wind Graph')
plt.show()


# In[5]:


wind.dropna()


# In[6]:


wind=wind.reset_index()['WS10M']


# In[7]:


wind


# In[8]:


data_copy = wind.copy()
train_data, test_data = train_test_split(data_copy, test_size = 0.10, shuffle = False)
print(train_data)


# In[9]:


arima_model_order = auto_arima(train_data, start_p=0, d=0, start_q=0, max_p=5, max_d=5, start_P=0,
                               D=0, start_Q=0, max_P=5, max_D=5, max_Q=5, test='adf', seasonal_test='ocsb',
                               error_aciton='warn', trace=True, supress_warnings =True, 
                               stepwise=True, random_state= None, n_fits=50)


# In[10]:


arima_model_order.plot_diagnostics(figsize=(12,10))
plt.show()


# In[11]:


#from statsmodels.tsa.arima.model import ARIMA
# 5,0,3 ARIMA Model
model=sm.tsa.statespace.SARIMAX(wind,order=(4, 0, 1),seasonal_order=(0,0,0,0))
#model = ARIMA(first_diff, order=(5,0,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[12]:


start=len(train_data)
end=len(train_data)+len(test_data)-1
pred=model_fit.predict(start=start, end=end, exog= None, dynamic=False).rename('ARIMA Predictions')
pred.plot(legend=True)
test_data.plot(legend=True)


# In[13]:


index_future_dates = pd.date_range(start='2022-01-01',end='2023-02-15')
print(index_future_dates)
pred=model_fit.predict(start=len(train_data),end=len(train_data)+410,typ='levels').rename('ARIMA predictions') 
pred.index=index_future_dates
print(pred)


# In[14]:


pred.plot(figsize=(12,5),legend=True)


# In[15]:


mae = metrics.mean_absolute_error(test_data, pred)
mse = metrics.mean_squared_error(test_data, pred)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(test_data, pred)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)


# In[16]:


pred.to_csv(r'C:\Users\Dhyey\Documents\wind speed data\predicted_arima.csv')


# In[ ]:





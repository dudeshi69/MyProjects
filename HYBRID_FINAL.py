#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np
import pandas as pd


# In[2]:


wind=pd.read_csv(r'C:\Users\Dhyey\Documents\wind speed data\predicted_arima.csv', index_col='Date')
wind=wind.dropna()
wind.head()


# In[4]:


import matplotlib.pyplot as plt
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Speed')
plt.title('Wind Graph')
plt.plot(wind)


# In[5]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
wind=scaler.fit_transform(np.array(wind).reshape(-1,1))


# In[6]:


print(wind)


# In[7]:


##splitting dataset into train and test split
training_size=int(len(wind)*0.65)
test_size=len(wind)-training_size
train_data,test_data=wind[0:training_size,:],wind[training_size:len(wind),:1]


# In[8]:


training_size,test_size


# In[9]:


train_data


# In[10]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[13]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[14]:


print(X_train.shape), print(y_train.shape)


# In[15]:


print(X_test.shape), print(ytest.shape)


# In[16]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[17]:


model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[18]:


model.summary()


# In[20]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[21]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[22]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[23]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[24]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[25]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[26]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(wind)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(wind)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(wind)-1, :] = test_predict
# plot baseline and predictions
plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Wind Graph')
plt.plot(scaler.inverse_transform(wind))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[27]:


len(test_data)


# In[28]:


x_input=test_data[44:].reshape(1,-1)
x_input.shape


# In[29]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[30]:


temp_input


# In[57]:


# demonstrate prediction for next 150 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<144):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=1)
        print("{} day input {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=1)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[32]:


day_new=np.arange(1,101)
day_pred=np.arange(101,251)


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


len(wind)


# In[35]:


plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Wind Graph with predicted value using LTSM')
plt.plot(day_new,scaler.inverse_transform(wind[311:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[37]:


plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Wind Graph')
wind1=wind.tolist()
wind1.extend(lst_output)
plt.plot(wind1[100:])


# In[38]:


wind1=scaler.inverse_transform(wind1).tolist()


# In[39]:


plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Final Wind Graph')
plt.plot(wind1)


# In[40]:


df = pd.DataFrame (lst_output, columns = ['Values'])
print (df)


# In[41]:


df.plot(figsize=(12,5),legend=True)


# In[42]:


true_predictions = scaler.inverse_transform(lst_output)


# In[43]:


true_predictions


# In[44]:


df = pd.DataFrame (true_predictions, columns = ['Values'])
print (df)


# In[45]:


df.plot(figsize=(12,5))


# In[46]:


wind


# In[47]:


wind = scaler.inverse_transform(wind)


# In[48]:


wind


# In[49]:


df1 = pd.DataFrame (wind, columns = ['Original'])
print (df1)


# In[50]:


df1['Predictions']=df


# In[51]:


df=df1.dropna()
df


# In[52]:


df.plot(figsize=(12,5))


# In[58]:


mae = metrics.mean_absolute_error(test_data, lst_output)
mse = metrics.mean_squared_error(test_data, lst_output)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(test_data, lst_output)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)


# In[ ]:





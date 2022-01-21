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


wind=pd.read_csv(r'C:\Users\Dhyey\Documents\wind speed data\freshwind.csv', index_col='Combined name')
wind=wind.dropna()
wind.head()


# In[3]:


wind=wind.reset_index()['WS10M']
wind.head()


# In[4]:


import matplotlib.pyplot as plt
plt.grid(True)
plt.xlabel('Days')
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


# In[84]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[85]:


print(X_train.shape), print(y_train.shape)


# In[86]:


print(X_test.shape), print(ytest.shape)


# In[87]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[88]:


model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[89]:


model.summary()


# In[90]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=30,batch_size=64,verbose=1)


# In[91]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# 

# In[53]:


last_train_batch = X_train[-12:]


# In[96]:


last_train_batch = last_train_batch.reshape((1, 1200, 1))


# In[97]:


scaled_test[0]


# In[ ]:





# In[18]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[19]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[20]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[21]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[22]:


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


# In[23]:


len(test_data)


# In[30]:


x_input=test_data[1338:].reshape(1,-1)
x_input.shape


# In[31]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[32]:


temp_input


# In[45]:


# demonstrate prediction for next 150 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<2670):
    
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


# In[46]:


day_new=np.arange(1,101)
day_pred=np.arange(101,2569)


# In[47]:


import matplotlib.pyplot as plt


# In[48]:


len(wind)


# In[49]:


plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Wind Graph with predicted value using LTSM')
plt.plot(day_new,scaler.inverse_transform(wind[4008:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[39]:


plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Wind Graph')
wind1=wind.tolist()
wind1.extend(lst_output)
plt.plot(wind1[1100:])


# In[40]:


wind1=scaler.inverse_transform(wind1).tolist()


# In[41]:


plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Speed')
plt.title('Final Wind Graph')
plt.plot(wind1)


# In[42]:


df = pd.DataFrame (lst_output, columns = ['Values'])
print (df)


# In[43]:


df.plot(figsize=(12,5),legend=True)


# In[50]:


mae = metrics.mean_absolute_error(train_data, lst_output)
mse = metrics.mean_squared_error(train_data, lst_output)
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(train_data, lst_output)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)


# In[58]:


true_predictions = scaler.inverse_transform(lst_output)


# In[59]:


true_predictions


# In[63]:


df = pd.DataFrame (true_predictions, columns = ['Values'])
print (df)


# In[64]:


df.plot(figsize=(12,5))


# In[65]:


wind


# In[66]:


wind = scaler.inverse_transform(wind)


# In[67]:


wind


# In[68]:


df1 = pd.DataFrame (wind, columns = ['Original'])
print (df1)


# In[72]:


df1['Predictions']=df


# In[76]:


df=df1.dropna()
df


# In[78]:


df.plot(figsize=(12,5))


# In[ ]:





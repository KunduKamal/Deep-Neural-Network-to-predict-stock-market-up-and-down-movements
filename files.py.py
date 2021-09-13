#!/usr/bin/env python
# coding: utf-8

# In[183]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time #helper libraries
import datetime, os
import tensorflow as tf
import seaborn as sns
sns.set()
get_ipython().run_line_magic('load_ext', 'tensorboard')
import pandas_datareader.data as web
from datetime import datetime

from scipy import stats


# In[184]:


# Creating function for data input 
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


# In[185]:


#Assiging Start and end date
start = datetime(2015, 1, 1)
end = datetime(2021, 1, 30)


# In[186]:


#Showing first five rows of the Indices
dowJones = web.DataReader("^DJI",'yahoo',start,end)
euroStoxx = web.DataReader("^STOXX50E",'yahoo',start,end)
alphabet = web.DataReader("GOOG",'yahoo',start,end)
americanGroup = web.DataReader("AIG",'yahoo',start,end)
walmart = web.DataReader("WMT",'yahoo',start,end)
sap = web.DataReader("SAP",'yahoo',start,end)
allianz = web.DataReader("ALV.DE",'yahoo',start,end) #Prices in Euro
ahold = web.DataReader("AD.AS",'yahoo',start,end) #Prices in Euro
softBank = web.DataReader("SFTBY",'yahoo',start,end)
daiIchi = web.DataReader("8750.T",'yahoo',start,end) #Prices in JPY(Japanese Currency)
#FamilyMart = web.DataReader("FYM.SG",'yahoo',start,end) #Prices in Eur
gold = web.DataReader("GC=F",'yahoo',start,end)
Oil = web.DataReader("CL=F",'yahoo',start,end)
bitcoin = web.DataReader("BTC-EUR",'yahoo',start,end) 
familymart = web.DataReader("FYRTY",'yahoo',start,end)


# In[187]:


msci= pd.read_csv("MSCI2.csv")


# In[189]:


# **use ticker to assign desirable company data in variable 'com'**

com=dowJones


# In[190]:


# Finding log return of the data using closing price
com['log_return']=np.log(com['Close']/com['Close'].shift(1))

# droping null values
Gm=com['log_return'].dropna()

Gm.head(5)


# In[191]:


#using z-score to remove outliers
z = np.abs(stats.zscore(Gm))
print(np.where(z >3))

Zm =Gm[z < 3]


# In[192]:



#Assinging clean data and reshaping it
all_y = Zm.values       
dataset=all_y.reshape(-1, 1)


# In[193]:


# Using min max scaler to convert the values  of a specific company within the range of 0 to 1.

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[194]:


train_size = int(len(dataset) * 0.80)  #Choosing 80% of the datapoint for training (except bitcoin, 60%)
test_size = len(dataset) - train_size  #Choosing 20% of the datapoint for testing (except bitcoin, 40%)
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]  # Using previously assigned variables as index


# In[195]:


#Matrix shape of testing and training data
test.shape,train.shape  


# In[196]:


# look_back; this much previous datapoints will be considered from to calculate next 91th datapoint.
look_back = 90 
# Creating array for training and testing data 
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[197]:


# reshaping training and testing data for giving them as input in the model
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[198]:


#LSTM model 
model = Sequential()

model.add(LSTM(70,input_shape=(1, look_back)))

model.add(Dropout(0.1))

model.add(Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()


# In[199]:


import datetime


# In[200]:


# Computation of the data using LSTM model
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.fit(trainX, trainY, epochs=300, batch_size=150,validation_data=(trainX,trainY), verbose=1,callbacks=[tensorboard_callback])


# In[201]:


# tensorboard for tracking loss function (If it do not work in Jupyter Please skip this.)
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[202]:


# Using model to predict train and test data
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inversing the scaler form of dataset
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[203]:


# Calculating train and test error (mean squared error)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print('Dfference of error : '+str(testScore-trainScore))


# In[204]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score


# In[205]:


# Error of testing data (mean absolute error)
mean_absolute_error(testY[0],testPredict[:,0])


# In[206]:


# Error of testing data (mean squared error)
math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))


# In[207]:


#Creating an array for ploting train data predictions
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# In[208]:


#Creating an array for ploting test data predictions
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[209]:


len(testPredictPlot),len(trainPredict)


# In[210]:


# Ploting Test data predictions 
plt.plot(testPredictPlot)
plt.show()


# In[211]:


# Ploting training data and testing data prediction relative to original data
plt.figure(figsize=(12,8))
plt.plot(scaler.inverse_transform(dataset), label="Dataset")
plt.plot(trainPredictPlot,label="Train data")
plt.plot(testPredictPlot,label="Test data")
print('testPrices:')
testPrices=scaler.inverse_transform(dataset[test_size+look_back:])
plt.xlabel('Days')
plt.ylabel('Log Return')
plt.legend(loc="upper right")


# In[212]:


# Creating array for prediction of stock price using model
x_input=test[(len(test)-91):].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[213]:


# Prediction fuction
from numpy import array
lst_output=[]
n_steps=90
i=0
while (i<10):
    if(len(temp_input)>90):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape(1,1,n_steps)
        yhat = model.predict(x_input,verbose=0,callbacks=[tensorboard_callback])
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape(1,1,n_steps)
        yhat = model.predict(x_input,verbose=0,callbacks=[tensorboard_callback])
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)


# In[214]:


# converting prediction values and inserting into array
d_rm=np.array(np.exp(scaler.inverse_transform(lst_output)))-1
print(d_rm)

# converting prediction values without inserting into array
q=np.exp(scaler.inverse_transform(lst_output))


# In[215]:


#Calculating daily volatily  of last 10 days predicted values 
d_vol=np.std(d_rm)

# Inserting daily volatility into an array of 10 row and 1 column
vol=np.full((10,1),d_vol)
print(vol)


# In[216]:


#Calculating daily risk free rate (using 1 year US Tresary bond)
rf=0.0008             # 12 months Treasury Yields in 0.08% from  https://www.bloomberg.com/markets/rates-bonds/government-bonds/us
d_rf=((1+rf)**(1/365))-1     

# Inserting daily risk free rate into an array of 10 row and 1 column
d_rf=np.full((10,1),d_rf)
print(d_rf)


# In[217]:


#Calculating daily sharp ratio
d_srp=np.array((d_rm-d_rf)/vol)
print(d_srp)


# In[218]:


# Creating dataframe of the predicted values
index=range(1,11)#pd.date_range('31/01/2021',periods=5,freq='D')
z3=pd.DataFrame(d_rm,index=index,columns=(['daily Returns']))
z3['daily Sharpe Ratio']=d_srp
z3['daily Risk free rate']=d_rf
z3['10 days Volatility']= vol
z3


# In[219]:


# Fuction for making long or short decision
last_day_return = np.exp(com.iloc[-1]['log_return'])-1   #using last day return

t=[]
t.append(last_day_return)

t2=list(z3['daily Returns'].values)  
for z in range(0,10):
    t.append(t2[z])


a,b=[],[]
invest_wt= 10000
invest_wot=10000
# with trading cost
for i in range(1,11):

  if (t[i]>=t[i-1]):
  
    long_return_t=abs(t[i])*(invest_wt-(2+(invest_wt*0.0011)))
    invest_wt=long_return_t+(invest_wt-(2+(invest_wt*0.0011)))     # trading cost €2.00+0.11% from https://www.degiro.co.uk/data/pdf/uk/UK_Feeschedule.pdf
    a.append(invest_wt)  
    print('day : '+str(i)+' (With Tradig Cost)' +" Go LONG with return of     \t"+ str(long_return_t) + '\n \t\t\t\tTotal amount will be \t'+ str(invest_wt)+'\n') 
    
  else:
    short_return_t=abs(t[i])*(invest_wt-(2+(invest_wt*0.0011)))
    invest_wt=short_return_t+(invest_wt-(2+(invest_wt*0.0011)))   # trading cost €2.00+0.11% from https://www.degiro.co.uk/data/pdf/uk/UK_Feeschedule.pdf
    a.append(invest_wt)   
    print('day: '+str(i)+' (With Tradig Cost)'+' Go SHORT with return of    \t'+ str(short_return_t)+ '\n \t\t\t\tTotal amount will be  \t'+ str(invest_wt)+'\n') 


# without trading cost

for i in range(1,11):
  if (t[i]>=t[i-1]):
  
    long_return=abs(t[i])*invest_wot
    invest_wot=long_return+invest_wot
    b.append(invest_wot) 
    print('day: '+str(i)+' (Without Tradig Cost)' +" Go LONG with return of   \t"+ str(long_return) + '\n \t\t\t\tTotal amount will be  \t'+ str(invest_wot)+'\n') 
    
  else:
    short_return=abs(t[i])*invest_wot
    invest_wot=short_return+invest_wot
    b.append(invest_wot) 
    print('day: '+str(i)+' (Without Tradig Cost)'+' Go SHORT with return of  \t'+ str(short_return)+ '\n \t\t\t\tTotal amount will be   \t'+ str(invest_wot)+'\n')
    
plt.figure(figsize=(12,7))
plt.plot(a,color='c',marker='o',markersize=12,linestyle='-.',label="With Tradibng cost")
plt.plot(b,color='r',marker='^',markersize=12,linestyle='-.',label="Without Trading cost")
plt.xlabel("Day")
plt.ylabel("Amount")
plt.legend(loc="upper right")


# In[220]:


# Total profit after long or short investment (WITHOUT trading cost)
total_profit_t=b[9]-10000
print("Profit without adjusting trading cost : "+str(total_profit_t))

# Total profit after long or short investment (WITH trading cost)

total_profit= a[9]-10000
print("Profit after adjusting trading cost : "+str(total_profit))


# In[221]:


#Creating list of 2 days prediction
p_return=[]
for i in range(1,5):
    p_return.append(t[i])

print(p_return)


# In[222]:


#Function of comparing predicted and original price movement 
def prediction_comparison(p_return,o_return):
  for i in range(0,4):
    if (p_return[i]>0 and o_return[i]>0):
      print("Predicted Correct price movement of Day "+ str(i+1))
    elif (p_return[i]<0 and o_return[i])<0:
      print("Predicted Correct price movement of Day "+str(i+1))
    else:
      print("Predicted Incorrect price movement of Day "+ str(i+1))  


# In[223]:


#All these original prices (returns) are taken from 01-02-2021 to 02-02-2021

o_return_adas=[0.00763035184400174,-0.00967606226335715,0.00254885301614283,-0.00593220338983058]
o_return_sft=[0.0480596247751222,-0.013241785188818,0.0340457256461233,0.00480653689017063]
o_return_goog=[0.0357403553880178,0.0137586451731664,0.0739607057810336,-0.00371968097697195]
o_return_alv=[0.00932875831010072,0.0236906406034207,0.00217932752179317,0.0146008077042561]
o_return_daiichi=[0.0126103404791928,0.0245952677459527,0.0346399270738378,0.0108663729809104]
o_return_aig=[-0.000534188034187921,0.0307322287546765,0.00985221674876846,0.0613607188703467]
o_return_eustox=[0.0141924031435268,0.0168826203322145,0.00537257064554408,0.00896738001246611]
o_return_dji=[0.00764743041135163,0.0157411431452033,0.00117702724368374,0.0108144878855343]

#Assign above variable names as 2nd parameter of the function to check the comparison of the predictin and orginal return. 
prediction_comparison(p_return,o_return_eustox)   


# In[224]:


# creating array for inserting 10 day prediction
from_=len(dataset)-90
day_new=np.arange(1,91) 
day_pred=np.arange(91,101) 


# In[225]:


# Ploting prediction data
m=np.exp(scaler.inverse_transform(dataset[from_:]))
plt.plot(day_new,m)
plt.plot(day_pred,q)
plt.xlabel("Days")
plt.ylabel("Log Return")


# In[226]:


# Using f.fn package for analysis of the market condition due to covid-19


# In[136]:


get_ipython().system('pip install ffn')


# In[49]:


import ffn


# In[50]:


#Price data from Yahoo! Finance
prices = ffn.get('GOOG,AIG,WMT,SAP,ALV.DE,AD.AS,SFTBY,8750.T,FYRTY', start='2019-06-01',end='2019-12-30')


# In[138]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[139]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[140]:


#Price data from Yahoo! Finance
prices = ffn.get('GOOG,AIG,WMT,SAP,ALV.DE,AD.AS,SFTBY,8750.T,FYRTY', start='2020-01-01',end='2020-06-30')


# In[141]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[142]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[143]:


#Price data from Yahoo! Finance
prices = ffn.get('GOOG,AIG,WMT,SAP,ALV.DE,AD.AS,SFTBY,8750.T,FYRTY', start='2020-07-01',end='2020-12-30')


# In[144]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[145]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[146]:


msci= pd.read_csv("MSCI2.csv")
prices = ffn.get('^DJI,^STOXX50E,msci', start='2019-06-01',end='2019-12-30')


# In[147]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[148]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[227]:


#Price data from Yahoo! Finance
prices = ffn.get('^DJI,^STOXX50E,msci', start='2020-01-01',end='2020-06-30')


# In[228]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[229]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[230]:


#Price data from Yahoo! Finance
prices = ffn.get('^DJI,^STOXX50E,msci', start='2020-07-01',end='2020-12-30')


# In[231]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[232]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[233]:


#Price data from Yahoo! Finance
prices = ffn.get('CL=F,GC=F,BTC-EUR', start='2019-06-01',end='2019-12-30')


# In[234]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[235]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[236]:


#Price data from Yahoo! Finance
prices = ffn.get('CL=F,GC=F', start='2020-01-01',end='2020-06-30')


# In[237]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[238]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[239]:


#Price data from Yahoo! Finance
prices = ffn.get('CL=F,GC=F,BTC-EUR', start='2020-07-01',end='2020-12-30')


# In[240]:


# Performance metrics
stats = prices.calc_stats()
stats.display()


# In[241]:


# Depicting drawdowns
ax = stats.prices.to_drawdown_series().plot()


# In[ ]:





# In[ ]:





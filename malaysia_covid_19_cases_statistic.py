# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:23:45 2022

This script is to study the Malaysia covid-19 cases statistic

credit to source :
    GitHub - MoH-Malaysia/covid19-public: 
        Official data on the COVID-19 epidemic in Malaysia.
        Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.

@author: Afiq Sabqi
"""


import os
import pickle
import datetime
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM,Dense,Dropout
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error


#%%                             STATIC

DATA_PATH=os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')

MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model_dev.h5')
MMS_PATH=os.path.join(os.getcwd(),'model','mms.pkl')

log_dir=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH=os.path.join(os.getcwd(),'logs',log_dir)

#%%                          DATA LOADING

df=pd.read_csv(DATA_PATH)


#%%                         DATA INSPECTION
df.info()
# cases_new seems in dtype object

df.describe().T
msno.matrix(df)

# to change dtype from object to numerical
df['cases_new']=pd.to_numeric(df['cases_new'],errors='coerce')

plt.figure()
plt.plot(df['cases_new'])
plt.plot(df['cases_recovered'])
plt.plot(df['cases_active'])
plt.legend(['cases_recovered','cases_active','cases_new'])
plt.title('Comparison for cases_new,cases_active and cases_recovered')
plt.show()


#%%                          DATA CLEANING

# there is ? character in column cases_new change to NaNs
df=df.replace('?', np.nan)

# NaNs value is replace using interpolate method
df['cases_new'].interpolate(method='polynomial', order=2,inplace=True)



#%%                         FEATURES SELECTION

# only cases_new is selected for this prediction

#%%                           PREPROCESSING

mms=MinMaxScaler()
df=mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))


X_train=[]
y_train=[]    # to initialize empty list

win_size=30

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])

X_train=np.array(X_train)
y_train=np.array(y_train)


#%%                          MODEL DEVELOPMENT

model=Sequential()
model.add(Input(shape=(np.shape(X_train)[1],1)))   # Input_length, #features
model.add(LSTM(64,return_sequences=(True)))  # LSTM,GRU,RNN need 3D layer for input
model.add(Dropout(0.3))
model.add(LSTM(64,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1,activation='relu'))
model.summary()


model.compile(optimizer='adam',
              loss='mse',
              metrics='mape')

plot_model(model,show_layer_names=(True),show_shapes=(True))

X_train=np.expand_dims(X_train,axis=-1)

tensorboard_callback=TensorBoard(log_dir=LOG_PATH)

hist=model.fit(X_train,y_train,
              batch_size=128,epochs=100,
              callbacks=[tensorboard_callback])
#%%                          MODEL EVALUATION
hist.history.keys()


plt.figure()
plt.plot(hist.history['mape'])
plt.title('MAPE calculations')
plt.show

plt.figure()
plt.plot(hist.history['loss'])
plt.title('Training loss')
plt.show

results=model.evaluate(X_train,y_train)
print(results)


#%%                   MODEL DEVELOPMENT AND ANALYSIS

CSV_TEST=os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')

df_test=pd.read_csv(CSV_TEST)

df_test.info()
df_test['cases_new'].interpolate(method='polynomial', order=2,inplace=True)

df_test=mms.transform(np.expand_dims(df_test['cases_new'],axis=-1))



con_test=np.concatenate((df,df_test),axis=0)
con_test=con_test[-130:]

print(con_test[-130:])

X_test=[]
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])
    
X_test=np.array(X_test)

predicted=model.predict(np.expand_dims(X_test,axis=-1))


#%%                    PLOTTING GRAPHS

plt.figure()
plt.plot(df_test,'b',label='actual cases_new')
plt.plot(predicted,'r',label='predicted cases_new')
plt.legend()
plt.title('Predicted vs Actual')
plt.show()


plt.figure()
plt.plot(mms.inverse_transform(df_test),'b',label='actual cases_new')
plt.plot(mms.inverse_transform(predicted),'r',label='predicted cases_new')
plt.legend()
plt.title('Predicted_inversed vs Actual_inversed')
plt.show()

#%%                           MSE & MAPE


print('mae: ',mean_absolute_error(df_test,predicted))
print('mse: ',mean_squared_error(df_test,predicted))
print('mape: ',mean_absolute_percentage_error(df_test,predicted))

df_test_inversed=mms.inverse_transform(df_test)
predicted_inversed=mms.inverse_transform(predicted)

print('mae_i: ',mean_absolute_error(df_test_inversed,predicted_inversed))
print('mse_i: ',mean_squared_error(df_test_inversed,predicted_inversed))
print('mape_i: ',mean_absolute_percentage_error(df_test_inversed,
                                               predicted_inversed))

print(mean_absolute_error(df_test,predicted)/sum(abs(df_test))*100)
#%%                          MODEL SAVING

# saving model development
model.save(MODEL_SAVE_PATH)

# saving mms.pkl
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

#%%                         CONCLUSION

'''
    *The model is able to predict the trend of the new cases of covid-19.
    
    *Despite the error is around mae = 7% and mape=0.1525%
    
    *Eventhough a good model must have mape <10%.But in this case
    we only need a trend. because this is something biologically effect
    it does not have any pattern. since it can predict the trend and the
    percentage error is very low, then maybe we can predict when
    the wave of covid 19 will hit again. the government can be on stanby
    in any related things.
    
    *To explain more we have to know the meaning of MAPE where it divides
    the absolute error by the actual data, hence it will affected when
    there is a lot of data values closes to 0.
    
    *Also MAPE this type of error may be misleading because of the value
    expressed in absolute value which is no signs
    
    

'''

























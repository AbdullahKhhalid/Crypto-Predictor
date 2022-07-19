from matplotlib import units
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer,Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import shape
from keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation,  Flatten


#The cryptocurrency to be predicted and base currency is defined here
crypto_currency = "ETH"
against_currency = "USD"

#Start and end date for test data
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

#uses DataReader from pandas module to get the historical price data from yahoo finance
data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end)

#Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

#sets the number of days to be predicted and test data
#looks at past [prediction_days] to predict
prediction_days = 60
future_day = 7

#creates x_train and y_train
x_train, y_train = [], []


for x in range(prediction_days, len(scaled_data)-future_day):
    #past [prediction_days] and appends the real data to x_train
    x_train.append(scaled_data[x-prediction_days:x, 0])
    #the [future_day] is appended as the trained predected data to y_train
    y_train.append(scaled_data[x+future_day, 0])
    
#turns x and y train into numpy arrays and then reshapes them
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),initializer="normal")
        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:

            return output
        return K.sum(output, axis=1)

#Create neural network

model = Sequential()

#LSTM layers for feeding data to neural network. Dropout layers to prevent overfitting

model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(layers.LSTM(units=50, return_sequences=True))
model.add(attention(return_sequences=True))
model.add(Dropout(0.2))
model.add(layers.LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

"""
model.add(layers.GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(layers.GRU(units=50, return_sequences=True))
model.add(attention(return_sequences=True))
model.add(Dropout(0.2))
model.add(layers.GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.add(Conv1D(filters=50, kernel_size=3,activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(filters=50, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(attention(return_sequences=True))
model.add(MaxPooling1D(pool_size=2 ))
model.add(layers.LSTM(units=50, return_sequences=True))
model.add(attention(return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
"""
# to do: define callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
checkpoint_path = "training_1/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=10)

#compiling and fitting the model 
model.compile(optimizer="adam", loss="mean_squared_error",metrics="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32,callbacks=[cp_callback ,early_stopping],shuffle=True)
#model.save('ETH_15_day')

#Testing the model

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", test_start, test_end)
actual_prices = test_data["Close"].values

#combine the test data and actual data
total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

#places the actual data - test and prediction data into the model input 
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Predicts the prices using the model
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

#Plots them on a graph
plt.plot(actual_prices, color="red", label="Actual Prices")
plt.plot(prediction_prices, color="green", label="Predicited Prices")
plt.title(f"{crypto_currency} Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

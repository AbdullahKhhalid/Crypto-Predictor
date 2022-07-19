#Testing the model
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np



#The cryptocurrency to be predicted and base currency is defined here
crypto_currency = "BTC"
against_currency = "USD"
prediction_days = 60

#Start and end date for test data
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

#uses DataReader from pandas module to get the historical price data from yahoo finance
data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end)


model = keras.models.load_model('BTC_60_day')
scaler = MinMaxScaler(feature_range=(0,1))

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
prediction_prices=prediction_prices.tolist()
prediction_prices.append([0])
#Plots them on a graph
plt.plot(actual_prices, color="red", label="Actual Prices")
plt.plot(prediction_prices, color="green", label="Predicited Prices")
plt.title(f"{crypto_currency} Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
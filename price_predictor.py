import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

"""
Download stock data over input period.
"""
def loadStockData(ticket, start_date, end_date):
  stock = yf.download(ticket, start=start_date, end=end_date)
  stock.reset_index(inplace=True)
  stock['Date'] = pd.to_datetime(stock['Date'])
  return stock

"""
Display plot of stock price over prediction range.
"""
def displayStockPrice(stock):
  plt.plot(stock['Date'], stock['Close'])
  plt.xlabel("Date")
  plt.ylabel("Close")
  plt.title("Apple Stock Prices")
  plt.show()

"""
Select a subset of data for training. Apply scaling and prepare features and
labels that are x_train and y_train.
"""
def createTrains(stock):
  close_data = stock.filter(['Close'])
  dataset = close_data.values
  training = int(np.ceil(len(dataset) * .95))

  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(dataset)

  train_data = scaled_data[0:int(training), :]
  # prepare feature and labels
  x_train = []
  y_train = []

  for i in range(60, len(train_data)):
      x_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  return dataset, training, scaler, scaled_data, x_train, y_train

"""
Build and return Gated RNN- LSTM network using TensorFlow.
"""
def createModel(x_train, y_train):
  model = keras.models.Sequential()

  # https://keras.io/api/layers/recurrent_layers/lstm/
  model.add(keras.layers.LSTM(units=64,
                              return_sequences=True,
                              input_shape=(x_train.shape[1], 1)))
  model.add(keras.layers.LSTM(units=64))
  model.add(keras.layers.Dense(32))
  model.add(keras.layers.Dropout(0.5)) # Dropout layer ignored set of random neurons, prevents overfitting
  model.add(keras.layers.Dense(1))

  return model

"""
Compile Model.
* optimizer: optimizes the cost function by using gradient descent.
* loss: monitors whether the model is improving with training or not.
* metrics: evaluates the model by predicting the training and the validation data.
"""
def compileModel(model, x_train, y_train):
  model.compile(optimizer='adam', loss='mean_squared_error')
  history = model.fit(x_train, y_train, epochs=10)

"""
Create the testing data and then proceed with the model prediction.
"""
def createTestingData(training, model, scaler, scaled_data, dataset):
  test_data = scaled_data[training - 60:, :]
  x_test = []
  y_test = dataset[training:, :]
  for i in range(60, len(test_data)):
      x_test.append(test_data[i-60:i, 0])

  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  # predict the testing data
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  # evaluation metrics
  mse = np.mean(((predictions - y_test) ** 2))
  print("MSE", mse)
  print("RMSE", np.sqrt(mse))

  return predictions

"""
Plot stock price: train, test, and predictions.
"""
def plotPredictions(stock, training, predictions):
  train = stock[:training]
  test = stock[training:]
  test['Predictions'] = predictions

  plt.figure(figsize=(10, 8))
  plt.plot(train['Date'], train['Close'])
  plt.plot(test['Date'], test[['Close', 'Predictions']])
  plt.title(f'{stock} Close Price')
  plt.xlabel('Date')
  plt.ylabel("Close")
  plt.legend(['Train', 'Test', 'Predictions'])

"""
Predict price of a given stock.
"""
def predictPrice(stock):
  dataset, training, scaler, scaled_data, x_train, y_train = createTrains(stock)
  model = createModel(x_train, y_train)
  compileModel(model, x_train, y_train)
  predictions = createTestingData(training, model, scaler, scaled_data, dataset)
  plotPredictions(stock, training, predictions)

if __name__ == "__main__":
  stock = loadStockData('PAYC', '2010-01-01', '2020-12-31')
  predictPrice(stock)

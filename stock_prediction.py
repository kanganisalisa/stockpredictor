"""
Created with guidance from https://www.geeksforgeeks.org/stock-price-prediction-project-using-tensorflow/.
Created to learn concepts.
"""

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

import warnings
warnings.filterwarnings("ignore")

"""
Download stock data from yfinance
"""

data = yf.download(['GOOG','MSFT','AAPL','LYFT','AMZN','TSLA','NVDA','INTC','PAYC'],
                      start='2019-01-01',
                      end='2021-06-12',
                      progress=False,
)
data.reset_index(inplace=True) # solves Date access issue

#print(data.head())
#print(data.sample(7))
#data.info()

"""
Example ticker data, AAPL
"""

aapl_ticker = yf.Ticker('AAPL')
aapl_df = aapl_ticker.history(period="10y")
aapl_df.reset_index(inplace=True) 
#aapl_df.head()
aapl_df['Close'].plot(title="APPLE's stock price")
#aapl_df['Date']

"""
Price plots, 9 stocks
"""

data['Date'] = pd.to_datetime(data['Date'])
plt.figure(figsize=(15, 8))

for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    ticker = yf.Ticker(company)
    c = ticker.history(period="10y")
    plt.plot(c['Close'], c="r", label="close", marker="+")
    plt.plot( c['Open'], c="g", label="open", marker="^")
    plt.title(f"{company} Price")
    plt.legend()
    plt.tight_layout()

"""
Volume plots, 9 stocks
"""

data['Date'] = pd.to_datetime(data['Date'])
plt.figure(figsize=(15, 8))

for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    ticker = yf.Ticker(company)
    c = ticker.history(period="10y")
    plt.plot(c['Volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.tight_layout()
    index = index + 1

"""
AAPL data from 2013 to 2018
"""

apple = yf.download('AAPL', start='2013-01-01', end='2018-01-01', progress=False,)
apple.reset_index(inplace=True)

prediction_range = apple.loc[(apple['Date'] > datetime(2013,1,1)) & (apple['Date'] < datetime(2018,1,1))]
plt.plot(apple['Date'], apple['Close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")
plt.show()

"""
Train AAPL data
"""

close_data = apple.filter(['Close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

from sklearn.preprocessing import MinMaxScaler

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

model = keras.models.Sequential()

model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1))) # https://keras.io/api/layers/recurrent_layers/lstm/
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5)) # Dropout layer ignores set of random neurons, prevents overfitting
model.add(keras.layers.Dense(1))
model.summary()

"""
Gradient descent is an optimization algorithm which is commonly-used to train machine learning models and neural networks.  
Training data helps these models learn over time, and the cost function within gradient descent specifically acts as a barometer, 
gauging its accuracy with each iteration of parameter updates. Until the function is close to or equal to zero, the model will 
continue to adjust its parameters to yield the smallest possible error. The cost (or loss) function measures the difference, or error, 
between actual y and predicted y at its current position. https://www.ibm.com/topics/gradient-descent
"""

"""
- optimizer: method that helps to optimize the cost function by using gradient descent.
- loss: function by which we monitor whether the model is improving with training or not.
- metrics: evaluates the model by predicting the training and the validation data.
"""

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=15)

"""
Predict Test Data
"""

"""
A larger MSE indicates that the data points are dispersed widely around its central moment (mean), 
whereas a smaller MSE suggests the opposite. Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. 
https://https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/#:~:text=Root%20
Mean%20Square%20Error%20(RMSE)%20is%20the%20standard%20deviation%20of,the%20line%20of%20best%20fit.
"""

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

train = apple[:training]
test = apple[training:]
test['Predictions'] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train['Date'], train['Close'])
plt.plot(test['Date'], test[['Close', 'Predictions']])
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Download stock data. 
"""
def load_default(ticket):
  stock = yf.download(ticket)
  stock.reset_index(inplace=True)
  return stock

"""
Download stock data over period. 
"""
def load_dates(ticket, start_date, end_date):
  stock = yf.download(ticket, start=start_date, end=end_date)
  stock.reset_index(inplace=True)
  return stock

"""
Calculate Daily Return (day-to-day volatility) and add to dataframe. 
"""
calcDailyReturn(stock):
  stock['Daily Return'] = stock['Adj Close'].pct_change(1)

"""
returns mean of stock price
"""
def calcmean(stock):
  return stock['Adj Close'].mean()

"""
determine number of days SP500 closed near average price (avg +/- 5).
"""
def nearness(stock):
  stock['boolean'] = stock['Adj Close'].between(math.floor(mean_price)-5, math.ceil(mean_price)+5)
  print(stock['boolean'].value_counts())

"""
Returns standard deviation of stock price using numpy.
"""
def calcstd(stock):
  return np.std(stock['Adj Close'])

"""
generates a plot of stock price. lines depict one and two standard deviations 
from the average price. 
"""
def plotstd(ticket, start_date, end_date):

  stock = load_dates(ticket, start_date, end_date)

  prices = stock['Adj Close']
  mean = stock['Adj Close'].mean()
  std = (np.std(stock['Adj Close']))
  min_value = min(prices)
  max_value = max(prices)

  plt.figure(figsize=(15,7.5))
  plt.title(f"{stock} Standard Deviation")
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.ylim(min_value - 10, max_value + 10)

  plt.scatter(x = stock.index, y=stock['Adj Close'])
  plt.hlines(y=mean, xmin=0, xmax=len(prices), color = 'green', linewidth=2.5, label="average price")
  plt.hlines(y=mean - std, xmin=0, xmax=len(prices), color='orange', linestyle="dashed", label="1 std")
  plt.hlines(y=mean + std, xmin=0, xmax=len(prices), color='orange', linestyle="dashed")
  plt.hlines(y=mean - 2*std, xmin=0, xmax=len(prices), color='red', linestyle="dotted", label="2 std")
  plt.hlines(y=mean + 2*std, xmin=0, xmax=len(prices), color='red',  linestyle="dotted")
  plt.legend()

  plt.show()

if __name__ == "__main__":
  stock = load_default('LYFT')
  print("average price is" + calcmean(stock))
  nearness(stock)
  print("standard deviation is" + calcstd(stock))
  plotstd('LYFT', '2020-01-01', '2020-12-31')

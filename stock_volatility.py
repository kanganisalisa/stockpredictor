import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

SP500 = yf.download('SPY', start = '2020-01-01', end='2020-12-31')
SP500.reset_index(inplace=True)

"""
Daily Return (day-to-day volatility)
"""

SP500['Daily Return'] = SP500['Adj Close'].pct_change(1)

"""
Standard Deviation (longer-term volatility) 
provides an intuition of how much a stock price differs from its average value over a specific period.
"""

mean_price = SP500['Adj Close'].mean()
mean_price

# determine number of days SP500 closed near average price (avg +/- 5)
SP500['boolean'] = SP500['Adj Close'].between(math.floor(mean_price)-5, math.ceil(mean_price)+5)
SP500['boolean'].value_counts()

"""
manual standard deviation calculation:
```
SP500['Difference'] = SP500['Adj Close'] - SP500['Adj Close'].mean(axis=0)
SP500['Difference_Squared'] = SP500['Difference']**2
sum = SP500['Difference_Squared'].sum() / len(SP500['Difference_Squared'])
std = np.sqrt(sum)
```
"""

std = (np.std(SP500['Adj Close'])) # standard deviation using numpy

"""
generates a plot of stock price. lined depict 1 and 2 standard deviations from the 
average price. 
"""
def plotstd(ticket, start_date, end_date):

  stock = yf.download(ticket, start=start_date, end=end_date)
  stock.reset_index(inplace=True)

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

# call fcn
plotstd('LYFT', '2020-01-01', '2020-12-31')

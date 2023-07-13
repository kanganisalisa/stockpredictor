import pandas as pd
import yfinance as yf
from datetime import datetime

class Stock:

    """
    Download stock data between start_date and end_date.
    """
    def __init__(self, ticket, start_date, end_date, data=None):
      self.ticket = ticket
      self.start_date = start_date
      self.end_date = end_date
      self.data = yf.download(ticket, start=start_date, end=end_date)
      self.data.reset_index(inplace=True)
      self.data['Date'] = pd.to_datetime(self.data['Date'])

    def getTicket(self):
       return self.ticket

    def getDates(self):
       return self.start_date, self.end_date


"""
accessing ticker data (from https://pypi.org/project/yfinance/)

msft = yf.Ticker("MSFT")

# get all stock info
msft.info

# get historical market data
hist = msft.history(period="1mo")

# show meta information about the history (requires history() to be called first)
msft.history_metadata

# show actions (dividends, splits, capital gains)
msft.actions
msft.dividends
msft.splits
msft.capital_gains  # only for mutual funds & etfs

# show share count
msft.get_shares_full(start="2022-01-01", end=None)

# show financials:
# - income statement
msft.income_stmt
msft.quarterly_income_stmt
# - balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet
# - cash flow statement
msft.cashflow
msft.quarterly_cashflow
# see `Ticker.get_income_stmt()` for more options

# show holders
msft.major_holders
msft.institutional_holders
msft.mutualfund_holders

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default. 
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
msft.earnings_dates

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options

# show news
msft.news

# get option chain for specific expiration
opt = msft.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts
"""
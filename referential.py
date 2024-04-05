from os import path
import yfinance as yf
import numpy as np
import pandas as pd

class downloader():
    def __init__(self, tickers, startdate, enddate):
        self.tickers = tickers
        self.startdate = startdate
        self.enddate = enddate

    def download(self, in_folder):
        for ticker in tickers:
            self.df = yf.download(tickers=ticker,
                                  start=self.startdate,
                                  end=self.enddate)
            self.df.to_csv(path.join(in_folder, ticker + '.csv'))

class data():
    def __init__(self) -> None:
        pass

if __name__ == "__main__":

    project_folder = path.abspath(path.dirname(__file__))
    in_folder = path.join(project_folder, 'instruments')

    tickers = ['NKE']
    end = '2023-12-31'
    start = '2006-01-01'
    generator = downloader(tickers=tickers, startdate=start, enddate=end)
    generator.download(in_folder=in_folder)
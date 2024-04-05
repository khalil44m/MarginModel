import numpy as np
import pandas as pd
from DateTime import DateTime
import os
import glob
from os import path
from pathlib import Path
import math as m
import logging
from sys import stdout

import matplotlib.pyplot as plt

# def marketData(in_folder):
#     # filelist = [file for file in os.listdir(in_folder) if file.endswith('.csv')]
    
#     all_files = glob.glob(os.path.join(in_folder, "*.csv"))
#     list = []
#     for f in all_files:
#         df = pd.read_csv(f,)
#     df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
#     return df

def price_returns(in_folder):
    dfm = pd.DataFrame()
    for file in os.listdir(in_folder):
        df = pd.read_csv(path.join(in_folder, file))
        data = df[['Date','Close']]
        data['Ticker'] = Path(path.join(in_folder, file)).stem
        data['log_returns'] = np.log1p(df['Close'].pct_change())
        dfm = pd.concat([data, dfm], axis=0, )
        # try:
        #     dfm = pd.merge(data, dfm, how='outer', left_index=True, right_on='z')
        # except IndexError:
        #     dfm = data if not data.empty else dfm
        # data_.merge(data.dropna(), right_on='rkey')
    return dfm.dropna()

def devolatilize(log_rt, N, mpor, j):

    seed = 200
    lambda_ = .99
    sig_ = np.mean(np.power(log_rt[:seed],2))
    vols = [m.sqrt(sig_)]
    inv = [log_rt[-N - mpor + 1]/m.sqrt(sig_)]
    for i in range(2, N + mpor):
        sig2 = lambda_ * vols[-1]**2 + (1 - lambda_) * log_rt[-N - mpor + i - j]**2
        vols.append(np.sqrt(sig2))

        inv_norm = np.sign(log_rt[-N - mpor + i - j]) * min(np.abs(log_rt[-N - mpor + i - j]/m.sqrt(sig2)), 30)
        inv.append(inv_norm)

    return vols, inv

def scenario_gen(vols, inv, log_rt, N, mpor):
    lambda_ = .99
    sig = m.sqrt(lambda_ * vols[-1]**2 + (1 - lambda_) * log_rt[-1]**2)
    rt = []
    for i in range(0, N):
        rt.append(sig * sum(inv[i: i + mpor]))

    return rt[::-1]

def pnl(rt, mktvalue):
    pnl = mktvalue * (np.exp(rt) - 1)
    return sorted(pnl)

def expected_shortfall(sorted_pnl, alpha, N):
    index = int(np.floor(N * (1 - alpha)))
    expected_shortfall = np.mean(sorted_pnl[:index])
    return expected_shortfall

class Portfolio():
    def __init__(self, df, tickers, qty):
        self.df = df
        self.tickers = tickers
        self.last = []
        for ticker in self.tickers:
            self.last.append(self.df[self.df['Ticker'] == ticker]['Close'].iloc[-1])
        self.qty = qty

    def pos_value(self, ticker):
        # mktval = 0
        idx = self.tickers.index(ticker)
        return self.qty[idx] * self.last[idx]
        # for t, p, q in zip(self.tickers, self.prices, self.qty):
        #     mktval += self.price * self.qty

    def historical_pnl(self):
        historical_pnl = []
        for d in self.df.Date.unique():
            df_ = self.df[self.df.Date == d]
            pnl_ = 0
            for ticker in self.tickers:
                df__ = df_[df_['Ticker'] == ticker]
                pnl_ += self.pos_value(ticker) * df__['log_returns'].iloc[0]
            historical_pnl.append(pnl_)
        return historical_pnl


if __name__ == "__main__":
    logging.basicConfig(stream=stdout, level=logging.DEBUG)
    project_folder = path.abspath(path.dirname(__file__))
    in_folder = path.join(project_folder, 'instruments')

    Tickers = []
    for f in os.listdir(in_folder):
        Tickers.append((Path(in_folder) / f).stem)

    df = price_returns(in_folder)
    position = Portfolio(df, ['AAPL', 'AMZN', 'GOOGL'], [100, 200, 400])

    es_ = []

    L = min([len(df[df['Ticker'] == ticker]) for ticker in Tickers])
    for j in range(L - 702,0,-1):
        logging.info('IM on ' + df['Date'].to_list()[-j])
        es = 0
        for ticker in Tickers:
            df_ = df[df['Ticker'] == ticker]
            vols, inv = devolatilize(df_['log_returns'].to_list(), 700, 3, j)
            rt = scenario_gen(vols, inv, df_['log_returns'].to_list()[:- j], 700, 3)
            pnl_ = pnl(rt, position.pos_value(ticker))
            es += expected_shortfall(pnl_, 0.95, 700)
        es_.append(es)

    # bool_ = position.marketvalue() * df['log_returns'][702:] > np.array(es_)
    historical_pnl = position.historical_pnl()[702:] # * bool_
    # breaches = position.marketvalue() * df['log_returns'][702:] * ~bool_

    plt.plot(df['Date'].unique()[702:], es_)

    col = np.where(np.array(historical_pnl) > np.array(es_), 'b', 'r')
    plt.scatter(df['Date'].unique()[702:], historical_pnl, c = col, marker='x')
    plt.show()
    a = 0
from statsmodels.tsa import x13
from pynowcasting import CRBVAR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

np.set_printoptions(precision=2, suppress=True, linewidth=150)
pd.options.display.max_columns = 50
pd.options.display.width = 200

# data_path = "/Users/gustavoamarante/Dropbox/BWGI/Nowcast/china_data.xlsx"  # iMac
data_path = "C:/Users/gamarante/Dropbox/BWGI/Nowcast/china_data.xlsx"  # BW

df_mf = pd.read_excel(data_path, index_col='Date', sheet_name='Data')
df_mf = df_mf.resample('M').last()

# ===== Apply logs =====
var2log = ['Industrial Value Added (NSA)',
           'Buildings Sold (NSA)',
           'Consumer Confidence (NSA)',
           'GDP (SA)']

for var in var2log:
    df_mf[var] = np.log(df_mf[var])

# ===== Seasonal adjustment =====
var2dessaz = ['Industrial Value Added (NSA)',
              'Buildings Sold (NSA)',
              'Consumer Confidence (NSA)']

os.chdir(r"G:\Gustavo Amarante\x13\x13as")
for var in var2dessaz:
    x13results = x13.x13_arima_analysis(endog=df_mf[var].dropna(), outlier=True, print_stdout=True)
    df_mf[var] = x13results.seasadj


bvar = CRBVAR(data=df_mf, lags=5, verbose=True, hz=24, mcmc=False, fcast=False,
              mnalpha=True, mnpsi=True, sur=True, noc=True)

# np.exp(df_mf['GDP (SA)']).plot(marker='o')
# np.exp(bvar.smoothed_states['GDP (SA)']).plot()
# plt.show()

# Grab only the desired forecasts
last_observed_gdp = df_mf['GDP (SA)'].dropna().index[-1]
forecasts = (np.exp(bvar.smoothed_states['GDP (SA)'])).resample('Q').last().rolling(4).sum().pct_change(4)
forecasts = forecasts[forecasts.index > last_observed_gdp]  # TODO this is what I need to save

forecasts_qoq = (np.exp(bvar.smoothed_states['GDP (SA)'])).resample('Q').last().pct_change(1)
forecasts_qoq = forecasts_qoq[forecasts_qoq.index > last_observed_gdp]
print(forecasts_qoq)

# Growth chart
np.exp(df_mf['GDP (SA)']).dropna().rolling(4).sum().pct_change(4).plot(marker='o')
forecasts.plot()
plt.show()

# Growth QoQ chart
# np.exp(df_mf['GDP (SA)']).dropna().pct_change(1).plot(marker='o')
# forecasts_qoq.plot()
# plt.show()

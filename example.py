from nowcasting import BVARGLP
from scipy.io import loadmat
import pandas as pd
import numpy as np

# ===== USER DEFINED PARAMETERS =====
show_charts = True
estimateBVAR = True # TODO If False, load previosly estimated parameters
unconditional_forecasts = True  # TODO If True, run the unconditional forecasts
vint = '2020-01-31'

# ====== LOAD DATA =====
data_path = '/Users/gustavoamarante/Dropbox/BWGI/Nowcast/HBVAR/rawdata/DataRaw_2020_0131.mat'
mat = loadmat(data_path)

transform_dict = {'Real Gross Domestic Product': 'log',
                  'Real Personal Consumption Expenditures': 'log',
                  'Real Private Residential Fixed Investment': 'log',
                  'Real Private Nonresidential Fixed Investment': 'log',
                  'Real Exports of Goods & Services': 'log',
                  'Real Imports of Goods & Services': 'log',
                  'Real Government Consumption Expenditures & Gross Investment': 'log',
                  'Real Federal Consumption Expenditures & Gross Investment': 'log',
                  'Gross Domestic Product: Chain Price Index': 'log',
                  'Personal Consumption Expenditures: Chain Price Index': 'log',
                  'PCE less Food & Energy: Chain Price Index': 'log',
                  'CPI-U: All Items': 'log',
                  'CPI-U: All Items Less Food & Energy': 'log',
                  'Business Sector: Real Output Per Hour of All Persons': 'log',
                  'Business Sector: Compensation Per Hour': 'log',
                  'Business Sector: Unit Labor Cost': 'log',
                  'All Employees: Total Nonfarm': 'log',
                  'Civilian Unemployment Rate: 16 yr +': 'lin',
                  'Industrial Production Index': 'log',
                  'Capacity Utilization: Manufacturing': 'lin',
                  'Housing Starts': 'log',
                  'Personal Saving Rate': 'lin',
                  'ISM Mfg: PMI Composite Index': 'lin',
                  'University of Michigan: Consumer Sentiment': 'lin',
                  'Federal Tax Receipts on Corporate Income': 'log',
                  'Federal Tax Base for Personal Income': 'log',
                  'Federal Tax Base for Corporate Income': 'log',
                  'Federal Tax Receipts on Personal Income': 'log',
                  '2-Year Treasury Note Yield at Constant Maturity': 'lin',
                  '5-Year Treasury Note Yield at Constant Maturity': 'lin',
                  '10-Year Treasury Note Yield at Constant Maturity': 'lin',
                  'Moodys Seasoned Aaa Corporate Bond Yield': 'lin',
                  'Moodys Seasoned Baa Corporate Bond Yield': 'lin',
                  'Nominal Trade-Weighted Exch Value of US$ vs Major Currencies': 'log',
                  'Stock Price Index: Standard & Poors 500 Composite': 'log',
                  'Spot Oil Price: West Texas Intermediate': 'log',
                  'S&P GSCI Non-Energy Commodities Nearby Index': 'log',
                  'Standard & Poors 500 Composite Realized Volatility': 'lin'}


index_range = pd.to_datetime(mat['Date'].flatten() - 719529, unit='D')
df_raw = pd.DataFrame(data=mat['DataRaw'], index=index_range, columns=list(transform_dict.keys()))

# Transform Data
df_trans = pd.DataFrame()
for var in transform_dict.keys():
    if transform_dict[var] == 'log':
        df_trans[var] = 100 * np.log(df_raw[var])  # TODO why multiply by 100?
    else:
        df_trans[var] = df_raw[var]

df_trans = df_trans.dropna()

# ===== NOWCAST =====
bvar = BVARGLP(data=df_trans, lags=5)

a = 1

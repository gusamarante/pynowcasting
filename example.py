from scipy.io import loadmat
import pandas as pd

# ===== USER DEFINED PARAMETERS =====
show_charts = True
estimateBVAR = True # TODO If False, load previosly estimated parameters
unconditional_forecasts = True  # TODO If True, run the unconditional forecasts
vint = '2020-01-31'

# ====== LOAD DATA =====
data_path = '/Users/gustavoamarante/Dropbox/BWGI/Nowcast/HBVAR/rawdata/DataRaw_2020_0131.mat'
mat = loadmat(data_path)


# pd.to_datetime(mat['Date'].flatten() - 719529, unit='D')
a = 1

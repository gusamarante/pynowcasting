from pynowcasting import CRBVAR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=150)
pd.options.display.max_columns = 50
pd.options.display.max_columns = 50

# data_path = "/Users/gustavoamarante/Dropbox/BWGI/Nowcast/mf_data.xlsx"  # Mac
data_path = "C:/Users/gamarante/Dropbox/BWGI/Nowcast/mf_data.xlsx"  # BW

df_mf = pd.read_excel(data_path, index_col=0)
df_mf = df_mf.resample('M').last()

bvar = CRBVAR(data=df_mf, lags=5, verbose=True,
              mnalpha=True, mnpsi=True, sur=True, noc=True, mcmc=False)

bvar.smoothed_states.plot()
plt.show()

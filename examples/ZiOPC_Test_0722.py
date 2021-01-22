import numpy as np
import pandas as pd
# import this after importing all other packages.
from zmiopc import ziopc
import matplotlib.pyplot as plt
import seaborn as sns

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
# Ziop and ziopc examples
# Specifying Xs, Zs, and Y
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
data = DAT

pstartziop = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4])

pstart = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4, .1])  # These are correct pstart
##Numbers all over the place (copied from R codes)
pstartx = np.array([-0.77, 0.90, 18.78, -2, .2, 0.04, -0.09, 0.26, 1.70, -0.42, -.1])

ziopc_JCR = ziopc.ziopcmod(pstart, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

ziop_JCR = ziopc.ziopmod(pstartziop, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

fitttedziopc = ziopc.ziopcfit(ziopc_JCR)
fitttedziop = ziopc.ziopfit(ziop_JCR)

print(ziopc_JCR.coefs)
print(ziop_JCR.coefs)
print(JCR_OP.AIC)
print(JCR_OP.llik)
print(JCR_OP.vcov)

# OP Model
pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
# Ziop and ziopc examples
# Specifying Xs, Zs, and Y
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Y = ['rep_civwar_DV']
data = DAT
JCR_OP = ziopc.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)

# Plots

sns.set(style="ticks", color_codes=True)

num_bins = 3
n, bins, patches = plt.hist(yx_, num_bins, facecolor='blue', rwidth=0.9)
data = data.dropna(how='any')
data['rep_civwar_DV'] = data['rep_civwar_DV'].astype(int)
sns.catplot(x='rep_civwar_DV', kind="count", palette="hls", data=data)

# Vuong test
ziopc.vuong_opziop(JCR_OP, ziop_JCR)
ziopc.vuong_opziopc(JCR_OP, ziopc_JCR)

# Box plots for predicted probabilities
ziopparl = ziopc.split_effects(ziop_JCR, 2)
ziopcparl = ziopc.split_effects(ziopc_JCR, 2)

ziopparl.plot.box(grid='False')
ziopcparl.plot.box(grid='False')

ziopord = ordered_effects(ziop_JCR, 1)
ziopcord = ordered_effects(ziopc_JCR, 1)

ziopord.plot.box(grid='False')
ziopcord.plot.box(grid='False')

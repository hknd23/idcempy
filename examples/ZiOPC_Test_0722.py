import numpy as np
import pandas as pd
# import this after importing all other packages.
from idcempy import zmiopc
import matplotlib.pyplot as plt
import seaborn as sns

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
# Ziop and ziopc examples
# Specifying Xs, Zs, and Y
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Xsmall = ['logGDPpc', 'parliament', 'disaster']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
data = DAT

pstartziop = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4])

pstartziopsmall = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2])

pstartziopc = np.array([-1.31, .32, 2.5, -.21,
                        .2, -0.2, -0.4, 0.2, .9, -.4, .1])

# These are correct pstart
##Numbers all over the place (copied from R codes)
pstartx = np.array(
    [-0.77, 0.90, 18.78, -2, .2, 0.04, -0.09, 0.26, 1.70, -0.42, -.1])

ziopc_JCR = zmiopc.iopcmod('ziopc',
                           data, X, Y, Z, pstart=pstartziopc, method='bfgs',
                           weights=1,
                           offsetx=0, offsetz=0)

ziop_JCR = zmiopc.iopmod('ziop',
                         data, X, Y, Z, pstart=pstartziop,
                         method='bfgs', weights=1,
                         offsetx=0,
                         offsetz=0)

ziopc_JCR_test = zmiopc.iopcmod('ziopc', data, X, Y, Z)

ziop_JCR = zmiopc.iopmod('ziop', data, X, Y, Z)

ziop_JCRsmall = zmiopc.iopmod('ziop', pstartziopsmall,
                              data, Xsmall, Y, Z, method='bfgs', weights=1,
                              offsetx=0, offsetz=0)

# ziopc_JCR.coefs.to_csv("ZIOPC_0131.csv")
# ziop_JCR.coefs.to_csv("ZIOP_0131.csv")


fitttedziopc = zmiopc.iopcfit(ziopc_JCR)
fitttedziop = zmiopc.iopfit(ziop_JCR)
fitttedmiopc = zmiopc.iopcfit(miopc_JCR)
fitttedmiop = zmiopc.iopfit(miop_JCR)

print(ziopc_JCR.coefs)
print(ziop_JCR.coefs)
print(JCR_OP.AIC)
print(JCR_OP.llik)
print(JCR_OP.vcov)

# OP Model
pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

array1 = np.array([1, 2, 3])
array2 = np.array([1, 2, 3])
list1 = [1, 2, 3]
list2 = [1, 2, 3]
pstartcut = [-1, 0.3]
pstartx = [-0.2, -0.5, 0.2, .9, -.4]

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
# Ziop and ziopc examples
# Specifying Xs, Zs, and Y
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Y = ['rep_civwar_DV']
data = DAT
JCR_OP = zmiopc.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)

# Plots

sns.set(style="ticks", color_codes=True)

num_bins = 3
n, bins, patches = plt.hist(yx_, num_bins, facecolor='blue', rwidth=0.9)
data = data.dropna(how='any')
data['rep_civwar_DV'] = data['rep_civwar_DV'].astype(int)
sns.catplot(x='rep_civwar_DV', kind="count", palette="hls", data=data)

# Vuong test
zmiopc.vuong_opiop(JCR_OP, ziop_JCR)
zmiopc.vuong_opiopc(JCR_OP, ziopc_JCR)

# Box plots for predicted probabilities
ziopparl = zmiopc.split_effects(ziop_JCR, 2)
ziopcparl = zmiopc.split_effects(ziopc_JCR, 2)

ziopparl.plot.box(grid='False')
ziopcparl.plot.box(grid='False')

ziopord = ordered_effects(ziop_JCR, 1)
ziopcord = ordered_effects(ziopc_JCR, 1)

ziopord.plot.box(grid='False')
ziopcord.plot.box(grid='False')

# MiOP Examples

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/EUKnowledge.dta")

vars = ["EU_support_ET", "polit_trust", "Xenophobia", "discuss_politics",
        "univers_ed", "Professional",
        "Executive", "Manual", "Farmer", "Unemployed", "rural", "female", "age",
        "EU_Know_obj", "Lie_EU_Know",
        "student", "EUbid_Know", "income", "dk", "dkORlie", "EU_Know_subj",
        "TV", "Educ_high", "Educ_high_mid",
        "Educ_low_mid"]

datax = DAT[vars]
datasetnew = datax.dropna(how='any')

Y = ["EU_support_ET"]
X = ['polit_trust', 'Xenophobia', 'discuss_politics', 'Professional',
     'Executive', 'Manual', 'Farmer',
     'Unemployed', 'rural', 'female', 'age', 'student', 'income', 'Educ_high',
     'Educ_high_mid', 'Educ_low_mid']
Z = ['discuss_politics', 'rural', 'female', 'age', 'student',
     'EUbid_Know', 'EU_Know_obj', 'TV', 'Educ_high', 'Educ_high_mid',
     'Educ_low_mid']

b = np.repeat(.01, 30)
bc = np.repeat(.01, 31)

MIOPEUx = zmiopc.iopmod('miop', b, datasetnew, X, Y, Z, method='bfgs', weights=1,
                        offsetx=0, offsetz=0)
MIOPcEUx = zmiopc.iopcmod('miopc', bc, datasetnew, X, Y, Z, method='bfgs',
                          weights=1, offsetx=0, offsetz=0)

fitttedmiop = zmiopc.iopfit(MIOPEUx)
fitttedmiopc = zmiopc.iopcfit(MIOPcEUx)

fitttedmiopc = ziopcfit(ziopc_JCR)

DATw = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication I/Replication I/Least_Informed.dta")
yy = ['votechoice_dk']
xx = ['phone_only', 'age']
zz = ['cXdem','cXrep']

boiii = zmiopc.iopcmod('miop', DATw, xx, yy, zz,
                       method='BFGS')
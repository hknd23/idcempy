"""Classes and Functions for the ziopcpy module."""
import numpy as np
from numpy import *
# ZiOPC model converges extremely
# faster with import * rather than import as np.
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import mvn
from zmiopc import biml

DAT = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication II/Replication "
    "II/replicationdata.dta", convert_categoricals=False)

x = ['educ', 'female', 'black', 'hispanic', 'party7', 'w3gbvalu2',
     'presbatt',
     'south', 'gayban2', 'agegroup2', 'bornagn_w', 'brnag_wXgmb2', 'catholic',
     'cathXgmb2', 'other_rel', 'secular', 'secXgmb2', 'ideo', 'w3mobidx']
z = ['educ', 'agegroup2', 'w3mobidx', 'secular']
DAT['votenum'] = pd.factorize(DAT['vote_turn'])[0]
y = ['votenum']

baseline = np.array([0, 1, 2])
baselines = np.array([0, 2, 1])
model = biml.bimnlmod(DAT, x, y, z, baseline,
                      method='BFGS')
models = biml.bimnlmod(DAT, x, y, z, baselines,
                       method='BFGS')
biml.bimnl3(pstart, x2, x3, y, z, baseline, data)
biml.bimnl3(pstart, x2, x3, y, z, baselines, data)

# 4 cats
DAT2 = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication I/Replication I/Least_Informed.dta")

x2 = ['canvass_all', 'phone_only', 'sid_dem', 'sid_rep', 'cXdem', 'cXrep',
      'poXdem',
      'poXrep', 'hhcount', 'age', 'female', 'vg04', 'd156']
z2 = ['hhcount', 'age', 'female', 'vg04', 'd156']
y2 = ['votechoice_dk']
baseline2 = np.array([1, 0, 2, 3])
baseline3 = np.array([1, 3, 2, 0])

model2 = biml.bimnlmod(DAT2, x2, y2, z2, baseline2,
                       method='BFGS')
model3 = biml.bimnlmod(DAT2, x2, y2, z2, baseline3,
                       method='BFGS')

varlist = np.unique(y + z + x)
dataset = data[varlist]
datasetnew = dataset.dropna(how='any')
datasetnew = datasetnew.reset_index(drop=True)
x_ = datasetnew[x]
y_ = datasetnew[y]
yx_ = y_.iloc[:, 0]
yncat = len(np.unique(yx_))
if yncat == 3:
    modeltype = 'bimnl3'
elif yncat == 4:
    modeltype = 'bimnl4'
else:
    print("Function only supports 3 or 4 categories.")
z_ = datasetnew[z]
z_.insert(0, 'int', np.repeat(1, len(z_)))
x_.insert(0, 'int', np.repeat(1, len(x_)))
if modeltype == 'bimnl3':
    x2 = x_
    x3 = x_
    if pstart is None:
        pstart = np.repeat(.01, (len(x2.columns) + len(x3.columns)
                                 + len(z_.columns)))

elif modeltype == 'bimnl4':
    x2 = x_
    x3 = x_
    x4 = x_
    if pstart is None:
        pstart = np.repeat(.01, (len(x2.columns) + len(x3.columns)
                                 + len(x4.columns) + len(z_.columns)))

biml.bimnl4(pstart, x2, x3, x4, yx_, z_, baseline2, datasetnew)
biml.bimnl4(pstart, x2, x3, x4, yx_, z_, baseline3, datasetnew)

n = len(data)
ycat = y.astype('category')
ycatu = np.unique(ycat)
yncat = len(ycatu)
b2 = pstart[len(z.columns):(len(z.columns) + len(x2.columns))]
b3 = pstart[(len(z.columns) + len(x2.columns)):
            (len(z.columns) + len(x2.columns) + len(x3.columns))]
b4 = pstart[(len(z.columns) + len(x2.columns) + len(x3.columns)):
            (len(pstart))]
gamma = pstart[0:(len(z.columns))]
xb2 = x2.dot(b2)
xb3 = x3.dot(b3)
xb4 = x4.dot(b4)
zg = z.dot(gamma)
pz = 1 / (1 + np.exp(-zg))
p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3) + np.exp(xb4))
p2 = p1 * exp(xb2)
p3 = p1 * exp(xb3)
p4 = p1 * exp(xb4)
lgp1 = np.log((1 - pz) + pz * p1) * (y == baseline[0])
lgp2 = np.log(pz * p2) * (y == baseline[1])
lgp3 = np.log(pz * p3) * (y == baseline[2])
lgp4 = np.log(pz * p4) * (y == baseline[3])

lik = (sum(np.log((1 - pz) + pz * p1) * (y == baseline[0])) +
       sum(np.log(pz * p2) * (y == baseline[1])) +
       sum(np.log(pz * p3) * (y == baseline[2])) +
       sum(np.log(pz * p4) * (y == baseline[3])))

llik = -1 * sum(lik)

lprobs = pd.DataFrame({"lgp1": lgp1, "lgp2": lgp2,
                       "lgp3": lgp3, "lgp4": lgp4})
lprobscols = ['lgp1', 'lgp2', 'lgp3', 'lgp4']
lprobscols2 = ['', '', '', '']
for j in range(yncat):
    lprobscols2[j] = lprobscols[baseline2[j]]
lprobs_inf = lprobs[lprobscols2]
lik = np.zeros((n, yncat))
y0 = np.sort(ycatu)
for j in range(yncat):
    v[:, j] = y == y0[j]
for i in range(n):
    for j in range(yncat):
        lik[i, j] = v[i, j] * lprobs_inf.iloc[i, j]
likk = sum(lik[lik != 0])
llik = -1 * sum(likk)

import pandas as pd
import numpy as np
from zmiopc import bimnl

DAT = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication II/Replication "
    "II/replicationdata.dta", convert_categoricals=False)
DATw = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication II/Replication "
    "II/replicationdata.dta")

x = ['educ', 'female', 'black', 'hispanic', 'party7', 'w3gbvalu2',
     'presbatt',
     'south', 'gayban2', 'agegroup2', 'bornagn_w', 'brnag_wXgmb2', 'catholic',
     'cathXgmb2', 'other_rel', 'secular', 'secXgmb2', 'ideo', 'w3mobidx']
z = ['educ', 'agegroup2', 'w3mobidx', 'secular']
DAT['votenum'] = pd.factorize(DAT['vote_turn'])[0]
y = ['vote_turn']

order = [0, 1, 2]
orders = [0, 2, 1]
second_order = [1, 0, 2]
second_order2 = [2, 0, 1]
torder = [2, 1, 0]
torders = [1, 2, 0]

binflatecat = "baseline"
sinflatecat = "second"
tinflatecat = "third"

model = bimnl.imnlmod(DAT, x, y, z, order, binflatecat,
                      method='BFGS')
models = bimnl.imnlmod(DAT, x, y, z, orders, binflatecat,
                       method='BFGS')

smodel = bimnl.imnlmod(DAT, x, y, z, second_order, sinflatecat,
                       method='BFGS')
smodels = bimnl.imnlmod(DAT, x, y, z, second_order2, sinflatecat,
                        method='BFGS')

tmodel = bimnl.imnlmod(DAT, x, y, z, torder, tinflatecat,
                       method='BFGS')
tmodels = bimnl.imnlmod(DAT, x, y, z, torders, tinflatecat,
                        method='BFGS')

print(model.coefs)
print(models.coefs)

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
else:
    print("Function only supports 3 categories.")
z_ = datasetnew[z]
z_.insert(0, 'int', np.repeat(1, len(z_)))
x_.insert(0, 'int', np.repeat(1, len(x_)))
if modeltype == 'bimnl3':
    x2 = x_
    x3 = x_
    if pstart is None:
        pstart = np.repeat(.01, (len(x2.columns) + len(x3.columns)
                                 + len(z_.columns)))

b2 = pstart[len(z.columns):(len(z.columns) + len(x2.columns))]
b3 = pstart[(len(z.columns) + len(x2.columns)):(len(pstart))]
gamma = pstart[0:(len(z.columns))]
xb2 = x2.dot(b2)
xb3 = x3.dot(b3)
zg = z.dot(gamma)
pz = 1 / (1 + np.exp(-zg))
p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
p2 = p1 * np.exp(xb2)
p3 = p1 * np.exp(xb3)

lik = sum(np.log(pz * p1) * (y == second_order[0]) +
          np.log((1 - pz) + pz * p2) * (y == second_order[1]) +
          np.log(pz * p3) * (y == second_order[2]))
llik = -1 * sum(lik)


likx = sum(np.log(pz * p1) * (y == second_order2[0]) +
           np.log((1 - pz) + pz * p2) * (y == second_order2[1]) +
           np.log(pz * p3) * (y == second_order2[2]))

liksss = (pz * p1) + ((1 - pz) + pz * p2)  + (pz * p3)
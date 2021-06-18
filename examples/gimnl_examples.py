import pandas as pd
from idcempy import gimnl

DAT = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication II/Replication "
    "II/replicationdata.dta", convert_categoricals=False)

x = ['educ', 'female', 'black', 'hispanic', 'party7', 'w3gbvalu2',
     'presbatt',
     'south', 'gayban2', 'agegroup2', 'bornagn_w', 'brnag_wXgmb2', 'catholic',
     'cathXgmb2', 'other_rel', 'secular', 'secXgmb2', 'ideo', 'w3mobidx']
z = ['educ', 'agegroup2', 'w3mobidx', 'secular']
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

modelmnl = gimnl.mnlmod(DAT, x, y, order)

model = gimnl.gimnlmod(DAT, x, y, z, order, binflatecat)

gimnl.vuong_gimnl(modelmnl, model)
models = gimnl.gimnlmod(DAT, x, y, z, orders, binflatecat)

smodel = gimnl.gimnlmod(DAT, x, y, z, second_order, sinflatecat,
                        method='BFGS')
smodels = gimnl.gimnlmod(DAT, x, y, z, second_order2, sinflatecat,
                         method='BFGS')

tmodel = gimnl.gimnlmod(DAT, x, y, z, torder, tinflatecat,
                        method='BFGS')
tmodels = gimnl.gimnlmod(DAT, x, y, z, torders, tinflatecat,
                         method='BFGS')

print(model.coefs)
print(models.coefs)

x2 = ['educ', 'party7', 'agegroup2']
z2 = ['educ', 'agegroup2']
y2 = ['vote_turn']

order = [0, 1, 2]
orders = [0, 2, 1]

model_small = gimnl.gimnlmod(DAT, x2, y2, z2, order, binflatecat)
mnl_small = gimnl.mnlmod(DAT, x2, y2, order)
gimnl.vuong_gimnl(mnl_small, model_small)

models_small = gimnl.gimnlmod(DAT, x2, y2, z2, orders, binflatecat)

# Failed Tests_ more than three categories

DAT2 = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication I/Replication "
    "I/Least_Informed.dta", convert_categoricals=False)

xf = ["canvass_all", "age"]
zf = ["age"]
yf = ["votechoice_dk"]
orderf = [0, 1, 2, 3]

model_small = gimnl.gimnlmod(DAT2, xf, yf, zf, orderf, inflatecat="baseline")

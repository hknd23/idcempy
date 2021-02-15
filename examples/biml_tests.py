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

model = gimnl.gimnlmod(DAT, x, y, z, order, binflatecat,
                       )
models = gimnl.gimnlmod(DAT, x, y, z, orders, binflatecat,
                        )

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

varlist = np.unique(y + z + x)
dataset = data[varlist]
datasetnew = dataset.dropna(how="any")
datasetnew = datasetnew.reset_index(drop=True)
x_ = datasetnew[x]
y_ = datasetnew[y]
yx_ = y_.iloc[:, 0]
yncat = len(np.unique(yx_))
x_.insert(0, "int", np.repeat(1, len(x_)))
x2 = x_
x3 = x_
pstart=None
if pstart is None:
    pstart = np.repeat(
        0.01, (len(x2.columns) + len(x3.columns)))

imnl3(pstart, x2, x3, yx_, reference)


def imnl3(pstart, x2, x3, y, reference):
    """
    Likelihood function for the baseline inflated three-category MNL model.

    :param pstart: starting parameters.
    :param x2: X covariates.
    :param x3: X covariates (should be identical to x2.
    :param y: Dependent Variable (DV).
    :param z: Inflation stage covariates.
    :param reference: order of categories (first category/baseline inflated).
    """
    b2 = pstart[0: len(x2.columns)]
    b3 = pstart[len(x2.columns): (len(pstart))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = np.sum(
        np.log(p1) * (y == reference[0])
        + np.log(p2) * (y == reference[1])
        + np.log(p3) * (y == reference[2])
    )
    llik = -1 * np.sum(lik)
    return llik



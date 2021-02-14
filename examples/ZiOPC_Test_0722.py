import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import mvn, norm
# import this after importing all other packages.
from idcempy import zmiopc

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

print(ziopc_JCR.coefs)
print(ziop_JCR.coefs)

# OP Model
pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

array1 = np.array([1, 2, 3])
array2 = np.array([1, 2, 3])
list1 = [1, 2, 3]
list2 = [1, 2, 3]
pstartcut = [-1, 0.3]
pstartx = [-0.2, -0.5, 0.2, .9, -.4]

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Y = ['rep_civwar_DV']
data = DAT
JCR_OP = zmiopc.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)

# Vuong test
zmiopc.vuong_opiop(JCR_OP, ziop_JCR)
zmiopc.vuong_opiopc(JCR_OP, ziopc_JCR)

# Box plots for predicted probabilities
ziopparl = zmiopc.split_effects(ziop_JCR, 2)
ziopcparl = zmiopc.split_effects(ziopc_JCR, 2)

ziopparl.plot.box(grid='False')
ziopcparl.plot.box(grid='False')

ziopord = zmiopc.ordered_effects(ziop_JCR, 1)
ziopcord = zmiopc.ordered_effects(ziopc_JCR, 1)

ziopord.plot.box(grid='False')
ziopcord.plot.box(grid='False')

# MiOP Examples

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/EUKnowledge.dta")

Y = ["EU_support_ET"]
X = ['Xenophobia', 'discuss_politics']
Z = ['discuss_politics', 'EU_Know_obj']

miopc_EU = zmiopc.iopcmod('miopc', DAT, X, Y, Z)
op_EU = zmiopc.opmod(DAT, X, Y)

zmiopc.vuong_opiopc(op_EU, miopc_EU)

miopc_EU_xeno = zmiopc.ordered_effects(miopc_EU, 0)
miopc_EU__diss = zmiopc.ordered_effects(miopc_EU, 1)

miopc_EU_xeno.plot.box(grid='False')
miopc_EU__diss.plot.box(grid='False')

miopc_EU_diss = zmiopc.split_effects(miopc_EU, 1)
miopc_EU_know = zmiopc.split_effects(miopc_EU, 2)

miopc_EU_diss.plot.box(grid='False')
miopc_EU_know.plot.box(grid='False')

# Tobacco


DAT = pd.read_csv("C:/Users/Nguyen/Google "
                  "Drive/zmiopc/zmiopc/data/tobacco_cons.csv")
X = ['age', 'grade', 'gender_dum']
Y = ['cig_count']
Z = ['gender_dum']
pstart = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01])

ziopc_tob = zmiopc.iopcmod('ziopc', DAT, X, Y, Z)
ziop_tob = zmiopc.iopmod('ziop', DAT, X, Y, Z)

op_tob = zmiopc.opmod(DAT, X, Y)

zmiopc.vuong_opiopc(op_tob, ziopc_tob)
zmiopc.vuong_opiop(op_tob, ziop_tob)


ziopcage = zmiopc.ordered_effects(ziopc_tob, 0)
ziopcgrade = zmiopc.ordered_effects(ziopc_tob, 1)
ziopcgender = zmiopc.ordered_effects(ziopc_tob, 2)

ziopcage = ordered_effects(ziopc_tob, 0)
ziopcgrade = ordered_effects(ziopc_tob, 1)
ziopcgender = ordered_effects(ziopc_tob, 2)

ziopcage.plot.box(grid='False')
ziopcgrade.plot.box(grid='False')
ziopcgender.plot.box(grid='False')

ziopcgenders = zmiopc.split_effects(ziopc_tob, 1)
g = split_effects(ziopc_tob, 1)
ziopcgenders.plot.box(grid='False')
g.plot.box(grid='False')


def split_effects(model, inflvar, nsims=10000):
    """Calculate change in probability of being 0 in the split-probit stage.

    This function calculates the predicted probabilities
    when there is change in value of a variable in the split-probit equation.
    The chosen dummy variable is changed from 0 to 1,
    and chosen numerical variable is mean value + 1 standard deviation.
    Other variables are kept at 0 or mean value
    (Note: the current version of the function
    recognize ordinal variables as numerical).

    :param model: :class:`IopModel` or :class:`IopCModel`.
    :param inflvar: int representing the location of variable
        in the split-probit equation.
        (attribute .inflate of :class:`IopModel` or :class:`IopCModel`)
    :param nsims: number of simulated observations, default to 10000.
    :return: changeprobs: a dataframe of the predicted
        probabilities when there is change in the variable (1)
        versus original values (0).
    """
    estimate = model.coefs.iloc[:, 0]
    vcov = model.vcov
    model_z = model.Z
    zsim1 = np.zeros(len(model_z.columns))
    zsim1[0] = 1
    zsima = np.zeros(len(model_z.columns))
    zsima[0] = 1
    for j in range(1, len(model_z.columns)):
        if (
                max(model_z.iloc[:, j]) == 1
                and min(model_z.iloc[:, j]) == 0
                and len(np.unique(model_z.iloc[:, j])) == 2
        ):
            zsim1[j] = 0
        else:
            zsim1[j] = np.mean(model_z.iloc[:, j])
    for j in range(1, len(model_z.columns)):
        if (
                max(model_z.iloc[:, j]) == 1
                and min(model_z.iloc[:, j]) == 0
                and len(np.unique(model_z.iloc[:, j])) == 2
        ):
            zsima[j] = 1
        else:
            zsima[j] = np.mean(model_z.iloc[:, j]) + np.std(model_z.iloc[:, j])
    zsim2 = zsim1.copy()
    zsim2[inflvar] = zsima[inflvar]
    np.random.seed(1)
    probs1 = np.zeros(nsims)
    probs2 = np.zeros(nsims)
    for i in range(nsims):
        gsim = np.random.multivariate_normal(estimate, vcov)
        gsim2 = gsim[model.ycat - 1: model.ycat - 1 + len(model.inflate)]
        zg1 = zsim1.dot(gsim2)
        zg2 = zsim2.dot(gsim2)
        probs1[i] = norm.cdf(zg1)
        probs2[i] = norm.cdf(zg2)
    name = model.coefs.index[model.ycat - 1 + inflvar]
    changeprobs = pd.DataFrame({name.replace("Inflation: ", "") + "= 0": probs1,
                                name.replace("Inflation: ", "") + "= 1": probs2}
                               )
    return changeprobs


def ordered_effects(model, ordvar, nsims=10000):
    """Calculate the changes in probability in each outcome in OP stage.

    This function calculates predicted probabilities
    when there is change in value of a variable
    in the ordered probit equation.
    The chosen dummy variable is changed from 0 to 1,
    and chosen numerical variable is mean value + 1 standard deviation.
    Other variables are kept at 0 or mean value
    (Note: the current version of the function
    recognize ordinal variables as numerical).

    :param model: :class:`IopModel` or :class:`IopCModel`.
    :param ordvar: int representing the location of variable
        in the ordered probit equation.
        (attribute .ordered of :class:`IopModel` or :class:`IopCModel`)
    :param nsims: number of simulated observations, default to 10000.
    :return: changeprobs: a dataframe of the predicted
        probabilities when there is change in the variable for each outcome (1)
        versus original values (0).
    """
    estimate = model.coefs.iloc[:, 0]
    vcov = model.vcov
    model_x = model.X
    xsim1 = np.zeros(len(model_x.columns))
    xsima = np.zeros(len(model_x.columns))
    for j in range(len(model_x.columns)):
        if (
                max(model_x.iloc[:, j]) == 1
                and min(model_x.iloc[:, j]) == 0
                and len(np.unique(model_x.iloc[:, j])) == 2
        ):
            xsim1[j] = 0
        else:
            xsim1[j] = np.mean(model_x.iloc[:, j])
    for j in range(len(model_x.columns)):
        if (
                max(model_x.iloc[:, j]) == 1
                and min(model_x.iloc[:, j]) == 0
                and len(np.unique(model_x.iloc[:, j])) == 2
        ):
            xsima[j] = 1
        else:
            xsima[j] = np.mean(model_x.iloc[:, j]) + np.std(model_x.iloc[:, j])
    xsim2 = xsim1.copy()
    xsim2[ordvar] = xsima[ordvar]
    np.random.seed(1)
    probsordered1 = np.zeros(model.ycat)
    probsordered2 = np.zeros(model.ycat)
    cprobs = np.zeros((model.ycat - 1, 1))
    cprobs[0, 0] = model.cutpoints[0]
    for j in range(1, model.ycat - 1):
        cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
    probs1 = pd.DataFrame(index=np.arange(nsims),
                          columns=np.arange(model.ycat))
    probs2 = pd.DataFrame(index=np.arange(nsims),
                          columns=np.arange(model.ycat))
    name = model.coefs.index[model.ycat - 1 + len(model.inflate) + ordvar]
    probs1 = probs1.add_suffix(": " + name.replace("Ordered: ", "") + " = 0")
    probs2 = probs2.add_suffix(": " + name.replace("Ordered: ", "") + " = 1")
    for i in range(nsims):
        bsim = np.random.multivariate_normal(estimate, vcov)
        bsim2 = bsim[model.ycat - 1 + len(model.inflate):
                     model.ycat - 1 + len(model.inflate) + len(model.ordered)]
        xb1 = xsim1.dot(bsim2)
        xb2 = xsim2.dot(bsim2)
        probsordered1[model.ycat - 1] = 1 - norm.cdf(
            cprobs[model.ycat - 2, 0] - xb1)
        probsordered1[0] = norm.cdf(cprobs[0, 0] - xb1)
        for j in range(1, model.ycat - 1):
            probsordered1[j] = norm.cdf(cprobs[j, 0] - xb1) - (
                norm.cdf(cprobs[j - 1, 0] - xb1)
            )
        probsordered2[model.ycat - 1] = 1 - norm.cdf(
            cprobs[model.ycat - 2, 0] - xb2)
        probsordered2[0] = norm.cdf(cprobs[0, 0] - xb2)
        for j in range(1, model.ycat - 1):
            probsordered2[j] = norm.cdf(cprobs[j, 0] - xb2) - (
                norm.cdf(cprobs[j - 1, 0] - xb2)
            )
        probs1.iloc[i:, ] = probsordered1
        probs2.iloc[i:, ] = probsordered2
    changeprobs = pd.DataFrame(index=np.arange(nsims),
                               columns=np.arange(2 * model.ycat))
    newnames = list(np.repeat("", model.ycat * 2))
    for j in range(0, 2 * model.ycat, 2):
        changeprobs.iloc[:, j] = probs1.iloc[:, round(j / 2)]
    for j in range(1, 2 * model.ycat, 2):
        changeprobs.iloc[:, j] = probs2.iloc[:, round((j - 1) / 2)]
    for j in range(0, 2 * model.ycat, 2):
        newnames[j] = list(probs1.columns)[round(j / 2)]
    for j in range(1, 2 * model.ycat, 2):
        newnames[j] = list(probs2.columns)[round((j - 1) / 2)]
    changeprobs.columns = newnames
    return changeprobs

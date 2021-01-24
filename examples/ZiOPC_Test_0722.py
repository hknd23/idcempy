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

ziopc_JCR = ziopc.iopcmod('ziopc', pstart, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

ziop_JCR = ziopc.iopmod('ziop', pstartziop, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)


miopc_JCR = ziopc.iopcmod('miopc', pstart, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

miop_JCR = ziopc.iopmod('miop', pstartziop, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

fitttedziopc = ziopc.iopcfit(ziopc_JCR)
fitttedziop = ziopc.iopfit(ziop_JCR)
fitttedmiopc = ziopc.iopcfit(miopc_JCR)
fitttedmiop = ziopc.iopfit(miop_JCR)


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

# MiOP Examples

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/EUKnowledge.dta")

vars = ["EU_support_ET", "polit_trust", "Xenophobia", "discuss_politics", "univers_ed", "Professional",
        "Executive", "Manual", "Farmer", "Unemployed", "rural", "female", "age", "EU_Know_obj", "Lie_EU_Know",
        "student", "EUbid_Know", "income", "dk", "dkORlie", "EU_Know_subj", "TV", "Educ_high", "Educ_high_mid",
        "Educ_low_mid"]

datax = DAT[vars]
datasetnew = datax.dropna(how='any')

Y = ["EU_support_ET"]
X = ['polit_trust', 'Xenophobia', 'discuss_politics', 'Professional', 'Executive', 'Manual', 'Farmer',
     'Unemployed', 'rural', 'female', 'age', 'student', 'income', 'Educ_high', 'Educ_high_mid', 'Educ_low_mid']
Z = ['discuss_politics', 'rural', 'female', 'age', 'student',
     'EUbid_Know', 'EU_Know_obj', 'TV', 'Educ_high', 'Educ_high_mid', 'Educ_low_mid']

b = np.repeat(.01, 30)
bc = np.repeat(.01, 31)


MIOPEUx = ziopc.iopmod('miop', b, datasetnew, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)
MIOPcEUx = ziopc.iopcmod('miopc', bc, datasetnew, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

fitttedmiop = ziopc.iopfit(MIOPEUx)
fitttedmiopc = ziopc.iopcfit(MIOPcEUx)


fitttedmiopc = ziopcfit(ziopc_JCR)



def ziopcfit(model):
    """Calculate fitted probabilities from :py:func:`iopcmod`.

    :param model: ZiopCModel object from iopcmod()
    :return: FittedVals object with fitted values
    """
    zg = model.Z.dot(model.inflate)
    xb = model.X.dot(model.ordered)
    cprobs = np.zeros((model.ycat - 1, 1))
    n = len(model.data)
    probs = np.zeros((n, model.ycat))
    cprobs[0, 0] = model.cutpoints[0]
    rho = model.coefs.iloc[-1, 0]
    means = np.array([0, 0])
    lower = np.array([-inf, -inf])
    sigma = np.array([[1, rho], [rho, 1]])
    nsigma = np.array([[1, -rho], [-rho, 1]])
    if model.modeltype == 'ziop':
        for j in range(1, model.ycat - 1):
            cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
        for i in range(n):
            probs[i, model.ycat - 1] = mvn.mvnun(
                lower, [zg[i], (xb[i] -
                                cprobs[model.ycat - 2][0])], means, sigma)[0]
            probs[i, 0] = ((1 - norm.cdf(zg[i]))
                           + mvn.mvnun(lower,
                                       [zg[i], (cprobs[0][0] - xb[i])],
                                       means, nsigma)[0])
        for i in range(n):
            for j in range(1, model.ycat - 1):
                probs[i, j] = ((mvn.mvnun(lower,
                                          [zg[i], (cprobs[j][0] - xb[i])],
                                          means, nsigma)[0])
                               - (mvn.mvnun(lower,
                                            [zg[i], (cprobs[j - 1][0] - xb[i])],
                                            means, nsigma)[0]))
    elif model.modeltype == 'miopc':
        for j in range(1, model.ycat - 1):
            cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
        for i in range(n):
            probs[i, model.ycat - 1] = mvn.mvnun(
                lower, [zg[i], (xb[i] -
                                cprobs[model.ycat - 2][0])], means, sigma)[0]
            probs[i, 0] = mvn.mvnun(lower,
                                    [zg[i], (cprobs[0][0] - xb[i])],
                                    means, nsigma)[0]
        for i in range(1, model.ycat - 1):
            if i == median(range(model.ycat)):
                probs[:, i] = ((1 - norm.cdf(zg))
                               + ((mvn.mvnun(lower,
                                             [zg[i], (cprobs[j][0] - xb[i])],
                                             means, nsigma)[0])
                                  - (mvn.mvnun(lower,
                                               [zg[i], (cprobs[j - 1][0] - xb[i])],
                                               means, nsigma)[0])))
            else:
                probs[i, j] = ((mvn.mvnun(lower,
                                          [zg[i], (cprobs[j][0] - xb[i])],
                                          means, nsigma)[0])
                               - (mvn.mvnun(lower,
                                            [zg[i], (cprobs[j - 1][0] - xb[i])],
                                            means, nsigma)[0]))

    # ordered
    probsordered = np.zeros((n, model.ycat))
    probsordered[:, model.ycat - 1] = (1 - norm.cdf(cprobs[model.ycat - 2, 0]
                                                    - xb))
    probsordered[:, 0] = (norm.cdf(cprobs[0, 0] - xb))
    for j in range(1, model.ycat - 1):
        probsordered[:, j] = (norm.cdf(cprobs[j, 0] - xb)
                              - (norm.cdf(cprobs[j - 1, 0] - xb)))

    probsinfl = np.zeros((n, 1))
    probsinfl[:, 0] = 1 - norm.cdf(zg)
    # linear
    probslin = pd.DataFrame({"zg": zg, "xb": xb})
    fitted = FittedVals(probs, probsordered, probsinfl, probslin)
    return fitted

model=miopc_JCR
model=ziopc_JCR

zg = model.Z.dot(model.inflate)
xb = model.X.dot(model.ordered)
cprobs = np.zeros((model.ycat - 1, 1))
n = len(model.data)
probs = np.zeros((n, model.ycat))
cprobs[0, 0] = model.cutpoints[0]
rho = model.coefs.iloc[-1, 0]
means = np.array([0, 0])
lower = np.array([-inf, -inf])
sigma = np.array([[1, rho], [rho, 1]])
nsigma = np.array([[1, -rho], [-rho, 1]])
if model.modeltype == 'ziopc':
    for j in range(1, model.ycat - 1):
        cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
    for i in range(n):
        probs[i, model.ycat - 1] = mvn.mvnun(
            lower, [zg[i], (xb[i] -
                            cprobs[model.ycat - 2][0])], means, sigma)[0]
        probs[i, 0] = ((1 - norm.cdf(zg[i]))
                       + mvn.mvnun(lower,
                                   [zg[i], (cprobs[0][0] - xb[i])],
                                   means, nsigma)[0])
    for i in range(n):
        for j in range(1, model.ycat - 1):
            probs[i, j] = ((mvn.mvnun(lower,
                                      [zg[i], (cprobs[j][0] - xb[i])],
                                      means, nsigma)[0])
                           - (mvn.mvnun(lower,
                                        [zg[i], (cprobs[j - 1][0] - xb[i])],
                                        means, nsigma)[0]))
elif model.modeltype == 'miopc':
    for j in range(1, model.ycat - 1):
        cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
    for i in range(n):
        probs[i, model.ycat - 1] = mvn.mvnun(
            lower, [zg[i], (xb[i] -
                            cprobs[model.ycat - 2][0])], means, sigma)[0]
        probs[i, 0] = mvn.mvnun(lower,
                                [zg[i], (cprobs[0][0] - xb[i])],
                                means, nsigma)[0]
    for i in range(n):
        for j in range(1, model.ycat - 1):
            if j == median(range(model.ycat)):
                probs[i, j] = ((1 - norm.cdf(zg[i]))
                               + ((mvn.mvnun(lower,
                                             [zg[i], (cprobs[j][0] - xb[i])],
                                             means, nsigma)[0])
                                  - (mvn.mvnun(lower,
                                               [zg[i], (cprobs[j - 1][0] - xb[i])],
                                               means, nsigma)[0])))
            else:
                probs[i, j] = ((mvn.mvnun(lower,
                                          [zg[i], (cprobs[j][0] - xb[i])],
                                          means, nsigma)[0])
                               - (mvn.mvnun(lower,
                                            [zg[i], (cprobs[j - 1][0] - xb[i])],
                                            means, nsigma)[0]))

# ordered
probsordered = np.zeros((n, model.ycat))
probsordered[:, model.ycat - 1] = (1 - norm.cdf(cprobs[model.ycat - 2, 0]
                                                - xb))
probsordered[:, 0] = (norm.cdf(cprobs[0, 0] - xb))
for j in range(1, model.ycat - 1):
    probsordered[:, j] = (norm.cdf(cprobs[j, 0] - xb)
                          - (norm.cdf(cprobs[j - 1, 0] - xb)))

probsinfl = np.zeros((n, 1))
probsinfl[:, 0] = 1 - norm.cdf(zg)
# linear
probslin = pd.DataFrame({"zg": zg, "xb": xb})
fitted = FittedVals(probs, probsordered, probsinfl, probslin)

sums= np.zeros(len(model.data))
for i in range(len(model.data)):
    sums[i]=sum(probs[i])
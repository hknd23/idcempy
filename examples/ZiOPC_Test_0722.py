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
ziopc.vuong_opiop(JCR_OP, ziop_JCR)
ziopc.vuong_opiopc(JCR_OP, ziopc_JCR)

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

model=miopc_JCR
model=ziopc_JCR

varlist = np.unique(Y + Z + X)
datasetnew = data[varlist]
datasetnew = datasetnew.dropna(how='any')
datasetnew = datasetnew.reset_index(drop=True)
x = datasetnew[X]
yx = datasetnew[Y]
y = yx.iloc[:, 0]
z = datasetnew[Z]
z.insert(0, 'ones', np.repeat(1, len(z)))


n = len(datasetnew)
ycat = y.astype('category')
ycatu = np.unique(ycat)
yncat = len(ycatu)
y0 = np.sort(ycatu)
V = np.zeros((len(datasetnew), yncat))
for j in range(yncat):
    V[:, j] = y == y0[j]
tau = np.repeat(1.0, yncat)
for i in range(yncat - 1):
    if i == 0:
        tau[i] = pstart[i]
    else:
        tau[i] = tau[i - 1] + exp(pstart[i])
beta = pstart[(yncat + len(z.columns) - 1): len(pstart) - 1]
gamma = pstart[(yncat - 1): (yncat + len(z.columns) - 1)]
X_beta = x.dot(beta)
rho = pstart[len(pstart) - 1]
cprobs = np.zeros((len(X_beta), yncat))
probs = np.zeros((len(X_beta), yncat))
cutpoint = np.zeros((len(X_beta), yncat))
cutpointb = np.zeros((len(X_beta), yncat))
zg = z.dot(gamma) + offsetz
xb = x.dot(beta) + offsetx
means = np.array([0, 0])
lower = np.array([-inf, -inf])
sigma = np.array([[1, rho], [rho, 1]])
nsigma = np.array([[1, -rho], [-rho, 1]])
for i in range(yncat - 1):
    cprobs[:, i] = norm.cdf(tau[i] - xb)
    cutpoint[:, i] = tau[i] - xb
    cutpointb[:, i] = xb - tau[i]
upperb = np.zeros((len(X_beta), 2))
upper = np.zeros((len(X_beta), 2))
for j in range(n):
    upperb[j, :] = [zg[j], cutpointb[j, yncat - 2]]
    upper[j, :] = [zg[j], cutpoint[j, 0]]
    probs[j, yncat - 1] = mvn.mvnun(lower, upperb[j], means, sigma)[0]
    probs[j, 0] = mvn.mvnun(lower, upper[j], means, nsigma)[0]
for i in range(n):
    for j in range(1, yncat - 1):
        if j == median(range(yncat)):
            probs[i, j] = ((1 - norm.cdf(zg[i]))
                           + mvn.mvnun(lower, [zg[i], cutpoint[i, j]],
                                       means, nsigma)[0]
                           - mvn.mvnun(lower, [zg[i], cutpoint[i, j - 1]],
                                       means, nsigma)[0])
        else:
            probs[i, j] = (mvn.mvnun(lower, [zg[i], cutpoint[i, j]],
                                     means, nsigma)[0]
                           - mvn.mvnun(lower, [zg[i], cutpoint[i, j - 1]],
                                       means, nsigma)[0])
lik = np.zeros((n, yncat))
for k in range(n):
    for j in range(yncat):
        lik[k, j] = V[k, j] * probs[k, j]
likk = np.log(lik[lik != 0])
llik = -1 * sum(likk * weights)

import numpy as np
import pandas as pd
import time
import os
from idcempy import zmiopc

DAT = pd.read_stata(
    os.getcwd() + "/data/bp_exact_for_analysis.dta",
    convert_categoricals=False)

# Ziop and ziopc examples
# Specifying Xs, Zs, and Y

X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
data = DAT

pstartziop = [-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4]

pstartziopc = [-1.31, .32, 2.5, -.21,
               .2, -0.2, -0.4, 0.2, .9, -.4, .1]


start_time = time.time()
ziopc_JCR = zmiopc.iopcmod('ziopc',
                           data, X, Y, Z, pstart=pstartziopc, method='bfgs',
                           weights=1,
                           offsetx=0, offsetz=0)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
ziop_JCR = zmiopc.iopmod('ziop',
                         data, X, Y, Z, pstart=pstartziop,
                         method='bfgs', weights=1,
                         offsetx=0,
                         offsetz=0)
print("--- %s seconds ---" % (time.time() - start_time))


#OP Model
pstartop = [-1, 0.3, -0.2, -0.5, 0.2, .9, -.4]

start_time = time.time()
JCR_OP = zmiopc.opmod(data, X, Y,
                      pstart=pstartop, method='bfgs', weights=1,
                      offsetx=0)
print("--- %s seconds ---" % (time.time() - start_time))



# MiOP Examples

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/EUKnowledge.dta")

Y = ["EU_support_ET"]
X = ['Xenophobia', 'discuss_politics']
Z = ['discuss_politics', 'EU_Know_obj']

miopc_EU = zmiopc.iopcmod('miopc', DAT, X, Y, Z)
miop_EU = zmiopc.iopmod('miop', DAT, X, Y, Z)

op_EU = zmiopc.opmod(DAT, X, Y)

zmiopc.vuong_opiopc(op_EU, miopc_EU)
zmiopc.vuong_opiop(op_EU, miop_EU)

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
                  "Drive/zmiopc/idcempy/data/tobacco_cons.csv")
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

ziopcage.plot.box(grid='False')
ziopcgrade.plot.box(grid='False')
ziopcgender.plot.box(grid='False')

ziopcgenders = zmiopc.split_effects(ziopc_tob, 1)

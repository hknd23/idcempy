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

# OP Model
pstartop = [-1, 0.3, -0.2, -0.5, 0.2, .9, -.4]

start_time = time.time()
JCR_OP = zmiopc.opmod(data, X, Y,
                      pstart=pstartop, method='bfgs', weights=1,
                      offsetx=0)
print("--- %s seconds ---" % (time.time() - start_time))

# Tobacco


DAT = pd.read_csv(os.getcwd() + "/data/tobacco_cons.csv")
X = ['age', 'grade', 'gender_dum']
Y = ['cig_count']
Z = ['gender_dum']

start_time = time.time()
ziopc_tob = zmiopc.iopcmod('ziopc', DAT, X, Y, Z)
model_time = time.time() - start_time
print("%s seconds" % model_time)

start_time = time.time()
ziop_tob = zmiopc.iopmod('ziop', DAT, X, Y, Z)
model_time = time.time() - start_time
print("%s seconds" % model_time)

start_time = time.time()
op_tob = zmiopc.opmod(DAT, X, Y)
model_time = time.time() - start_time
print("%s seconds" % model_time)

zmiopc.vuong_opiopc(op_tob, ziopc_tob)
zmiopc.vuong_opiop(op_tob, ziop_tob)

ziopcage = zmiopc.ordered_effects(ziopc_tob, 0)
ziopcgrade = zmiopc.ordered_effects(ziopc_tob, 1)
ziopcgender = zmiopc.ordered_effects(ziopc_tob, 2)

ziopcage.plot.box(grid='False')
ziopcgrade.plot.box(grid='False')
ziopcgender.plot.box(grid='False')

ziopcgenders = zmiopc.split_effects(ziopc_tob, 1)

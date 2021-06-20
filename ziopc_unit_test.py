import pandas as pd
from idcempy import zmiopc
import os
import unittest
import numpy as np

"""
Replication of results from 
Bagozzi, Benjamin E., Daniel W. Hill Jr., Will H. Moore, and Bumba Mukherjee .
 2015. "Modeling Two Types of Peace: The Zero-inflated Ordered Probit (ZiOP) 
 Model in Conflict Research."  Journal of Conflict Resolution. 59(4): 728-752.
"""

DAT = pd.read_stata(
    os.getcwd() + "/data/bp_exact_for_analysis.dta",
    convert_categoricals=False)

# Ziop and ziopc examples
# Specifying Xs, Zs, and Y
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
data = DAT

pstartziop = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4])

pstartziopsmall = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2])

pstartziopc = np.array([-1.31, .32, 2.5, -.21,
                        .2, -0.2, -0.4, 0.2, .9, -.4, .1])

pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

varlist = np.unique(Y + Z + X)
dataset = data[varlist]
datasetnew = dataset.dropna(how="any")
datasetnew = datasetnew.reset_index(drop=True)
x_ = datasetnew[X]
y_ = datasetnew[Y]
yx_ = y_.iloc[:, 0]
yncat = len(np.unique(yx_))
z_ = datasetnew[Z]
z_.insert(0, "ones", np.repeat(1, len(z_)))


class TestZiopLlike(unittest.TestCase):
    def test_zioploglike(self):
        self.assertAlmostEqual(zmiopc.ziop(pstartziop, x_, yx_, z_,
                                           datasetnew, 1, 0, 0), 1486.169,
                               places=0)


class TestZiopCLlike(unittest.TestCase):
    def test_ziopcloglike(self):
        self.assertAlmostEqual(zmiopc.ziopc(pstartziopc, x_, yx_, z_,
                                           datasetnew, 1, 0, 0), 1487.748,
                               places=0)


class TestOp(unittest.TestCase):
    def test_opmodel(self):
        self.assertAlmostEqual(zmiopc.opmod(data, X, Y,
                                            pstart=pstartop, method='bfgs',
                                            weights=1,
                                            offsetx=0).coefs.iloc[2, 0],
                               -0.21, places=0)


class TestZiop(unittest.TestCase):
    def test_ziopmodel(self):
        self.assertAlmostEqual(zmiopc.iopmod('ziop',
                                             data, X, Y, Z, pstart=pstartziop,
                                             method='bfgs', weights=1,
                                             offsetx=0,
                                             offsetz=0).coefs.iloc[4, 0],
                               -0.29, places=0)
        self.assertAlmostEqual(zmiopc.iopmod('ziop',
                                             data, X, Y, Z, pstart=pstartziop,
                                             method='bfgs', weights=1,
                                             offsetx=0,
                                             offsetz=0).coefs.iloc[5, 0],
                               0.04, places=0)


class TestZiopc(unittest.TestCase):
    def test_ziopcmodel(self):
        self.assertAlmostEqual(zmiopc.iopcmod('ziopc',
                                              data, X, Y, Z,
                                              pstart=pstartziopc,
                                              method='bfgs', weights=1,
                                              offsetx=0,
                                              offsetz=0).coefs.iloc[-1, 0],
                               -0.889, places=0)
        self.assertAlmostEqual(zmiopc.iopcmod('ziopc',
                                              data, X, Y, Z,
                                              pstart=pstartziopc,
                                              method='bfgs', weights=1,
                                              offsetx=0,
                                              offsetz=0).coefs.iloc[4, 0],
                               -0.37, places=0)
        self.assertAlmostEqual(zmiopc.iopcmod('ziopc',
                                              data, X, Y, Z,
                                              pstart=pstartziopc,
                                              method='bfgs', weights=1,
                                              offsetx=0,
                                              offsetz=0).coefs.iloc[5, 0],
                               0.33, places=0)


class VuongOpZiopc(unittest.TestCase):
    def test_vuongopziopc(self):
        self.assertAlmostEqual(zmiopc.vuong_opiop(zmiopc.opmod(data, X, Y,
                                                               pstart=pstartop,
                                                               method='bfgs',
                                                               weights=1,
                                                               offsetx=0),
                                                  zmiopc.iopmod('ziop',
                                                                data, X, Y, Z,
                                                                pstart=
                                                                pstartziop,
                                                                method='bfgs',
                                                                weights=1,
                                                                offsetx=0,
                                                                offsetz=0)),
                               -4.909, places=0)
        self.assertAlmostEqual(zmiopc.vuong_opiopc(zmiopc.opmod(data, X, Y,
                                                                pstart=
                                                                pstartop,
                                                                method='bfgs',
                                                                weights=1,
                                                                offsetx=0),
                                                   zmiopc.iopcmod('ziopc',
                                                                  data, X, Y,
                                                                  Z,
                                                                  pstart=
                                                                  pstartziopc,
                                                                  method='bfgs'
                                                                  , weights=1,
                                                                  offsetx=0,
                                                                  offsetz=0)),
                               -5.424, places=0)

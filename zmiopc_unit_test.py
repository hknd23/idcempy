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

JCR_OP = zmiopc.opmod(data, X, Y,
                      pstart=pstartop, method='bfgs', weights=1,
                      offsetx=0)
ziopc_JCR = zmiopc.iopcmod('ziopc',
                           data, X, Y, Z, pstart=pstartziopc, method='bfgs',
                           weights=1,
                           offsetx=0, offsetz=0)

ziop_JCR = zmiopc.iopmod('ziop',
                         data, X, Y, Z, pstart=pstartziop,
                         method='bfgs', weights=1,
                         offsetx=0,
                         offsetz=0)


class TestOp(unittest.TestCase):
    def test_opmodel(self):
        self.assertAlmostEqual(zmiopc.opmod(data, X, Y,
                                            pstart=pstartop, method='bfgs',
                                            weights=1,
                                            offsetx=0).coefs.iloc[2, 0],
                               -0.21, places=1)


class TestZiop(unittest.TestCase):
    def test_ziopmodel(self):
        self.assertAlmostEqual(zmiopc.iopmod('ziop',
                                             data, X, Y, Z, pstart=pstartziop,
                                             method='bfgs', weights=1,
                                             offsetx=0,
                                             offsetz=0).coefs.iloc[4, 0],
                               -0.29, places=1)
        self.assertAlmostEqual(zmiopc.iopmod('ziop',
                                             data, X, Y, Z, pstart=pstartziop,
                                             method='bfgs', weights=1,
                                             offsetx=0,
                                             offsetz=0).coefs.iloc[5, 0],
                               0.04, places=1)


class TestZiopc(unittest.TestCase):
    def test_ziopcmodel(self):
        self.assertAlmostEqual(zmiopc.iopcmod('ziopc',
                                              data, X, Y, Z,
                                              pstart=pstartziopc,
                                              method='bfgs', weights=1,
                                              offsetx=0,
                                              offsetz=0).coefs.iloc[-1, 0],
                               -0.889, places=2)
        self.assertAlmostEqual(zmiopc.iopcmod('ziopc',
                                              data, X, Y, Z,
                                              pstart=pstartziopc,
                                              method='bfgs', weights=1,
                                              offsetx=0,
                                              offsetz=0).coefs.iloc[4, 0],
                               -0.3, places=1)
        self.assertAlmostEqual(zmiopc.iopcmod('ziopc',
                                              data, X, Y, Z,
                                              pstart=pstartziopc,
                                              method='bfgs', weights=1,
                                              offsetx=0,
                                              offsetz=0).coefs.iloc[5, 0],
                               0.3, places=1)


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
                               -4.909, places=3)
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
                               -5.424, places=3)

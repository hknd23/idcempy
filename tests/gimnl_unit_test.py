import pandas as pd
from idcempy import gimnl
import os
import unittest
import numpy as np

"""
Replication of results from 
Bagozzi, Benjamin E. and Kathleen Marchetti. 2017. 
"Distinguishing Occasional Abstention 
rom Routine Indifference in Models of Vote Choice." 
Political Science Research  and Methods.  5(2): 277-249.
"""

DAT = pd.read_stata(
    os.getcwd() + "/data/replicationdata.dta", convert_categoricals=False)

x = ['educ', 'female', 'black', 'hispanic', 'party7', 'w3gbvalu2',
     'presbatt',
     'south', 'gayban2', 'agegroup2', 'bornagn_w', 'brnag_wXgmb2', 'catholic',
     'cathXgmb2', 'other_rel', 'secular', 'secXgmb2', 'ideo', 'w3mobidx']
z = ['educ', 'agegroup2', 'w3mobidx', 'secular']
y = ['vote_turn']

order_Kerry = [0, 1, 2]
order_Bush = [0, 2, 1]

binflatecat = "baseline"
sinflatecat = "second"
tinflatecat = "third"

varlist = np.unique(y + z + x)
dataset = DAT[varlist]
datasetnew = dataset.dropna(how="any")
datasetnew = datasetnew.reset_index(drop=True)
x_ = datasetnew[x]
y_ = datasetnew[y]
yx_ = y_.iloc[:, 0]
yncat = len(np.unique(yx_))
z_ = datasetnew[z]
z_.insert(0, "int", np.repeat(1, len(z_)))
x_.insert(0, "int", np.repeat(1, len(x_)))

pstart_mnl = np.repeat(0.01, len(x_.columns) + len(x_.columns))
pstart_gimnl = np.repeat(0.01, len(x_.columns) + len(x_.columns)
                         + len(z_.columns))


class TestBimnlLlike(unittest.TestCase):
    def test_volume(self):
        self.assertEqual(gimnl.bimnl3(pstart_gimnl, x_, x_, yx_, z_,
                                      [0, 1, 2]),
                         gimnl.bimnl3(pstart_gimnl, x_, x_, yx_, z_,
                                      [0, 2, 1]))
        self.assertAlmostEqual(gimnl.bimnl3(pstart_mnl, x_, x_, yx_, z_
                                            [0, 1, 2]), 2109.38, places=2)


class TestSimnlTimnlLlike(unittest.TestCase):
    def test_volume(self):
        self.assertEqual(gimnl.simnl3(pstart_gimnl, x_, x_, yx_, z_,
                                      [2, 0, 1]),
                         gimnl.timnl3(pstart_gimnl, x_, x_, yx_, z_,
                                      [2, 1, 0]))
        self.assertAlmostEqual(gimnl.simnl3(pstart_gimnl, x_, x_, yx_, z_,
                                            [2, 0, 1]), 2242.67, places=2)
        self.assertEqual(gimnl.simnl3(pstart_gimnl, x_, x_, yx_, z_,
                                      [1, 0, 2]),
                         gimnl.timnl3(pstart_gimnl, x_, x_, yx_, z_,
                                      [1, 2, 0]))
        self.assertAlmostEqual(gimnl.timnl3(pstart_gimnl, x_, x_, yx_, z_,
                                            [1, 2, 0]), 2242.88, places=2)


class TestMnlLlike(unittest.TestCase):
    def test_volume(self):
        self.assertEqual(gimnl.mnl3(pstart_mnl, x_, x_, yx_, [0, 1, 2]),
                         gimnl.mnl3(pstart_mnl, x_, x_, yx_, [0, 2, 1]))
        self.assertAlmostEqual(gimnl.mnl3(pstart_mnl, x_, x_, yx_, [0, 1, 2]),
                               1401.98,
                               places=2)


class TestMnl(unittest.TestCase):
    def test_volume(self):
        self.assertEqual(len(
            gimnl.mnlmod(DAT, x, y, order_Bush).coefs),
            40)
        self.assertAlmostEqual(gimnl.mnlmod(
            DAT, x, y, order_Kerry).coefs.iloc[0, 0], -2.189,
                               places=2)
        self.assertAlmostEqual(gimnl.mnlmod(
            DAT, x, y, order_Kerry).coefs.iloc[5, 0], 0.344,
                               places=2)
        self.assertAlmostEqual(gimnl.mnlmod(
            DAT, x, y, order_Bush).coefs.iloc[0, 0], -3.018,
                               places=2)
        self.assertAlmostEqual(gimnl.mnlmod(
            DAT, x, y, order_Bush).coefs.iloc[5, 0], -0.386,
                               places=2)


class TestGimnl(unittest.TestCase):
    def test_volume(self):
        self.assertEqual(len(
            gimnl.gimnlmod(DAT, x, y, z, order_Kerry, "baseline").coefs),
            45)
        self.assertAlmostEqual(gimnl.gimnlmod(
            DAT, x, y, z, order_Bush, "baseline").coefs.iloc[0, 0], -4.938,
                               places=2)
        self.assertAlmostEqual(gimnl.gimnlmod(
            DAT, x, y, z, order_Bush, "baseline").coefs.iloc[5, 0], -1.632,
                               places=2)
        self.assertAlmostEqual(gimnl.gimnlmod(
            DAT, x, y, z, order_Bush, "baseline").coefs.iloc[10, 0], -0.390,
                               places=2)
        self.assertAlmostEqual(gimnl.gimnlmod(
            DAT, x, y, z, order_Kerry, "baseline").coefs.iloc[0, 0], -4.938,
                               places=2)
        self.assertAlmostEqual(gimnl.gimnlmod(
            DAT, x, y, z, order_Kerry, "baseline").coefs.iloc[5, 0], -0.724,
                               places=2)
        self.assertAlmostEqual(gimnl.gimnlmod(
            DAT, x, y, z, order_Kerry, "baseline").coefs.iloc[10, 0], 0.344,
                               places=2)


class TestVuongGimnl(unittest.TestCase):
    def test_volume(self):
        self.assertAlmostEqual(gimnl.vuong_gimnl(gimnl.mnlmod(
            DAT, x, y, order_Kerry), gimnl.gimnlmod(
            DAT, x, y, z, order_Kerry, "baseline")), -1.9174562877, places=5)

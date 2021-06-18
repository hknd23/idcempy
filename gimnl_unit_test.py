import pandas as pd
from idcempy import gimnl
import os
import unittest

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
            DAT, x, y, z, order_Kerry, "baseline")), -1.9174562877)

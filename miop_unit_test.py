import pandas as pd
from idcempy import zmiopc
import os
import unittest
import numpy as np

"""
Replication of results from 
Bagozzi, Benjamin E. and Bumba Mukherjee.  2012. 
"A Mixture Model for Middle-category Inflation in Ordered Survey Responses."  
Political Analysis. 20(3): 369-386
"""
DAT = pd.read_stata(
    os.getcwd() + "/data/EUKnowledge.dta", convert_categoricals=False)

Y = ["EU_support_ET"]

X = ["polit_trust", "Xenophobia", "discuss_politics", "Professional",
     "Executive", "Manual", "Farmer", "Unemployed", "rural", "female", "age",
     "student", "income", "Educ_high", "Educ_high_mid", "Educ_low_mid"]

Z = ["discuss_politics", "rural", "female", "age",
     "student", "EUbid_Know", "EU_Know_obj", "TV", "Educ_high",
     "Educ_high_mid", "Educ_low_mid"]


# miopc_EU = zmiopc.iopcmod('miopc', DAT, X, Y, Z)
# miop_EU = zmiopc.iopmod('miop', DAT, X, Y, Z)
# op_EU = zmiopc.opmod(DAT, X, Y)


class TestOp(unittest.TestCase):
    def test_opmodel(self):
        self.assertAlmostEqual(zmiopc.opmod(DAT, X, Y).coefs.iloc[2, 0],
                               0.76, places=0)


class TestMiop(unittest.TestCase):
    def test_miopmodel(self):
        self.assertAlmostEqual(zmiopc.iopmod('miop',
                                             DAT, X, Y, Z).coefs.iloc[2, 0],
                               0.43, places=0)
        self.assertAlmostEqual(zmiopc.iopmod('miop',
                                             DAT, X, Y, Z).coefs.iloc[14, 0],
                               0.90, places=0)


class TestMiopc(unittest.TestCase):
    def test_miopcmodel(self):
        self.assertAlmostEqual(zmiopc.iopcmod('miopc',
                                              DAT, X, Y, Z).coefs.iloc[14, 0],
                               0.84, places=0)


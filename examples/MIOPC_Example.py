import pandas as pd
from idcempy import zmiopc
import os
import time
import numpy as np

DAT = pd.read_stata(
    os.getcwd() + "/data/EUKnowledge.dta", convert_categoricals=False)

Y = ["EU_support_ET"]

X = ["polit_trust", "Xenophobia", "discuss_politics", "Professional",
     "Executive", "Manual", "Farmer", "Unemployed", "rural", "female", "age",
     "student", "income", "Educ_high", "Educ_high_mid", "Educ_low_mid"]

Z = ["discuss_politics", "rural", "female", "age",
     "student", "EUbid_Know", "EU_Know_obj", "TV", "Educ_high",
     "Educ_high_mid", "Educ_low_mid"]

start_time = time.time()
miop_model_paper = zmiopc.iopmod('miop', DAT, X, Y, Z)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
miopc_model_paper = zmiopc.iopcmod('miopc', DAT, X, Y, Z)
print("--- %s seconds ---" % (time.time() - start_time))

Y2 = ["EU_support_ET"]
X2 = ['Xenophobia', 'discuss_politics']
Z2 = ['discuss_politics', 'EU_Know_obj']

start_time = time.time()
miop_model_short = zmiopc.iopmod('miop', DAT, X2, Y2, Z2)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
miopc_model_short = zmiopc.iopcmod('miopc', DAT, X2, Y2, Z2)
print("--- %s seconds ---" % (time.time() - start_time))

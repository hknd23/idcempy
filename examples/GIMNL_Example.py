import pandas as pd
from idcempy import gimnl
import os
import time
import numpy as np

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

import pandas as pd
from zmiopc import bimnl

DAT = pd.read_stata(
    "C:/Users/Nguyen/Documents/Replication II/Replication "
    "II/replicationdata.dta", convert_categoricals=False)

x = ['educ', 'female', 'black', 'hispanic', 'party7', 'w3gbvalu2',
     'presbatt',
     'south', 'gayban2', 'agegroup2', 'bornagn_w', 'brnag_wXgmb2', 'catholic',
     'cathXgmb2', 'other_rel', 'secular', 'secXgmb2', 'ideo', 'w3mobidx']
z = ['educ', 'agegroup2', 'w3mobidx', 'secular']
DAT['votenum'] = pd.factorize(DAT['vote_turn'])[0]
y = ['votenum']

baseline = np.array([0, 1, 2])
baselines = np.array([0, 2, 1])
model = bimnl.bimnlmod(DAT, x, y, z, baseline,
                       method='BFGS')
models = bimnl.bimnlmod(DAT, x, y, z, baselines,
                        method='BFGS')

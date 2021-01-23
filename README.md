# The `ZiopcPy` Package

*Nguyen K. Huynh, Sergio Bejar, Nicolas Schmidt, Vineeta Yadav, Bumba Mukherjee*

<!-- badges: start -->

[![PyPI version fury.io](https://badge.fury.io/py/ziopcpy.svg)](https://pypi.org/project/ziopcpy/0.1.2/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ziopcpy.svg)](https://pypi.org/project/ziopcpy/0.1.2/)
[![Downloads](https://pepy.tech/badge/ziopcpy)](https://pepy.tech/project/ziopcpy)
[![Project Status: Active â€“ The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
<!-- badges: end -->

### Description
The `ZiopcPy` Python package fits the following models for (zero-inflated) ordered dependent variables in which the lowest outcome category has "excessive" zero observations: the zero-inflated ordered probit without correlated errors (ZiOP), the zero-inflated ordered probit with correlated errors (ZiOPC), and (as a baseline comparison) the standard ordered probit (OP) model.  The ZiOP and ZiOPC models jointly estimates a probit split-stage equation and an OP outcome-stage equation, which allows researchers to statistically account for excessive zeros in zero-inflated ordered dependent variables that relate to two data generating processes (d.g.p's).`ZiopcPy` also contains functions that allow users to compute the probability with which observations are in the always-zero group within the zero-inflated ordinal outcome variable, assess the goodness of fit of the models, obtain the marginal effects of the specified covariates, and to implement tests to perform model comparison. 

  - Documentation: https://ziopcpy.readthedocs.io/



### Functions

| Function         | Description                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `opmod`; `iopmod`; `iopcmod` | fit the standard OP model, the zero-inflated OP model without correlated errors (ZiOP), and the zero-inflated OP model with correlated errors (ZiOPC) respectively. |
|`opresults`; `iopresults`; `iopcresults`| Stores and presents the covariate estimates, the Variance-Covariance (VCV) matrix, the Log-Likelihood and the AIC of `opmod`, `iopmod`, and `iopcmod` respectively. |
| `iopfit`; `ziopcfit`| Computes the fitted probabilities from the ZiOP and ZiOPC models respectively.|
| `vuong_opziop`;  `vuong_opziopc` | Calculates the Vuong test statistic to compare the performance of the OP versus the ZiOP and ZiOPC models respectively.|

### Compatibility
Package compatible with [Python] 3.7+

### Dependencies
- scipy
- numpy
- pandas

### Installation

From [PyPi](https://pypi.org/project/ziopcpy/0.1.2/):

```sh
$ pip install ziopcpy
```

### Using the Package

We illustrate the functionality of ZiopcPy using data from Besley and Persson (2009) that is included and described in the package. Specifically, we estimate the effects of economic and political covariates on their ordered dependent variable, political violence, which is labeled as “rep_civwar_DV”.

Import `ZiopcPy` and other required packages:
```
from ziopcpy import ziopcpy
import pandas as pd
import urllib
```

```
url='https://github.com/hknd23/ziopcpy/raw/master/data/bp_exact_for_analysis.dta'
DAT=pd.read_stata(url)

# Specifying array of variable names (strings) X,Y,Z:
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
```
#### Running the ZiOP, ZiOPC, and OP model

Users should define an array of starting parameters before estimating `ziop`, `ziopc`, or `op` models. 
```
# Starting parameters for optimization:
pstartziop=np.array( [-1.31, .32, 2.5, -.21,.2, -0.2, -0.4, 0.2,.9,-.4])
pstartziopc = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4, .1])
pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

```

### Estimation of `ziop`, `ziopc` or `op` models
```
# Model estimation:
ziop_JCR = ziopcpy.iopmod(pstartziop, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)
ziopc_JCR = ziopcpy.iopcmod(pstartziopc, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)
JCR_OP = ziopcpy.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)
```
The estimation results from the table above are stored in the three classes `ZiopModel`, `ZiopcModel`, and `OpModel` with the following attributes:

  * *coefs*: Model coefficients and standard errors
  * *llik*: Log-likelihood
  * *AIC*: Akaike information criterion
  * *vcov*: Variance-covariance matrix
  
  The following table summarizes the results obtained from the three models (standard errors in parentheses)

|                           | OP       |         | ZiOP     |         | ZiOPC    |         |
| ------------------------- | -------- | ------- | -------- | ------- | -------- | ------- |
| OP Outcome-Stage   |          |         |          |         |          |         |
| logGDPpc                | \-0.212  | (0.035) | 0.041    | (0.049) | 0.332    | (0.053) |
| parliament              | \-0.538  | (0.100) | \-0.095  | (0.134) | 0.313    | (0.293) |
| disaster                | 0.220    | (0.026) | 0.265    | (0.034) | 0.197    | (0.033) |
| major\_oil              | 0.907    | (0.359) | 1.707    | (0.299) | 1.183    | (0.373) |
| major\_primary          | \-0.427  | (0.245) | \-0.422  | (0.263) | \-0.237  | (0.209) |
| cut1                      | \-1.073  | (0.269) | 0.772    | (0.353) | 2.763    | (0.370) |
| cut2                      | \-0.171  | (0.046) | \-0.098  | (0.047) | \-0.214  | (0.049) |
| Probit Split-Stage |          |         |          |         |          |         |
| int                     |          |         | 18.782   | (0.289) | 11.598   | (0.408) |
| logGDPpc                |          |         | \-2.082  | (0.026) | \-1.280  | (0.049) |
| parliament              |          |         | \-0.293  | (0.251) | \-0.370  | (0.297) |
| rho                       |          |         |          |         | \-0.889  | (0.040) |
| Log likelihood            | 1432.241 |         | 1385.909 |         | 1374.172 |         |
| AIC                       | 2878.483 |         | 2791.818 |         | 2770.344 |         |

The coefficients in the probit split-stage equation in the ZiOP(C) models reveal that ln GDP per capita has a negative and significant effect—while the parliament dummy has a negative but insignificant effect—on the likelihood of observations in the sample not being in the ‘‘always zero’’ group. The estimate of the parliament dummy is positive and insignificant in the outcome stage of the ZiOPC model, but negative and insignificant in the other two models. Other outcome stage coefficient estimates are largely similar. The estimate of the rho parameter in the ZiOPC model’s outcome stage is significant, which suggests that allowing for correlated disturbances between the two stages of the ZiOP is justified.  The Akaike information criterion (AIC) values reported for all the models in above table strongly favors the ZiOP and ZiOPC models over the OP model.

Users can obtain the values of the attributes by using the `print(Model_name.Attribute)` function. For example, to see the Variance-covariance matrix:

```
print(ziop_JCR.vcov)

[[ 1.24353127e-01  1.25663548e-03 -5.75548917e-02  1.70236103e-03
  5.05273309e-02  1.70531099e-02 -2.86418193e-02  2.58717572e-03
  -8.30490698e-03 -2.11871734e-03]
  ...
[-2.11871734e-03  5.64634344e-04 -9.57288274e-03  3.62751905e-04
  8.65751652e-03 -3.86427924e-04  1.58932049e-03  2.96437285e-04
  -5.25452969e-02  6.93057415e-02]]

print(ziopc_JCR.vcov)

[[ 1.36766528e-01 -1.50391291e-03 -2.25732999e-02 -1.42852474e-03
  4.18278908e-03  1.95389976e-02  3.02647268e-03 -1.09348495e-03
  3.22896421e-02 -9.24547286e-03 -3.83238156e-03]
  ...
[-3.83238156e-03  8.85000862e-04  3.45224424e-03 -4.08558670e-04
  -8.30687503e-04 -5.47455159e-04 -1.33691918e-03  3.12422823e-04
  -3.71512027e-03 -7.29939034e-04  1.60875279e-03]]

print(JCR_OP.vcov)

[[ 7.22800339e-02 -7.80059925e-04  9.35795290e-03 -1.10683026e-02
  -6.57753182e-05 -4.83722782e-03  3.86783131e-03]
  ...
[ 3.86783131e-03 -2.83366327e-04  3.16586107e-04  1.71164606e-03
  2.83414563e-04 -5.98088317e-02  6.01466912e-02]]
```

`ZiopcPy` also provides code to calculate and illustrate--from the estimated ZiOP(C) models’--the marginal effect (with 95% confidence intervals) of the (i) probit split-stage covariates on the probability of observations being in the always-zero group and (ii) OP outcome stage covariates on the probability of each outcome category. For example, from the Besley and Persson (2009) data, users can employ the relevant code to calculate and illustrate the marginal effect of the probit split-stage covariate--the parliamentary dummy--on the probability of observations entering the always-zero (versus not always-zero) group as:

![Marginal Effect of Split-stage Parliamentary Dummy (ZiOPC)](https://github.com/hknd23/ziopcpy/raw/master/graphics/ZiOPC_Parliament.png)

### Vuong Test

`ZiopcPy` also allows users to employ a variant of the Vuong test developed by Harris and Zhao (2007) to compare the performance of the OP model to both the ZiOP and ZiOPC models. The Vuong test results obtained from the OP, ZiOP, and ZiOPC models estimated on Besley and Persson’s (2009) data (reported below)  shows that the performance of the ZiOP and especially the ZiOPC model is superior to the OP model.    
```
ziopcpy.vuong_opziop(JCR_OP, ziop_JCR)

  -4.909399264831751

ziopcpy.vuong_opziopc(JCR_OP, ziopc_JCR)

  -5.424415009176218
```

For more information on the models, see Documentation.
### References
Bagozzi, Benjamin E., Daniel W. Hill Jr., Will H. Moore, and Bumba Mukherjee. 2015. "Modeling Two Types of Peace: The Zero-inflated Ordered Probit (ZiOP) Model in Conflict Research." *Journal of Conflict Resolution*, 59(4): 728-752.

Besley, Timothy and Persson, Torsten. 2009. "The origins of state capacity: property rights, taxation and politics." *American Economic Review*, 99(4): 1218-1244. 

Harris, Mark N. and Zhao, Xueyan. 2007. "A zero-inflated ordered probit model, with an application to modelling tobacco consumption." *Journal of Econometrics*, 141(2):1073-1099.


License
----
MIT License



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
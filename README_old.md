# ZiopcPy

<!-- badges: start -->

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Project Status: Active â€“ The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![](https://img.shields.io/badge/devel%20version-0.1.0-blue.svg)](https://github.com/hknd23/ziopc)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
<!-- badges: end -->

`ziopc` estimates split-population (or "zero-inflated") order probit models that minimize econometric challenges that researchers face in the presence of a "zero-inflated" outcome variable. The current version of `ziopc` allows users to estimate the model without correlated errors (`ziop`) and with correlated errors (`ziopc`) where the two error terms are correlated with one another.  

  - Documentation:
  - GitHub Repository: https://github.com/hknd23/ziopc

### To Do List

| Task |Done |
| ----------------- | -------------------------------------------------------------- |
|Edit Documentation|:heavy_check_mark:|
|Add results of Vuong test on Git|:heavy_check_mark:|
|Add results of post-estimation diagnostics on Git|:heavy_check_mark:|
|Add results of OP model|:heavy_check_mark:|
|Discuss Results of Vuong Test||
|Update Functions Section Paper||
|Edit paper||
|Create markdown file paper||
|Make repository public||
|Create documentations on readthedocs.io||
|Submit package to TestPyPI||
|Submit package to PyPI||



### Functions

| Function         | Description                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `iopmod` | Fits a SP ordered probit model without correlated errors.  This estimator contains two latent equations: a split-probit equation in the first stage, and an OP equation in the second (outcome) stage.|
| `iopcmod` | Fits a SP ordered probit model with correlated errors.  It contains two latent equations: a split-probit equation in the first stage, and an OP equation in the second (outcome) stage.|
| `opmod` | Fits an ordered probit model|
|`iopresults` | Stores the estimates of `iopmod` and the following goodness of fit tests: Log-Likelihood, AIC, and variance-covariance matrix. |
|`iopcresults` | Stores the estimates of `iopcmod` and the following goodness of fit tests: Log-Likelihood, AIC, and variance-covariance matrix.|
|`opresults`| Stores the estimates of `opmod` and the following goodness of fit tests: Log-Likelihood, AIC, and variance-covariance matrix.|
| `iopfit` | Generates individual probabilities for each outcome from the ZiOP model.|
| `iopcfit` | Generates individual probabilities for each outcome from the estimated ZiOPC model.|
| `vuong_opziop` | Vuong test to compare the performance of the Ordered Probit model versus the ZiOP model.|
| `vuong_opziopc` | Vuong test to compare the performance of the Ordered Probit model versus the ZiOPC model.|

### Compatibility
Package compatible with [Python] 3.5.2+

### Dependencies
- numpy
- pandas
- scipy

### Installation

Install package using pip:

```sh
$ pip install ziopcpy
```

Install package from GitHub repository:

```sh
$ git clone https://github.com/hknd23/ziopcpy.git
$ python python setup.py install
```

### Using Package
Import package:
```
from ziopcpy import ziopcpy
```

Example from Besley and Persson (2009):

```sh
import urllib
url='https://github.com/hknd23/ziopcpy/raw/master/data/bp_exact_for_analysis.dta'
DAT=pd.read_stata(url)

# Specifying array of variable names (strings) X,Y,Z:
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
```
#### Running the ZiOP model:

```
# Starting parameters for optimization:
pstartziop=np.array( [-1.31, .32, 2.5, -.21,.2, -0.2, -0.4, 0.2,.9,-.4])

# Model estimation:
ziop_JCR = ziopcpy.iopmod(pstartziop, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

```

### Results of the ZiOP model:

The ZiOP model estimates are stored in object of class 'ZiopModel'. ZiopModel.coefs displays the full estimates of the model.

```
print(ziop_JCR.coefs)

                      Coef        SE         2.5%      97.5%
  cut1              0.771855  0.352637     0.080686   1.463024
  cut2             -0.098204  0.046598    -0.189536  -0.006872
  Z int            18.781755  0.289231    18.214862  19.348647
  Z logGDPpc       -2.081926  0.025977    -2.132841  -2.031010
  Z parliament     -0.292586  0.251139    -0.784819   0.199647
  X logGDPpc        0.041251  0.048662    -0.054127   0.136629
  X parliament     -0.095081  0.133979    -0.357679   0.167517
  X disaster        0.264986  0.034355     0.197651   0.332321
  X major_oil       1.706935  0.299351     1.120208   2.293663
  X major_primary  -0.422205  0.263260    -0.938194   0.093785
```

Class 'ZiopModel' also stores:

 * ZiopModel.llik: log-likelihood ,
 * ZiopModel.AIC: Akaike information criterion ,
 * ZiopModel.vcov: variance-covariance matrix.

```
print(ziop_JCR.llik)

  2791.818107276211

print(ziop_JCR.AIC)

  1385.9090536381054

print(ziop_JCR.vcov)

[[ 1.24353127e-01  1.25663548e-03 -5.75548917e-02  1.70236103e-03
 5.05273309e-02  1.70531099e-02 -2.86418193e-02  2.58717572e-03
-8.30490698e-03 -2.11871734e-03]
[ 1.25663548e-03  2.17137151e-03 -1.34757273e-03 -7.02547396e-05
 1.13053836e-03  2.62162025e-04 -4.77649967e-04  1.44015924e-04
-1.67677862e-03  5.64634344e-04]
[-5.75548917e-02 -1.34757273e-03  8.36543691e-02 -5.46192479e-03
-6.94681932e-02 -7.86348532e-03  1.59032987e-02 -4.13098540e-03
-4.07218269e-03 -9.57288274e-03]
[ 1.70236103e-03 -7.02547396e-05 -5.46192479e-03  6.74821349e-04
 3.82236915e-03  1.96977904e-04 -1.30929783e-03  4.55240903e-04
 1.30803246e-03  3.62751905e-04]
[ 5.05273309e-02  1.13053836e-03 -6.94681932e-02  3.82236915e-03
 6.30709182e-02  7.08458425e-03 -1.21971176e-02  1.30910082e-03
 1.32403894e-03  8.65751652e-03]
[ 1.70531099e-02  2.62162025e-04 -7.86348532e-03  1.96977904e-04
 7.08458425e-03  2.36800214e-03 -3.99730801e-03  2.80982592e-04
-1.14559462e-03 -3.86427924e-04]
[-2.86418193e-02 -4.77649967e-04  1.59032987e-02 -1.30929783e-03
-1.21971176e-02 -3.99730801e-03  1.79502507e-02 -1.21790020e-03
-1.30486558e-03  1.58932049e-03]
[ 2.58717572e-03  1.44015924e-04 -4.13098540e-03  4.55240903e-04
 1.30910082e-03  2.80982592e-04 -1.21790020e-03  1.18023536e-03
 6.03715401e-04  2.96437285e-04]
[-8.30490698e-03 -1.67677862e-03 -4.07218269e-03  1.30803246e-03
 1.32403894e-03 -1.14559462e-03 -1.30486558e-03  6.03715401e-04
 8.96108237e-02 -5.25452969e-02]
[-2.11871734e-03  5.64634344e-04 -9.57288274e-03  3.62751905e-04
 8.65751652e-03 -3.86427924e-04  1.58932049e-03  2.96437285e-04
-5.25452969e-02  6.93057415e-02]]
```

 See Documentation for further details.  

#### Running the ZiOPC model:


```
# Starting parameters for optimization, note the extra parameter for rho:
pstartziopc = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4, .1])

# Model estimation:
ziopc_JCR = ziopcpy.iopcmod(pstartziopc, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)
```

Results of the ZiOPC model:

The ZiOPC model estimates are stored in object of class 'ZiopcModel'. ZiopcModel.coefs displays the full estimates of the model.


```
print(ziopc_JCR.coefs)

                      Coef          SE         2.5%      97.5%
  cut1              2.762593  0.369820     2.037746   3.487439
  cut2             -0.214227  0.048677    -0.309634  -0.118820
  Z int            11.597619  0.407915    10.798106  12.397132
  Z logGDPpc       -1.279668  0.049340    -1.376374  -1.182961
  Z parliament     -0.370217  0.296634    -0.951619   0.211186
  X logGDPpc        0.331656  0.053253     0.227281   0.436032
  X parliament      0.312728  0.292929    -0.261414   0.886869
  X disaster        0.197342  0.033247     0.132179   0.262506
  X major_oil       1.182631  0.373049     0.451455   1.913806
  X major_primary  -0.236625  0.209179    -0.646615   0.173365
  rho              -0.889492  0.040109    -0.968106  -0.810878

```
Similar to ZiOP, for the ZiOPC model class 'ZiopModel' also stores:

 * ZiopcModel.llik: log-likelihood ,
 * ZiopcModel.AIC: Akaike information criterion ,
 * ZiopcModel.vcov: variance-covariance matrix.

```
print(ziopc_JCR.llik)

  2770.3437983426634

print(ziopc_JCR.AIC)

  1374.1718991713317

print(ziopc_JCR.vcov)

[[ 1.36766528e-01 -1.50391291e-03 -2.25732999e-02 -1.42852474e-03
   4.18278908e-03  1.95389976e-02  3.02647268e-03 -1.09348495e-03
   3.22896421e-02 -9.24547286e-03 -3.83238156e-03]
 [-1.50391291e-03  2.36945679e-03  1.88639146e-03 -2.57219217e-04
  -1.04505315e-03 -1.21329416e-04 -4.32553907e-04  2.10605810e-04
  -1.04276766e-03 -1.76389572e-03  8.85000862e-04]
 [-2.25732999e-02  1.88639146e-03  1.66394352e-01 -1.90961806e-02
   8.75162735e-02 -3.40985040e-03 -9.01483743e-02  2.41857068e-03
  -1.13991388e-01  1.27784177e-02  3.45224424e-03]
 [-1.42852474e-03 -2.57219217e-04 -1.90961806e-02  2.43442994e-03
  -1.08611849e-02 -2.30111793e-04  1.04338186e-02 -2.39108240e-04
   1.31222174e-02 -1.37775886e-03 -4.08558670e-04]
 [ 4.18278908e-03 -1.04505315e-03  8.75162735e-02 -1.08611849e-02
   8.79916323e-02  4.52656852e-04 -7.94776031e-02  2.29393658e-03
  -3.81845254e-02  1.78214211e-02 -8.30687503e-04]
 [ 1.95389976e-02 -1.21329416e-04 -3.40985040e-03 -2.30111793e-04
   4.52656852e-04  2.83584035e-03  7.68676947e-04 -2.34876608e-04
   4.63026916e-03 -1.43687570e-03 -5.47455159e-04]
 [ 3.02647268e-03 -4.32553907e-04 -9.01483743e-02  1.04338186e-02
  -7.94776031e-02  7.68676947e-04  8.58076092e-02 -3.16698131e-03
   3.52385742e-02 -1.59600901e-02 -1.33691918e-03]
 [-1.09348495e-03  2.10605810e-04  2.41857068e-03 -2.39108240e-04
   2.29393658e-03 -2.34876608e-04 -3.16698131e-03  1.10535159e-03
  -1.63083331e-03  7.22361297e-04  3.12422823e-04]
 [ 3.22896421e-02 -1.04276766e-03 -1.13991388e-01  1.31222174e-02
  -3.81845254e-02  4.63026916e-03  3.52385742e-02 -1.63083331e-03
   1.39165200e-01 -4.19442153e-02 -3.71512027e-03]
 [-9.24547286e-03 -1.76389572e-03  1.27784177e-02 -1.37775886e-03
   1.78214211e-02 -1.43687570e-03 -1.59600901e-02  7.22361297e-04
  -4.19442153e-02  4.37557067e-02 -7.29939034e-04]
 [-3.83238156e-03  8.85000862e-04  3.45224424e-03 -4.08558670e-04
  -8.30687503e-04 -5.47455159e-04 -1.33691918e-03  3.12422823e-04
  -3.71512027e-03 -7.29939034e-04  1.60875279e-03]]
```
For detailed tutorial of ZiOP and ZiOPC, see Documentation page.

#### Running the OP model:

ziopcpy.opmod() fits a standard Ordered Probit model:

```
# Starting parameters for optimization:
pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

# Model estimation:
JCR_OP = ziopcpy.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)
```

TheOP model estimates are stored in object of class 'OpModel'. OpModel.coefs displays the full estimates of the model:

```
print(JCR_OP.coefs)

                      Coef        SE    tscore       2.5%     97.5%
  cut1            -1.072649  0.268849 -3.989777  -1.599594 -0.545704
  cut2            -0.171055  0.045801 -3.734712  -0.260826 -0.081284
  X logGDPpc      -0.212266  0.035124 -6.043404  -0.281108 -0.143424
  X parliament    -0.538013  0.099811 -5.390330  -0.733642 -0.342384
  X disaster       0.220324  0.026143  8.427678   0.169084  0.271564
  X major_oil      0.907116  0.358585  2.529714   0.204290  1.609942
  X major_primary -0.426577  0.245248 -1.739370  -0.907264  0.054109
```

Goodness-of-fit tests can be extracted from class `OpModel` with the following:

  *OpModel.llik: log-likelihood ,
  *OpModel.AIC: Akaike information criterion ,
  *OpModel.vcov: variance-covariance matrix.

```
print(JCR_OP.llik)

  2878.4827153434617

print(JCR_OP.AIC)

  1432.2413576717308

print(JCR_OP.vcov)

[[ 7.22800339e-02 -7.80059925e-04  9.35795290e-03 -1.10683026e-02
  -6.57753182e-05 -4.83722782e-03  3.86783131e-03]
 [-7.80059925e-04  2.09776589e-03 -5.03000582e-05 -9.68947891e-05
   9.77656183e-05  4.05978785e-04 -2.83366327e-04]
 [ 9.35795290e-03 -5.03000582e-05  1.23366596e-03 -1.58148181e-03
  -5.27931917e-05 -6.16426045e-04  3.16586107e-04]
 [-1.10683026e-02 -9.68947891e-05 -1.58148181e-03  9.96218655e-03
  -2.36019628e-04  4.38730116e-04  1.71164606e-03]
 [-6.57753182e-05  9.77656183e-05 -5.27931917e-05 -2.36019628e-04
   6.83452972e-04  7.70453121e-05  2.83414563e-04]
 [-4.83722782e-03  4.05978785e-04 -6.16426045e-04  4.38730116e-04
   7.70453121e-05  1.28582894e-01 -5.98088317e-02]
 [ 3.86783131e-03 -2.83366327e-04  3.16586107e-04  1.71164606e-03
   2.83414563e-04 -5.98088317e-02  6.01466912e-02]]
```

### Vuong Test

To compare the performance of the ZiOP/ZiOPC model versus the OP model, the Vuong test can be implemented:

```
ziopcpy.vuong_opziop(JCR_OP, ziop_JCR)

  -4.909399264831751

ziopcpy.vuong_opziopc(JCR_OP, ziopc_JCR)

  -5.424415009176218
```


### Reference
Bagozzi, Benjamin E., Daniel W. Hill Jr., Will H. Moore, and Bumba Mukherjee. 2015. "Modeling Two Types of Peace: The Zero-inflated Ordered Probit (ZiOP) Model in Conflict Research." *Journal of Conflict Resolution*. 59(4): 728-752.



License
----
MIT License



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

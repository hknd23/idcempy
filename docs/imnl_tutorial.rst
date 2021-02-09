The IDCeMPy package contains a number of functions that allow users to estimate "inflated" ordered probit (OP) models when the outcome variable has an excess number of observations in either the "zero" or the "middle" categories. The models can be estimated without (ZiOP and MiOP) and with (ZiOPC and MiOPC) correlated errors. In addition, IDCeMPy allows users to easily obtain fitted values, calculate goodness-of-fit and diagnostic tests, and marginal effects. This tutorial introduces all the aforementioned features of the package.

.. testcode::
from zmiopc import bimnl
url= 'https://github.com/hknd23/zmiopc/raw/main/data/replicationdata.dta'
data= pd.read_stata(url)


.. testcode:: 
x = ['educ', 'party7', 'agegroup2']
z = ['educ', 'agegroup2']
y = ['vote_turn']

.. testcode:: 
order = [0, 1, 2]
inflatecat = "baseline"

.. testcode:: 
imnl_2004vote = bimnl.imnlmod(data, x, y, z, order, inflatecat)

.. testoutput:: 
                       Coef    SE  tscore     p    2.5%  97.5%
Inflation: int       -4.935 2.777  -1.777 0.076 -10.379  0.508
Inflation: educ       1.886 0.293   6.441 0.000   1.312  2.460
Inflation: agegroup2  1.295 0.768   1.685 0.092  -0.211  2.800
1: int               -4.180 1.636  -2.556 0.011  -7.387 -0.974
1: educ               0.334 0.185   1.803 0.071  -0.029  0.697
1: party7             0.454 0.057   7.994 0.000   0.343  0.566
1: agegroup2          0.954 0.248   3.842 0.000   0.467  1.441
2: int                0.900 1.564   0.576 0.565  -2.166  3.966
2: educ               0.157 0.203   0.772 0.440  -0.241  0.554
2: party7            -0.577 0.058  -9.928 0.000  -0.691 -0.463
2: agegroup2          0.916 0.235   3.905 0.000   0.456  1.376
```

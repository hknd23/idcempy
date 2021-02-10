***********
imnl Module
***********

The IDCeMPy package also includes a function that estimates "inflated" Multinomial Logit models (iMNL).  iMNL models minimize issues present when unordered polytomous outcome variables have an excessive share and heterogeneous pool of observations in the lower category.  The application below uses data from Arcenaux and Kolodny (`2018 <https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-5907.2009.00399.x>`__) who analyze the effect of economic, social and political variables on *vote choice*.  In this case, the baseline category is frequently inflated as it includes non-voters who abstain from voting in an election owing to temporary factors and “routine” non-voters who are consistently disengaged from the political process.  Faling to account for such inflation could lead to inaccurate inferences.

To estimate the iMNL model, we first import the library and the dataset introduced above.

.. testcode::

   from idcempy import imnl
   url= 'https://github.com/hknd23/zmiopc/raw/main/data/replicationdata.dta'
   data= pd.read_stata(url)

We the define the list of covariates in the split-stage (z), the second-stage (x) and the outcome variable (y).

.. testcode::

   x = ['educ', 'party7', 'agegroup2']
   z = ['educ', 'agegroup2']
   y = ['vote_turn']

Users must then define the order of the variables and type of "inflation".  In the example below, we estimate a model with three categories and inflation in the baseline category.

.. testcode::

   order = [0, 1, 2]
   inflatecat = "baseline"

The following line of code estimates the iMNL model described above.

.. testcode::

   imnl_2004vote = bimnl.imnlmod(data, x, y, z, order, inflatecat)

class `imnlModel` stores the results from the regression object `imnl_2004vote` from :py:func:`imnlmod`.

The following line of code prints the coefficients of the covariates.

.. testcode::

   print(imnl_2004vote.coefs)

.. testoutput::

                          Coef   SE    tscore   p    2.5%   97.5%
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

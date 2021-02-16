***********
gimnl Module
***********

The IDCeMPy package also includes a function that estimates General "inflated" Multinomial Logit models (GiMNL).  GiMNL models minimize issues present when unordered polytomous outcome variables have an excessive share and heterogeneous pool of observations in the lower category.  The application below uses data from Campbell and Monson (`2008 <https://academic.oup.com/poq/article-abstract/72/3/399/1836972>`__) who use 'vote choice' as their outcome variable.  The 0,1,2 unordered-polytomous Presidential 'vote choice' doutcome variable in their data includes the following options: abstained (their MNL baseline category), Bush, or Kerry. In this case, the baseline category is inflated as it includes non-voters who abstain from voting in an election owing to temporary factors and “routine” non-voters who are consistently disengaged from the political process.  Faling to account for such inflation could lead to inaccurate inferences.

The covariates used to estimate the GiMNL model are:

- educ: Highest level of education completed.
- agegroup2: Indicator of age cohort.
- party7: Party identification.

To estimate the GiMNL model, we first import the library and the dataset introduced above.

.. testcode::

   from idcempy import gimnl
   url= 'https://github.com/hknd23/zmiopc/raw/main/data/replicationdata.dta'
   data= pd.read_stata(url)

We the define the list of covariates in the split-stage (z), the second-stage (x) and the outcome variable (y).

.. testcode::

   x = ['educ', 'party7', 'agegroup2']
   z = ['educ', 'agegroup2']
   y = ['vote_turn']

Users can employ the argument `inflatecat` to specify any unordered category as the inflated category (dictated by the distribution) in their unordered-polytomous outcome measure. If a higher category (say 1) is inflated in a 0,1,2 unordered outcome measure, then users can specify inflatecat as follows
.. testcode::

   order = [0, 1, 2]
   inflatecat = "baseline"


Further, employing the argument `reference`, users can select which category of the unordered outcome variable is the baseline ("reference") category by placing it first. Since the baseline ("0") category in the Presidential vote choice outcome measure is inflated, the following code fits the BIMNL Model.

.. testcode::

   gimnl_2004vote = gimnl.gimnlmod(data, x, y, z, order, inflatecat)


The following line of code prints the coefficients of the covariates.

.. testcode::

   print(gimnl_2004vote.coefs)

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

The results from the BIMNL model for this application are stored in a class (gimnlModel) with the following attributes:

- coefs: Model coefficients and standard errors
- llik: Log-likelihood
- AIC: Akaike information criterion
- vcov: Variance-covariance matrix

The AIC, for exmaple is given by,

.. testcode::
    print(gimnl_2004vote.AIC)

.. testoutput::
    1656.8324085039708

Using the function :py:func:`mnlmod`, users can fit a standard Multinomial Logit Model (MNL) by specifying the list of **X**, **Y**, and baseline (using `reference`).

.. testcode::

   mnl_2004vote = gimnl.mnlmod(data, x, y, z, order)
   print(mnl_2004vote.coefs)

.. testoutput::

  Coef    SE  tscore     p   2.5%  97.5%
  1: int       -4.914 0.164 -29.980 0.000 -5.235 -4.593
  1: educ       0.455 0.043  10.542 0.000  0.371  0.540
  1: party7     0.462 0.083   5.571 0.000  0.300  0.625
  1: agegroup2  0.951 0.029  32.769 0.000  0.894  1.008
  2: int        0.172 0.082   2.092 0.036  0.011  0.334
  2: educ       0.282 0.031   9.011 0.000  0.221  0.343
  2: party7    -0.567 0.085  -6.641 0.000 -0.734 -0.399
  2: agegroup2  0.899 0.138   6.514 0.000  0.629  1.170

Similar to the BiMNL model, the AIC for the MNL model can also be given by:

The AIC, for exmaple is given by,

.. testcode::
    print(mnl_2004vote.AIC)

.. testoutput::
    1657.192925769978

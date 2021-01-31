Tutorial for Estimating ZiOPC Model
===================================

Our package also includes functions to fit split-population (SP) " or zero-inflated" ordered probit models (ziopc) under the assumption that the two errors are correlated with each other (i.e. correlated errors). (see :doc:`ziop_tutorial` for detail on data and variable descriptions). The model also includes the estimate 'rho'.

Data from Besley and Persson (2009).

.. testcode::

  import numpy as np
  import pandas as pd
  import urllib
  #Import pandas and urllib to read data from url
  from ziopcpy import ziopcpy
  #Import data
  url='https://github.com/hknd23/ziopcpy/raw/master/data/bp_exact_for_analysis.dta'
  data=pd.read_stata(url)

  # Specify list of variable names (strings) X,Y,Z:
  X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
  Z = ['logGDPpc', 'parliament']
  Y = ['rep_civwar_DV']

  # Starting parameters for optimization, note the extra parameter for rho:
  pstart = np.array([-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4, .1])

  # Model estimation:
  ziopc_JCR = ziopcpy.ziopcmod(pstart, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

  # See estimates:
  print(ziopc_JCR.coefs)

:class:`ziopcpy.ZiopcModel` stores results from model estimation and other information.

Results from the model:

The following message will appear when the model finishes converging:

.. testoutput::

  Warning: Desired error not necessarily achieved due to precision loss.
        Current function value: 1374.171899
        Iterations: 44
        Function evaluations: 963
        Gradient evaluations: 74

Use print(ziopc_JCR.coefs) to see model results

.. testoutput::

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

:class:`ziopcpy.ZiopcModel` also stores information such as log-likelihood, AIC, and Variance-Covariance matrix

.. testcode::

  print(ziopc_JCR.llik)
  print(ziopc_JCR.AIC)
  print(ziopc_JCR.vcov)

To extract predicted probabilities from the model:
:func:`ziopcpy.ziopcfit` returns :class:`ziopcpy.FittedVals` containing fitted probablities.

.. testcode::

  fitttedziopc = ziopcpy.ziopcfit(ziopc_JCR)
  print(fitttedziopc.responsefull)

.. testoutput::

  array([[9.68868303e-01, 3.01063427e-02, 1.02535403e-03],
      [9.07563628e-01, 7.88301952e-02, 1.36061769e-02],
      [9.76972004e-01, 2.23954809e-02, 6.32514846e-04],
      ...,
      [9.66496738e-01, 3.19780772e-02, 1.52518446e-03],
      [9.82515374e-01, 1.70648356e-02, 4.19790597e-04],
      [9.83907141e-01, 1.57240833e-02, 3.68775369e-04]])

The Vuong Test with a v statistic can be performed to compare the performance of the ZiOPC model versus the standard Ordered Probit (OP) model using :func:`ziopcpy.vuong_opziopc`.
The OP and ZiOPC must have the same number of observations, and the OP must have the same number of covariates as ZiOPC's OP stage. (see :doc:`op_tutorial` for details on fitting the OP model)

.. testcode::

  ziopcpy.vuong_opziopc(JCR_OP, ziopc_JCR)

.. testoutput::

   -5.424415009176218

A v statistic where v < -1.96 favors the ZiOPC model, -1.96 < v < 1.96 favors neither model, and v > 1.96 favors the OP model.

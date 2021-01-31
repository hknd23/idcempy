Tutorial for Estimating OP Model
================================

Users can also use ziopcpy to estimate standard Ordered Probit (OP) models (without accounting for the "zero-inflation").

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
  Y = ['rep_civwar_DV']

  # Starting parameters for optimization:
  pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

  # Model estimation:
  JCR_OP = ziopcpy.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)

  # See estimates:
  print(JCR_OP.coefs)

:class:`ziopcpy.OpModel` stores results from model estimation and other information.

Results from the model:

The following message will appear when the model finishes converging

.. testoutput::

  Warning: Desired error not necessarily achieved due to precision loss.
      Current function value: 1385.909054
      Iterations: 34
      Function evaluations: 529
      Gradient evaluations: 44

Use print(JCR_OP.coefs) to see model results:

.. testoutput::

                      Coef        SE    tscore       2.5%     97.5%
  cut1            -1.072649  0.268849 -3.989777  -1.599594 -0.545704
  cut2            -0.171055  0.045801 -3.734712  -0.260826 -0.081284
  X logGDPpc      -0.212266  0.035124 -6.043404  -0.281108 -0.143424
  X parliament    -0.538013  0.099811 -5.390330  -0.733642 -0.342384
  X disaster       0.220324  0.026143  8.427678   0.169084  0.271564
  X major_oil      0.907116  0.358585  2.529714   0.204290  1.609942
  X major_primary -0.426577  0.245248 -1.739370  -0.907264  0.054109

Log-likelihood, AIC, and Variance-Covariance matrix can be extracted with:

.. testcode::

  print(JCR_OP.llik)
  print(JCR_OP.AIC)
  print(JCR_OP.vcov)

***************
IDCeMPy Package
***************

The IDCeMPy package contains a number of functions that allow users to estimate "inflated" ordered probit (OP) models when the outcome variable has an excess number of observations in either the "zero" or the "middle" categories.  The models can be estimated without (ZiOP and MiOP) and with (ZiOPC and MiOPC) correlated errors.  In addition, IDCeMPy allows users to easily obtain fitted values, calculate goodness-of-fit and diagnostic tests, and marginal effects.  This tutorial introduces all the aforementioned features of the package.  

Data Description
================

Data for the applications below come from two different sources.

**Data Source for "Zero-inflated" Ordered Probit Models (ZiOP/ZiOPC)**

We use data from the National Youth Tobacco Survey (`2018 <https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/index.htm>`__) to illustrate the use of ordered probit models with a "zero-inflated" outcome variable. This survey allows us to measure self-reported *Tobacco Consumption* among youngsters. Specifically, respondents of the survey answered the following question [Q9]: About how many cigarretes have you smoked in your entire life, and their responses are coded as 0 for "never smoked cigarettes", 1 for "1 cigarette", 2 for "2 to 5 cigarettes", 3 for "6 to 15 cigarettes," and 4 for "more than 15 cigarettes."
    
82% of the observations are in the first category (i.e. "never smoked cigarettes) of outcome variable *tobacco consumption*. So, there is an excessive number of "0" -zero- observations thus making the aforementioned dataset appropriate for the estimation of ZiOP and ZiOPC models.  

**Data Source for "Middle-inflated" Ordered Probit Models (MiOP/MiOPC)**

Data for the "Middle-inflated" Ordered Probit Models (MiOP and MiOPC) comes from Elgun and Tillman(`2007 <https://journals.sagepub.com/doi/10.1177/1065912907305684>`_) who use ordered categorical responses to the following question in the Candidate Countries Eurobarometer 2002.2 survey to evaluate public attitudes toward European Union membership in 13 CEE candidate countries: “Generally speaking, do you think that (your country’s) membership of the European Union would be a good thing, a bad thing, or neither good nor bad?” Based on responses to this question, the discrete ordered-dependent variable—usually labeled as EU support—in Elgun and Tillman’s (`2007 <https://journals.sagepub.com/doi/10.1177/1065912907305684>`_) study and in similar related studies is coded as 1 for “a bad thing,” 2 for “neither good nor bad,” and 3 for “a good thing.” A close examination of the ordered *EU support* variable indicates that 39% of all respondents tothe survey question mentioned above opted for the middle category
response, which is indeed high. Therefore, this dataset is useful to illustrate some important features of our package.  

Estimation of Zero-inflated and Middle-inflated Ordered Probit Models Without Correlated Errors
=================================================================================================
The `iopmod` function estimates regression objects for "zero-inflated" and "middle-inflated" ordered probit models without correlated errors.  Below you will find instructions to estimate a ZiOP model.  The estimation of MiOP models is also illustrated.  


**1. Import the required libraries, set up the package and import the dataset:**

.. testcode::

  # Import the necessary libraries and package
  
  import numpy as np
  import pandas as pd
  import urllib
  from zmiopc import zmiopc
  
  # Import the "Youth Tobacco Consumption" dataset described above
  
  url='https://github.com/hknd23/zmiopc/blob/main/data/tobacco_cons.csv'
  data=pd.read_stata(url)

**2. Estimation of the ZiOP model.**

.. testcode::

  # Define a list of variable names (strings) X,Y,Z:
  X = ['age', 'grade', 'gender_dum']
  Z = ['gender_dum']
  Y = ['cig_count']

X is the list of variables in the Ordered Probit equation (second-stage).
Z is the list of variables in the split-probit equation (first-stage). 
Y is the outcome variable.


Users must set up an array of starting parameters (one for each covariate) before estimating the model.

:func:`zmiopc.iopmod` estimates the ZiOP model and returns :class:`zmiopc.IopModel`.

.. testcode::

  # Starting parameters for optimization:
  pstartziop = np.array([.01, .01, .01, .01, .01, .01, .01 , .01, .01])

  # Model estimation:
  ziop_tob= zmiopc.iopmod('ziop', pstartziop, data, X, Y, Z, method='bfgs', weights= 1,offsetx= 0, offsetz=0)

  # See estimates:
  print(ziop_tob.coefs)

Results from the model:

The following message will appear when the model has converged:

.. testoutput:: 
         Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 5060.160903
         Iterations: 79
         Function evaluations: 1000
         Gradient evaluations: 100
         
Object :class:`zmiopc.IopModel` stores model results and goodness-of-fit tests in its attributes 'coefs', 'AIC', 'llik', and 'vcov'.

Use the following line of code to see the estimates of coefficients:

.. testcode::
   print(ziop_tob.coefs)
  
.. testoutput:: 

                            Coef        SE      tscore        p           2.5%      97.5%
   cut1                   1.693797  0.054383  31.145912  0.000000e+00   1.587207   1.800387
   cut2                  -0.757830  0.032290 -23.469359  0.000000e+00  -0.821119  -0.694542
   cut3                  -1.804483  0.071237 -25.330846  0.000000e+00  -1.944107  -1.664860
   cut4                  -0.691907  0.052484 -13.183210  0.000000e+00  -0.794775  -0.589038
   Inflation: int         4.161455  3.864721   1.076780  2.815784e-01  -3.413398  11.736309
   Inflation: gender_dum -3.462848  3.857160  -0.897772  3.693074e-01 -11.022881   4.097185
   Ordered: age          -0.029139  0.013290  -2.192508  2.834282e-02  -0.055187  -0.003090
   Ordered: grade         0.177897  0.012133  14.661952  0.000000e+00   0.154116   0.201678
   Ordered: gender_dum    0.206509  0.034914   5.914823  3.322323e-09   0.138078   0.274940

In addition to coefficient estimates, the table also presents the standard errors, and confidence intervals.

The model object also stores three (3) different diagnostic tests: (1) Log-likelihood, (2) Akaike Information Criteria (AIC), and Variance-Covariance Matrix (VCM).  You can obtain them via the following commands:

.. testcode::

  print(ziop_tob.llik)
  print(ziop_tob.AIC)
  print(ziop_tob.vcov)

An example for the AIC:
.. testcode::
   print(ziop_tob.AIC)
   
.. testoutput:: 
   10138.321806674261

**2.1 To extract predicted probabilities from the model:**
:func:`zmiopc.iopfit` returns :class:`zmiopc.FittedVals` containing fitted probablities.

.. testcode::

  fitttedziop = ziopc.iopfit(ziop_tob)
  print(fitttedziopc.responsefull)

.. testoutput::

  array[[0.8822262  0.06879832 0.01455244 0.0242539  0.01016914]
 [0.84619828 0.08041296 0.01916279 0.03549797 0.01872801]
 [0.93105632 0.04349743 0.00831396 0.0127043  0.004428  ]
 ...
 [0.73347708 0.1291157  0.03295816 0.06500889 0.03944016]
 [0.87603805 0.06808193 0.01543795 0.02735256 0.01308951]
 [0.82681957 0.08778215 0.02153509 0.04095753 0.02290566]]
 

**3. Estimation of the MiOP model**
 
We begin by importing the Elgun and Tilam (`2007 <https://journals.sagepub.com/doi/10.1177/1065912907305684>`_) data on European Integration described above.  Recall that our outcome variable is "inflated" in the middle category.  

.. testcode::
 
    url = 'https://github.com/hknd23/zmiopc/blob/main/data/'
    data2 = pd_read.stata(url)
 
We then define the lists with the names of the variables used in the model

.. testcode::

  X = ['Xenophobia', 'discuss_politics']
  Z = ['discuss_politics', EU_Know_ob]
  Y = ['EU_support_ET']

X is the list of variables in the Ordered Probit equation (second-stage).
Z is the list of variables in the split-probit equation (first-stage). 
Y is the outcome variable. 

Users must then set up an array of starting parameters (one for each covariate) before estimating the model.

:func:`zmiopc.iopmod` estimates the MiOP model and returns :class:`zmiopc.IopModel`.

.. testcode::

  # Starting parameters for optimization:
  pstartziop = np.array([.01, .01, .01, .01, .01, .01, .01 , .01, .01])

  # Model estimation:
  miop_EU = zmiopc.iopmod('miop', pstartziop, data, X, Y, Z, method='bfgs', weights= 1,offsetx= 0, offsetz=0)

.. testoutput::
         Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 10857.695490
         Iterations: 37
         Function evaluations: 488
         Gradient evaluations: 61  # See estimates:
.. testcode::         
         print(miop_EU.coefs)

.. testoutput::
                                 Coef        SE       tscore         p         2.5%     97.5%
   cut1                        -1.159621  0.049373 -23.487133  0.000000e+00 -1.256392 -1.062851
   cut2                        -0.352743  0.093084  -3.789492  1.509555e-04 -0.535188 -0.170297
   Inflation: int              -0.236710  0.079449  -2.979386  2.888270e-03 -0.392431 -0.080989
   Inflation: discuss_politics  0.190595  0.035918   5.306454  1.117784e-07  0.120197  0.260993
   Inflation: EU_Know_obj       0.199574  0.020308   9.827158  0.000000e+00  0.159770  0.239379
   Ordered: Xenophobia         -0.663551  0.044657 -14.858898  0.000000e+00 -0.751079 -0.576024
   Ordered: discuss_politics    0.023784  0.029365   0.809964  4.179609e-01 -0.033770  0.081339

In addition to coefficient estimates, the table also presents the standard errors, and confidence intervals.

The model object also stores three (3) different diagnostic tests: (1) Log-likelihood, (2) Akaike Information Criteria (AIC), and Variance-Covariance Matrix (VCM).  You can obtain them via the following commands:

.. testcode::

  print(miop_EU.llik)
  print(miop_EU.AIC)
  print(miop_EU.vcov)

An example for the AIC:
.. testcode::
   print(miop_EU.AIC)
   
.. testoutput:: 
   21729.390980849777

Please see **Section 2.1** for instructions on how to calculate and print the fitted values. 
   
Estimation of Zero-inflated and Middle-inflated Ordered Probit Models "With" Correlated Errors
==========================

The package also includes the function `iopcmod` which fits "zero-inflated" ordered probit models (ZiOPC) and "middle-inflated" ordered probit models (MiOP) under the assumption that the two errors are correlated with each other (i.e. correlated errors). Both models include the estimate of'rho'. The models in this section use the same specification as the models estimated without correlated errors presented above.  

**1. Define an array with values of starting parameters**
.. testcode::
    pstart = np.array([.01, .01, .01, .01, .01, .01, .01 , .01, .01, .01])
    
**2. Estimate the ZiOPC model**
.. testcode::
    ziopc_tob = zmiopc.iopcmod('ziopc', pstart, data, X, Y, Z, method='bfgs', weights=1, offsetx=0, offsetz=0)

Similar to ZiOP, the results are stored in the attributes of :class:`zmiopc.IopCModel`.

.. testoutput::
         Current function value: 5060.051910
         Iterations: 119
         Function evaluations: 1562
         Gradient evaluations: 142

**2.1 Print the results**
.. testcode::
    print(ziopc_tob.coefs)

.. testoutput::
                            Coef        SE     tscore             p       2.5%      97.5%
   cut1                   1.696160  0.044726  37.923584  0.000000e+00   1.608497   1.783822
   cut2                  -0.758095  0.033462 -22.655678  0.000000e+00  -0.823679  -0.692510
   cut3                  -1.812077  0.060133 -30.134441  0.000000e+00  -1.929938  -1.694217
   cut4                  -0.705836  0.041432 -17.036110  0.000000e+00  -0.787043  -0.624630
   Inflation: int         9.538072  3.470689   2.748178  5.992748e-03   2.735521  16.340623
   Inflation: gender_dum -9.165963  3.420056  -2.680062  7.360844e-03 -15.869273  -2.462654
   Ordered: age          -0.028606  0.008883  -3.220369  1.280255e-03  -0.046016  -0.011196
   Ordered: grade         0.177541  0.010165  17.465452  0.000000e+00   0.157617   0.197465
   Ordered: gender_dum    0.602136  0.053084  11.343020  0.000000e+00   0.498091   0.706182
   rho                   -0.415770  0.074105  -5.610526  2.017123e-08  -0.561017  -0.270524
   
To print the estimates of the log-likelihood, AIC, and Variance-Covariance matrix, you should type:

.. testcode::
  print(ziopc_tob.llik)
  print(ziopc_tob.AIC)
  print(ziopc_tob.vcov)
  
The AIC of the ziopc_tob model, for example, is:

.. testoutput::
    10140.103819465658

**2.2. Obtain predicted probabilities from the ziopc_tob model:**
:func:`zmiopc.iopcfit` returns :class:`zmiopc.FittedVals` containing fitted probablities.

.. testcode::

  fitttedziopc = zmiopc.iopcfit(ziopc_tob)
  print(fitttedziopc.responsefull)

.. testoutput::

  array[[0.88223509 0.06878162 0.01445941 0.0241296  0.01039428]
 [0.84550989 0.08074461 0.01940226 0.03589458 0.01844865]
 [0.93110954 0.04346074 0.00825639 0.01264189 0.00453143]
 ...
 [0.73401588 0.12891071 0.03267436 0.06438928 0.04000977]
 [0.87523652 0.06888286 0.01564958 0.0275354  0.01269564]
 [0.82678185 0.0875059  0.02171135 0.04135142 0.02264948]]
 
 **3. Estimation of MiOPC**
This example uses the the Elgun and Tilam (`2007 <https://journals.sagepub.com/doi/10.1177/1065912907305684>`_) data on European Integration described above.  Recall that our outcome variable is "inflated" in the middle category.  

.. testcode::
 
    url = 'https://github.com/hknd23/zmiopc/blob/main/data/'
    data2 = pd_read.stata(url)
 
We then define the lists with the names of the variables used in the model

.. testcode::

  X = ['Xenophobia', 'discuss_politics']
  Z = ['discuss_politics', EU_Know_ob]
  Y = ['EU_support_ET']

X is the list of variables in the Ordered Probit equation (second-stage).
Z is the list of variables in the split-probit equation (first-stage). 
Y is the outcome variable. 

Users must then set up an array of starting parameters (one for each covariate) before estimating the model.

:func:`zmiopc.iopmod` estimates the MiOP model and returns :class:`zmiopc.IopModel`.

.. testcode::

  # Starting parameters for optimization:
  pstartziop = np.array([.01, .01, .01, .01, .01, .01, .01 , .01, .01, .01])

  # Model estimation:
  miopc_EU = zmiopc.iopcmod('miopc', pstartziop, data, X, Y, Z, method='bfgs', weights= 1,offsetx= 0, offsetz=0)

.. testcode::         
         print(miopc_EU.coefs)

.. testoutput::
                                 Coef  SE     tscore  p     2.5%  97.5%
   cut1                        -1.370 0.044 -30.948 0.000 -1.456 -1.283
   cut2                        -0.322 0.103  -3.123 0.002 -0.524 -0.120
   Inflation: int              -0.129 0.021  -6.188 0.000 -0.170 -0.088
   Inflation: discuss_politics  0.192 0.026   7.459 0.000  0.142  0.243
   Inflation: EU_Know_obj       0.194 0.027   7.154 0.000  0.141  0.248
   Ordered: Xenophobia         -0.591 0.045 -13.136 0.000 -0.679 -0.502
   Ordered: discuss_politics   -0.029 0.021  -1.398 0.162 -0.070  0.012
   rho                         -0.707 0.106  -6.694 0.000 -0.914 -0.500

In addition to coefficient estimates, the table also presents the standard errors, and confidence intervals.

The model object also stores three (3) different diagnostic tests: (1) Log-likelihood, (2) Akaike Information Criteria (AIC), and Variance-Covariance Matrix (VCM).  You can obtain them via the following commands:

.. testcode::

  print(miop_EU.llik)
  print(miop_EU.AIC)
  print(miop_EU.vcov)

Above you can read the instructions on how to calculate and print the fitted values.  

Estimating the OP Model
=======================

The package also includes a fucntion that estimates a standard Ordered Probit (OP) model.
The OP model does not account for the "zero inflation", so it does not have a split-probit stage. 

.. testcode::
     # Define a list of variable names (strings) X,Y,Z:
     X = ['age', 'grade', 'gender_dum']
     Y = ['cig_count']

X is the list of variables in the Ordered Probit equation.
Y is the outcome variable.

.. testcode::

  # Starting parameters for optimization:
  pstartop = np.array([.01, .01, .01, .01, .01, .01, .01])

  # Model estimation:
  op_tob = zmiopc.opmod(pstartop, data, X, Y, method='bfgs', weights=1, offsetx=0)
  
  # See estimates:
  print(ziop_tob.coefs)

Results from the model:

The following message will appear when the model has converged:

.. testoutput:: 
         Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 4411.710049
         Iterations: 10
         Function evaluations: 976
         Gradient evaluations: 121         

:class:`zmiopc.OpModel` stores results from model estimation and other information in its attributes.
The following line of code to see the estimates of coefficients:

.. testcode::
   print(op_tob.coefs)
  
.. testoutput::
                Coef        SE     tscore         p      2.5%     97.5%
   cut1        1.696175  0.047320  35.844532  0.000000  1.603427  1.788922
   cut2       -0.705037  0.031650 -22.276182  0.000000 -0.767071 -0.643004
   cut3       -2.304405  0.121410 -18.980329  0.000000 -2.542369 -2.066441
   cut4        2.197381  0.235338   9.337141  0.000000  1.736119  2.658643
   age        -0.070615  0.007581  -9.314701  0.000000 -0.085474 -0.055756
   grade       0.233741  0.010336  22.614440  0.000000  0.213483  0.254000
   gender_dum  0.020245  0.032263   0.627501  0.530331 -0.042991  0.083482

Log-likelihood, AIC, and Variance-Covariance matrix can be extracted with:

.. testcode::

  print(op_tob.llik)
  print(op_tob.AIC)
  print(op_tob.vcov)

The Vuong Test
==============

Harris and Zhao (`2007 <https://doi.org/10.1016/j.jeconom.2007.01.002>`__) suggest that a variant of the Vuong (`1989 <https://www.jstor.org/stable/1912557>`__) Test (with a v statistic) can be used to compare the performance of the ZiOP versus the standard Ordered Probit (OP) model using :func:`zmiopc.vuong_opiop`.
The Vuong test denotes m\ :sub:`i`\ as the natural logarithm of the ratio of the predicted probablity that i\ :sub:`j`\ of the simpler OP model (in the numerator) and the more general (ZiOP/ZiOPC) model (in the denominaor) and evaluates m\ :sub:`i`\
via a bidirectional test statistic of:

.. math::

   v = \frac{\sqrt{N}(\frac{1}{N}\sum_{i}^{N}m_{i})}{\sqrt{\frac{1}{N}\sum_{i}^{N}(m_{i}-\bar{m})^{2}}}

where v < -1.96 favors the more general (ZiOP/ZiOPC) model, -1.96 < v < 1.96 lends no support to either model, and v > 1.96 supports the simpler (OP) model.

The OP and ZiOP models must have the same number of observations, and the OP must have the same number of covariates as ZiOP's OP stage.

.. testcode::

  zmiopc.vuong_opiop(op_tob, ziop_tob)

.. testoutput::

   6.624742132792222
   
The Vuong test can also be implemented to compare the ZiOPC, MiOP and MiOPC models and the OP model.

Split Equtation Predicted Probablities
======================================

:func:`zmiopc.split_effects` simulates data from ZiOP/ZiOPC and MiOP/MiOPC model results and computes changes in predicted probabilities when the value of a variable changes.
This allows you to illustrate how the changes in the split-probit covariates affect the probablilities of being in one population versus another. The example below illustrates the marginal effects of the variable 'gender_dum' on the outcome variable in the ZiOPC model estimated in ths documentation.

.. testcode::

    ziopcgender = idcempy.split_effects(ziopc_tob, 1)
   
The returned dataframe contains predicted probabilities when 'parliament' equals 0, and when 'parliament' equals 1.
The box plots below illustrate the change in predicted probablities using the values from the 'ziopparl' dataframe.
 
.. testcode::
     ziopcgender.plot.box(grid='False')
 
 .. image:: ../graphics/ziopc_me.png

Outcome Equation Predicted Probabilities
========================================




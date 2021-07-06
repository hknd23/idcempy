***************
IDCeMPy Package
***************

Description
===========
`IDCeMPy` is a Python package that provides functions to fit and assess the performance of the following distinct
sets of “inflated” discrete choice models:

* Fit the Zero-Inflated Ordered Probit (ZiOP) model without and with correlated errors (ZiOPC model) to evaluate zero-inflated ordered choice outcomes that results from a dual data generating process (d.g.p.).

* Fit the Middle-Inflated Ordered Probit (MiOP) model without and with correlated errors (MiOPC) to account for the inflated middle-category in ordered choice measures that relates to a dual d.g.p.

* Fit Generalized Inflated Multinomial Logit (GiMNL) models that account for the preponderant and heterogeneous share of observations in the baseline or any lower category in unordered polytomous choice outcomes.

* Compute AIC and Log-likelihood statistics and the Vuong Test statistic to assess the performance of each inflated discrete choice model in the package.

`IDCeMPy` uses Newton numerical optimization methods to estimate the models listed above via Maximum Likelihood Estimation (MLE).

When Should You use `IDCeMPy`?
==============================

An excessive (“inflated”) share of observations—stemming from two distinct d.g.p’s—fall into a single choice category in many ordered and unordered polytomous outcome variables.
Standard Ordered Probit and Multinomial Logit models cannot account for such category inflation which leads to biased inferences. Examples for such d.g.p’s include:

* The inflated zero-category of "no smoking" in ordered measures of self-reported smoking behavior is generated from nonsmokers who never smoke cigarettes and those who smoked previously but temporarily stopped smoking because of high cigarette prices.

* The inflated "indifference" middle-category in ordered measures of immigration attitudes includes respondents truly indifferent to immigration and those that choose indifference for social desirability reasons.

* The inflated baseline or other lower outcome categories of unordered polytomous outcome measures of vote choice include nonvoters who temporarily abstain from voting and routine nonvoters who always abstain.

`IDCeMPy` includes the ZIOP(C) models for evaluating zero-inflated ordered choice outcomes that results from a dual d.g.p, the MIOP(C) models that address inflated middle-category ordered outcome measures arising from distinct d.g.p’s, and GIMNL models that account for inflated baseline or other categories for unordered polytomous outcomes.

Each inflated discrete choice model in this package addresses category inflation in one’s discrete outcome—unordered or unordered polytomous—of interest by jointly estimating a binary split-stage equation and an ordered or multinomial discrete choice outcome equation.

Installation
============
The package can be installed in two different ways:

1. From `PyPi <https://pypi.org/project/idcempy/>`__:

.. testcode::

  $  pip install idcempy

2. From its `GitHub Repository <https://github.com/hknd23/idcempy/>`__:

.. testcode::

  $  git clone https://github.com/hknd23/idcempy.git
  $  cd idcempy
  $  python setup.py install

Examples
========

The examples below demonstrate how to use the `IDCeMPy` package to estimate the inflated discrete choice models ZiOP(C), MiOP(C), and GiMNL.
The example code files, with rough calculation of model run time, are available in the `/examples <https://github.com/hknd23/idcempy/tree/main/examples>`__ directory.
For each model example below, the run time is available as reference point. The specification used to record the times is Intel Core i7-2600 (3.40GHz Quad core), 16GB RAM.
Please note that for models in the `zmiopc` module, the run-time for models with correlated errors estimated with :func:`zmiopc.iopcmod` is substantially higher
than their without correlated errors counterparts using :func:`zmiopc.iopmod`. Other factors affecting run-time are the number of observations and the number of covariates.

The examples use the pandas, urllib, and matplotlib packages for importing and visualizing data:

.. testcode::

  $  pip install pandas
  $  pip install matplotlib
  $  pip install urllib

Zero-inflated Ordered Probit (ZiOP) Model without Correlated Errors
-------------------------------------------------------------------
The :func:`zmiopc.iopmod` function estimates regression objects for "zero-inflated" and "middle-inflated" ordered probit models without correlated errors.
This section provides instruction to estimate the ZiOP model using the self-reported smoking behavior as empirical example.

We first import the required libraries, set up the package and import the dataset:

.. testcode::

  # Import the necessary libraries and package
  import pandas as pd
  import urllib
  import matplotlib.pyplot as plot
  from idcempy import zmiopc

  # Import the "Youth Tobacco Consumption" dataset as a pandas.DataFrame
  url='https://github.com/hknd23/zmiopc/blob/main/data/tobacco_cons.csv'
  data = pd.read_csv(url)

The data is now a `pandas` DataFrame, and we can proceed to estimate the ZiOP model as follows.

.. testcode::

  # First, define a list of variable names of X, Z, and Y.
  # X = Column names of covariates (from `DataFrame`) used in ordered probit stage.
  # Z = Column names of covariates (from `DataFrame`) used in split-population stage.
  # Y = Column name of ordinal outcome variable (from `DataFrame`).

  X = ['age', 'grade', 'gender_dum']
  Z = ['gender_dum']
  Y = ['cig_count']

The package sets a default start value of .01 for all parameters.
Users can specify their own starting parameters by creating a list or numpy.array with their desired values.

:func:`zmiopc.iopmod` estimates the ZiOP model and returns :class:`zmiopc.IopModel`.

.. testcode::

   # Model estimation:
   ziop_tob= zmiopc.iopmod('ziop', data, X, Y, Z, method = 'bfgs', weights = 1, offsetx = 0, offsetz = 0)

   # 'ziop' = model to be estimated. In this case 'ziop'
   # data = name of Pandas DataFrame
   # X = variables in the ordered probit stage.
   # Y = dependent variable.
   # Z = variables in the inflation stage.
   # method = method for optimization.  By default set to 'bfgs'
   # weights = weights.
   # offsetx = offset of X.  By Default is zero.
   # offsetz = offset of z


Results from the model:

The following message will appear when the model has converged:

.. testoutput::

         Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 5060.160903
         Iterations: 79
         Function evaluations: 1000
         Gradient evaluations: 100

The run-time for this model is 80.006 seconds (N= 9624).
Object :class:`zmiopc.IopModel` stores model results and goodness-of-fit tests in its attributes 'coefs', 'AIC', 'llik', and 'vcov'.

The following line of code prints the estimates of coefficients:

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

The model object :class:`zmiopc.IopModel` also stores three different diagnostic tests: (1) Log-likelihood, (2) Akaike Information Criteria (AIC), and Variance-Covariance Matrix (VCM).
They can be obtained via the following:

.. testcode::

  print(ziop_tob.llik)
  print(ziop_tob.AIC)
  print(ziop_tob.vcov)

An example for the AIC:

.. testcode::

  print(ziop_tob.AIC)

.. testoutput::

  10138.321806674261

The following funtion extracts predicted probabilities from the model:
:func:`zmiopc.iopfit` returns :class:`zmiopc.FittedVals` containing fitted probablities.

.. testcode::

  fittedziop = ziopc.iopfit(ziop_tob)

  # Print the predicted probabilities
  print(fittedziopc.responsefull)

.. testoutput::

  array[[0.8822262  0.06879832 0.01455244 0.0242539  0.01016914]
 [0.84619828 0.08041296 0.01916279 0.03549797 0.01872801]
 [0.93105632 0.04349743 0.00831396 0.0127043  0.004428  ]
 ...
 [0.73347708 0.1291157  0.03295816 0.06500889 0.03944016]
 [0.87603805 0.06808193 0.01543795 0.02735256 0.01308951]
 [0.82681957 0.08778215 0.02153509 0.04095753 0.02290566]]

:func:`zmiopc.split_effects` and :func:`zmiopc.ordered_effects` compute changes in predicted probabilities when the value of a variable changes in the Inflation or Ordered stages, respectively.

:func:`zmiopc.split_effects` computes how changes in the split-probit covariates affect the probabilities of
being in one population versus another. The example below illustrates the marginal effects of the variable
'gender_dum' on the outcome variable in the ZiOP model estimated above.

.. testcode::

    ziopcgender = zmiopc.split_effects(ziop_tob, 1, nsims = 10000)

The returned dataframe contains predicted probabilities when 'gender_dum' equals 0, and when 'gender_dum' equals 1.

Likewise, :func:`zmiopc.ordered_effects` can also calculate the change in predicted probabilities in each of the ordered outcomes in the ordered-probit stage when the value of a covarariate changes.
Results from :func:`zmiopc.split_effects` and :func:`zmiopc.ordered_effects` can be illustrated using `matplotlib` box plots:

.. testcode::

    gender = zmiopc.ordered_effects(ziop_tob, 2, nsims = 10000)

    # The box plot from the results:
    gender.plot.box(grid='False')

Zero-inflated Ordered Probit (ZiOPC) with Correlated Errors
-----------------------------------------------------------

The package also includes :func:`zmiopc.iopcmod` which fits "zero-inflated" ordered probit models (ZiOPC) under the assumption that the two errors are correlated with each other (i.e. correlated errors).

We first import the required libraries, set up the package and import the dataset:

.. testcode::

  # Import the necessary libraries and IDCeMPy.
  import pandas as pd
  import urllib
  import matplotlib.pyplot as plot
  from idcempy import zmiopc

  # Import the "Youth Tobacco Consumption" dataset.
  url='https://github.com/hknd23/zmiopc/blob/main/data/tobacco_cons.csv'

  # Define a `Pandas` DataFrame.
  data = pd.read_stata(url)

.. testcode::

  # First, define a list of variable names of X, Z, and Y.
  # X = Column names of covariates (from `DataFrame`) used in ordered probit stage.
  # Z = Column names of covariates (from `DataFrame`) used in split-population stage.
  # Y = Column name of ordinal outcome variable (from `DataFrame`).

  X = ['age', 'grade', 'gender_dum']
  Z = ['gender_dum']
  Y = ['cig_count']

:func:`zmiopc.iopcmod` estimates the ZiOPC model using the keyword `'ziopc'` in the first argument:

.. testcode::

   ziopc_tob = zmiopc.iopcmod('ziopc', data, X, Y, Z, method = 'bfgs', weights = 1, offsetx = 0, offsetz = 0)

   # 'ziopc' = model to be estimated. In this case 'ziopc'
   # data = name of Pandas DataFrame
   # X = variables in the ordered probit stage.
   # Y = dependent variable.
   # Z = variables in the inflation stage.
   # method = method for optimization.  By default set to 'bfgs'
   # weights = weights.
   # offsetx = offset of X.  By Default is zero.
   # offsetz = offset of z

The run-time for this ZiOPC model is 4261.707 seconds. The results are stored in the attributes of :class:`zmiopc.IopCModel`.

.. testoutput::

         Current function value: 5060.051910
         Iterations: 119
         Function evaluations: 1562
         Gradient evaluations: 142

The following line of code prints the results:

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

To print the estimates of the log-likelihood, AIC, and Variance-Covariance matrix:

.. testcode::

  # Print Log-Likelihood
  print(ziopc_tob.llik)

  # Print AIC
  print(ziopc_tob.AIC)

  # Print VCOV matrix
  print(ziopc_tob.vcov)

The AIC of the ziopc_tob model, for example, is:

.. testoutput::

  10140.103819465658

The predicted probabilities from the `ziopc_tob` model can be obtained with :func:`zmiopc.iopcfit` as follows.

.. testcode::

  # Define the model for which you want to estimate the predicted probabilities
  fittedziopc = zmiopc.iopcfit(ziopc_tob)

  # Print predicted probabilities
  print(fittedziopc.responsefull)

.. testoutput::

 array[[0.88223509 0.06878162 0.01445941 0.0241296  0.01039428]
 [0.84550989 0.08074461 0.01940226 0.03589458 0.01844865]
 [0.93110954 0.04346074 0.00825639 0.01264189 0.00453143]
 ...
 [0.73401588 0.12891071 0.03267436 0.06438928 0.04000977]
 [0.87523652 0.06888286 0.01564958 0.0275354  0.01269564]
 [0.82678185 0.0875059  0.02171135 0.04135142 0.02264948]]

Similar to the ZiOP model, :func:`zmiopc.split_effects` and :func:`zmiopc.ordered_effects` can also compute changes in predicted probabilities for the ZiOPC model.

.. testcode::

  ziopcgender = zmiopc.split_effects(ziopc_tob, 1, nsims = 10000)

.. testcode::

  # Calculate change in predicted probabilities
  gender = zmiopc.ordered_effects(ziopc_tob, 1, nsims = 10000)

  # Box-plot of precicted probabilities
  gender.plot.box(grid='False')

Middle-inflated Ordered Probit (MiOP) without Correlated Errors
---------------------------------------------------------------

A Middle-inflated Ordered Probit (MiOP) model should be estimated when the ordered outcome variable is inflated in the middle category.

The following example uses 2004 presidential vote data from Elgun and Tilam (`2007 <https://journals.sagepub.com/doi/10.1177/1065912907305684>`_).

We begin by loading the required libraries and IDCeMPy:

.. testcode::

  # Import the necessary libraries and IDCeMPy.
  import pandas as pd
  import urllib
  import matplotlib.pyplot as plot
  from idcempy import zmiopc

Next, we load the dataset:

.. testcode::

  # Import and read the dataset
  url = 'https://github.com/hknd23/idcempy/raw/main/data/EUKnowledge.dta'

  # Define a `Pandas` DataFrame
  data = pd_read.stata(url)

We then define the lists with the names of the variables used in the model

.. testcode::

  # First, define a list of variable names of X, Z, and Y.
  # X = Column names of covariates (from `DataFrame`) used in ordered probit stage.
  # Z = Column names of covariates (from `DataFrame`) used in split-population stage.
  # Y = Column name of ordinal outcome variable (from `DataFrame`).

  X = ['Xenophobia', 'discuss_politics']
  Z = ['discuss_politics', 'EU_Know_ob']
  Y = ['EU_support_ET']

After importing the dataset and specifying the list of variables from it, the MiOP model is estimated with the following step:

.. testcode::

 # Model estimation:
 miop_EU = zmiopc.iopmod('miop', data, X, Y, Z, method = 'bfgs', weights = 1,offsetx = 0, offsetz = 0)

 # 'miop' = Type of model to be estimated. In this case 'miop'
 # data = name of Pandas DataFrame
 # X = variables in the ordered probit stage.
 # Y = dependent variable.
 # Z = variables in the inflation stage.
 # method = method for optimization.  By default set to 'bfgs'
 # weights = weights.
 # offsetx = offset of X.  By Default is zero.
 # offsetz = offset of z

The following message will appear when the model finishes converging:

.. testoutput::

         Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 10857.695490
         Iterations: 37
         Function evaluations: 488
         Gradient evaluations: 61

The run-time for the model is: 18.886 seconds (N= 11887). Print the results of the model with:

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

The model object :class:`zmiopc.IopModel` also stores three different diagnostic tests: (1) Log-likelihood, (2) Akaike Information Criteria (AIC), and Variance-Covariance Matrix (VCM).

.. testcode::

   # Print estimates of LL, AIC and VCOV

   # Print Log-Likelihood
   print(miop_EU.llik)

   # Print AIC
   print(miop_EU.AIC)

   # Print VCOV
   print(miop_EU.vcov)


:func:`zmiopc.iopfit` calculates the predicted probabilities for the MiOP model:

.. testcode::

   # Define the model for which you want to estimate the predicted probabilities
   fittedmiop = zmiopc.iopfit(miop_EU)

   # Print predicted probabilities
   print(fittedmiop.responsefull)

The MiOP model can also work with :func:`zmiopc.split_effects` and :func:`zmiopc.ordered_effects` to compute changes in predicted probabilities when the value of a variable changes:

.. testcode::

    # Define model from which predicted probabilities will be estimated and the number of simulations.
    miopxeno = zmiopc.split_effects(miop_EU, 1, nsims = 10000)

To plot the predicted probabilities:

.. testcode::

     # Get box plot of predicted probabilities
     miopxeno.plot.box(grid='False')

.. testcode::

    # Define model from which predicted probabilities will be estimated and the number of simulations.
    xeno = zmiopc.ordered_effects(miop_EU, 2, nsims = 10000)

    # Get box plot of predicted probabilities
    xeno.plot.box(grid='False')

Middle-inflated Ordered Probit (MiOPC) Model with Correlated Errors
-------------------------------------------------------------------

The steps to estimate the Middle-inflated Ordered Probit (MiOPC) with correlated errors is as follows:

First is importing the data and libraries:

.. testcode::

  # Import the necessary libraries and IDCeMPy.
  import pandas as pd
  import urllib
  import matplotlib.pyplot as plot
  from idcempy import zmiopc

Next, we load the dataset:

.. testcode::

  # Import and read the dataset
  url = 'https://github.com/hknd23/idcempy/raw/main/data/EUKnowledge.dta'

  # Define a `Pandas` DataFrame
  data = pd_read.stata(url)

We then define the lists with the names of the variables used in the model:

.. testcode::

   # First, define a list of variable names of X, Z, and Y.
   # X = Column names of covariates (from `DataFrame`) used in ordered probit stage.
   # Z = Column names of covariates (from `DataFrame`) used in split-population stage.
   # Y = Column name of ordinal outcome variable (from `DataFrame`).

   X = ['Xenophobia', 'discuss_politics']
   Z = ['discuss_politics', EU_Know_ob]
   Y = ['EU_support_ET']

The model can be estimated as follows:

.. testcode::

   # Model estimation
   miopc_EU = zmiopc.iopcmod('miopc', data, X, Y, Z, method = 'bfgs', weights = 1,offsetx = 0, offsetz =0 )

   # 'miopc' = Type of model to be estimated. In this case 'miopc'
   # data = name of Pandas DataFrame
   # X = variables in the ordered probit stage.
   # Y = dependent variable.
   # Z = variables in the inflation stage.
   # method = method for optimization.  By default set to 'BFGS'
   # weights = weights.
   # offsetx = offset of X.  By Default is zero.
   # offsetz = offset of z

The run-time for the model is: 1929.000 seconds (N= 11887). Print model coefficients:

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

The model object :class:`zmiopc.IopCModel` also stores three different diagnostic tests: (1) Log-likelihood, (2) Akaike Information Criteria (AIC), and Variance-Covariance Matrix (VCM).
They can be obtained via the following:

.. testcode::

   # Print Log-Likelihood
   print(miopc_EU.llik)

   # Print AIC
   print(miopc_EU.AIC)

   # Print VCCOV matrix
   rint(miopc_EU.vcov)

To calculate the predicted probabilities:

.. testcode::

   # Define model to fit
   fittedmiopc = zmiopc.iopcfit(miopc_EU)

   # Print predicted probabilities
   print(fittedziopc.responsefull)

The following line of code computes changes in predicted probabilities when the value of a chosen variable in the split stage changes:

.. testcode::

   # Define model from which effects will be estimated and number of simulations
   miopcxeno = zmiopc.split_effects(miopc_EU, 1, nsims = 10000)

A box plot can illustrate the change in predicted probabilities:

.. testcode::

    # Get box plot of predicted probabilities
    miopcxeno.plot.box(grid='False')


To calculate the change in predicted probabilities of the outcome variable in the outcome-stage when the value of a covarariate changes.
The box plots below display the change in predicted probabilities of the outcome variable in the MiOPC model estimated above when Xenophobia increases one standard deviation from its mean value.

.. testcode::

    # Define model from which effects will be estimated and number of simulations
    xeno = zmiopc.ordered_effects(miopc_EU, 2, nsims = 10000)

    # Get box plot of predicted probabilities
    xeno.plot.box(grid='False')


The Standard Ordered Probit (OP) model
--------------------------------------

The package also includes :func:`zmiopc.opmod` that estimates a standard Ordered Probit (OP) model.
The OP model does not account for "zero inflation" or "middle inflation," so it does not have a split-probit stage.

First, import the required libraries and data:

.. testcode::

  # Import the necessary libraries and package
  import pandas as pd
  import urllib
  from idcempy import zmiopc

  # Import the "Youth Tobacco Consumption" dataset.
  url='https://github.com/hknd23/zmiopc/blob/main/data/tobacco_cons.csv'

  # Define a `Pandas` DataFrame
  data = pd.read_csv(url)

The list of variable names for the Independent and Dependent variables needs to be specified:

.. testcode::

  # Define a list of variable names (strings) X,Y:
  # X = Column names of covariates (from `DataFrame`) in the OP equation
  # Y = Column name of outcome variable (from `DataFrame`).

  X = ['age', 'grade', 'gender_dum']
  Y = ['cig_count']

After importing the data and specifying the model, the following code fits the OP model:

.. testcode::

  # Model estimation:
  op_tob = zmiopc.opmod(data, X, Y, method = 'bfgs', weights = 1, offsetx  =0)

  # data = name of pandas DataFrame
  # X = variables in the ordered probit stage.
  # Y = dependent variable.
  # method = method for optimization.  By default set to 'bfgs'
  # weights = weights.
  # offsetx = offset of X.  By Default is zero.
  # offsetz = offset of z


The following message will appear when the model has converged:

.. testoutput::

         Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 4411.710049
         Iterations: 10
         Function evaluations: 976
         Gradient evaluations: 121

The model's run-time is 37.694 seconds (N= 9624). :class:`zmiopc.OpModel` stores results from model estimation and other information in its attributes.
The following line of code to see the estimates of coefficients:

.. testcode::

   # Print coefficients of the models
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

  # Print Log-Likelihood
  print(op_tob.llik)

  # Print AIC
  print(op_tob.AIC)

  # Print VCOV matrix
  print(op_tob.vcov)

The Vuong Test
--------------

Harris and Zhao (`2007 <https://doi.org/10.1016/j.jeconom.2007.01.002>`__) suggest that a variant of the Vuong (`1989 <https://www.jstor.org/stable/1912557>`__)
Test (with a v statistic) can be used to compare the performance of the ZiOP versus the standard Ordered Probit (OP) model. The Vuong's test formula is:

.. math::

    v = \frac{\sqrt{N}(\frac{1}{N}\sum_{i}^{N}m_{i})}{\sqrt{\frac{1}{N}\sum_{i}^{N}(m_{i}-\bar{m})^{2}}}

where v < -1.96 favors the more general (ZiOP/ZiOPC) model, -1.96 < v < 1.96 lends no support to either model, and v > 1.96 supports the simpler (OP) model.

The OP and ZiOP models must have the same number of observations, and the OP must have the same number of covariates as ZiOP's OP stage.
The statistic below reveals that the OP model is preferred over the ZiOP model.

.. testcode::

   # Estimate Vuong test.  OP model first, ZIOP model specified next in this case
   zmiopc.vuong_opiop(op_tob, ziop_tob)

.. testoutput::

   6.624742132792222

The Vuong test can also be implemented to compare the ZiOPC, MiOP and MiOPC models with the OP model.

Generalized Inflated Multinomial Logit (GiMNL) Model
----------------------------------------------------

The :py:mod:`gimnl` module provides :func:`gimnl.gimnlmod` to estimate the General "inflated" Multinomial Logit models (GiMNL) with three outcomes in the dependent variable.
The GiMNL model minimize issues present when unordered polytomous outcome variables have an excessive share and heterogeneous pool of observations in the lower category.

Similar to the models in the :py:mod:`zmiopc` module, the first step is to import the libraries and 2004 presidential vote choice dataset.

.. testcode::

  # Import the module
  import pandas as pd
  import urllib
  from idcempy import gimnl

  # Load the dataset
  url= 'https://github.com/hknd23/zmiopc/raw/main/data/replicationdata.dta'

  # Define a `Pandas` DataFrame
  data = pd.read_stata(url)

We the define the list of covariates in the split-stage (z), the multinomial logit-stage (x) and the outcome variable (y).
The values of the dependent variable must be represented numerically as "0", "1", and "2" to represent each category.
To specify the baseline/reference category, users provide a three-element list for the `reference` argument (e.g [0,1,2]).
The first element of the list is the baseline/reference category.

.. testcode::

   # x = Column names of covariates (from `DataFrame`) in the outcome-stage.
   # z = Column names of covariates (from `DataFrame`) in the split-stage.
   # y = Column names of outcome variable (from `DataFrame`).

   x = ['educ', 'party7', 'agegroup2']
   z = ['educ', 'agegroup2']
   y = ['vote_turn']

The flexibility of :func:`gimnl.gimnlmod` allows users to customize the baseline and inflated categories.
Users can employ the argument `inflatecat` with `'baseline'`, `'second'`, or `'third'` to specify any unordered category as the inflated category (dictated by the distribution) in their unordered-polytomous outcome measure.
If `'baseline'` is selected, the first element (baseline/reference category) in `reference` is the inflated outcome.
Likewise, if `'second'` or `'third'` is selection, the second or third element will be the inflated outcome. The following code specifies the outcome '0' (Abstain) as both the baseline and inflated category.

.. testcode::

   # Define order of variables
   order = [0, 1, 2]

   # Define "inflation" category
   inflatecat = "baseline"

.. testcode::

   # Estimate the model
   gimnl_2004vote = gimnl.gimnlmod(data, x, y, z, method = 'bfgs', order, inflatecat)

   # data = name of pandas DataFrame.
   # x = variables in the ordered stage.
   # y = dependent variable.
   # z = variables in the inflation stage.
   # method = optimization method.  Default is 'bfgs'
   # order = order of variables.
   # inflatecat = inflated category.

The following line of code prints the coefficients of the covariates:

.. testcode::

   # Print coefficients
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

The model's run-time is 16.646 seconds (N= 1341). The results from the model are stored in a :class:`gimnlModel` with the following attributes:

- coefs: Model coefficients and standard errors.
- llik: Log-likelihood.
- AIC: Akaike information criterion.
- vcov: Variance-covariance matrix.

For example, AIC can be printed as follows.

.. testcode::

  # Print Log_Likelihood
  print(gimnl_2004vote.llik)

  # Print AIC
  print(gimnl_2004vote.AIC)

  # Print VCOV matrix
  print(gimnl_2004vote.vcov)

Users can fit a standard three-category Multinomial Logit Model (MNL) by specifying the list of **x**, **y**, and baseline (using `reference`).

.. testcode::

   #Estimate the model
   mnl_2004vote = gimnl.mnlmod(data, x, y, method = 'bfgs')

   # data = name of Pandas DataFrame.
   # x = variables in MNL stage.
   # y = dependent variable
   # method = optimization method. Default is 'bfgs'

   # Print the coefficients
   print(mnl_2004vote.coefs)

.. testoutput::

     Coef        SE  tscore     p   2.5%  97.5%
  1: int       -4.914 0.164 -29.980 0.000 -5.235 -4.593
  1: educ       0.455 0.043  10.542 0.000  0.371  0.540
  1: party7     0.462 0.083   5.571 0.000  0.300  0.625
  1: agegroup2  0.951 0.029  32.769 0.000  0.894  1.008
  2: int        0.172 0.082   2.092 0.036  0.011  0.334
  2: educ       0.282 0.031   9.011 0.000  0.221  0.343
  2: party7    -0.567 0.085  -6.641 0.000 -0.734 -0.399
  2: agegroup2  0.899 0.138   6.514 0.000  0.629  1.170

The MNL model's run-time is 8.276 seconds. Similar to the GiMNL model, the AIC for the MNL model can also be given by:

.. testcode::

  # Print Log-Likelihood
  print(mnl_2004vote.AIC)

  # Print AIC
  print(mnl_2004vote.AIC)

  # Print VCOV matrix
  print(mnl_2004vote.vcov)

Contributions
=============

The authors welcome and encourage new contributors to help test `IDCeMPy` and add new functionality.
You can find detailed instructions on "how to contribute" to `IDCeMPy` `here <https://github.com/hknd23/idcempy/blob/main/CONTRIBUTING.md>`_.

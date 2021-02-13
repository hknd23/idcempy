# IDCeMPy: Estimation of "Inflated" Discrete Choice Models

*Nguyen K. Huynh, Sergio Bejar, Vineeta Yadav, Bumba Mukherjee*

<!-- badges: start -->

[![PyPI version fury.io](https://badge.fury.io/py/ziopcpy.svg)](https://pypi.org/project/ziopcpy/0.1.2/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ziopcpy.svg)](https://pypi.org/project/ziopcpy/0.1.2/)
[![Downloads](https://pepy.tech/badge/ziopcpy)](https://pepy.tech/project/ziopcpy)
[![Project Status: Active â€“ The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
<!-- badges: end -->

**IDCeMPy** is a Python package that provides fucntions to fit and assess the performance of three distinct
sets of “inflated” discrete choice models. Specifically, it contains functions to:

* Fit the Zero-Inflated Ordered Probit (ZIOP) model without and with correlated errors (ZIOPC
model) to evaluate zero-inflated ordered choice outcomes that results from a dual data generating
process (d.g.p.).
* Fit the Middle-Inflated Ordered Probit (MIOP) model without and with correlated errors (MIOPC) to account for the inflated middle-category in ordered choice measures that relates to a dual d.g.p.
* Fit inflated Multinomial Logit (IMNL) models that account for the preponderant and heterogeneous share of observations in the baseline or any lower category in unordered polytomous choice outcomes.
* •	Compute goodness-of-fit (AIC and Log-likelihood) statistics and the Vuong Test statistic to assess the performance of each "inflated" discrete choice model included in the package.

**IDCeMPy** uses Newton numerical optimization methods to estimate the "inflated" discrete choice models described above via Maximum Likelihood Estimation (MLE).  
**IDCeMPY** is compatible with [Python](https://python.org) 3.7+

## Why **IDCeMPy**?

An excessive (“inflated”) share of observations—stemming from two distinct d.g.p’s—fall into a single choice category in many ordered and unordered polytomous outcome variables. Standard discrete choice Ordered Probit and Multinomial Logit models cannot account for such category inflation which leads to biased inferences. Examples include,

*	The inflated zero category of no smoking in ordered measures of self-reported smoking behavior is generated from nonsmokers who never smoke cigarettes and those who smoked previously but temporarily stopped smoking because of high cigarette prices.

*	The inflated indifference middle category within survey measures of immigration attitudes includes respondents truly indifferent to immigration and those that choose indifference for social desirability reasons.  

*	The inflated baseline or other lower outcome categories of unordered polytomous outcome measures of vote choice include nonvoters who temporarily abstain from voting and routine nonvoters who always abstain. 

**IDCeMPy** includes the ZIOP(C) models for evaluating zero-inflated ordered choice outcomes stemming from a dual d.g.p, MIOP(C) models that address inflated middle-category ordered outcome measures arising from distinct d.g.p’s, and IMNL models that account for inflated baseline or other lower categories in unordered polytomous outcome variables. 

Each inflated discrete choice model in this package addresses category inflation in one’s discrete outcome—unordered or unordered polytomous—of interest by jointly estimating a binary split-stage equation and an ordered or multinomial discrete choice outcome equation.   

## Functions in the **IDCeMPy** Package

| Function         | Description                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `opmod`; `iopmod`; `iopcmod` |Fits respectively the ordered probit (OP) model, the Zero-Inflated (ZIOP) and Middle-Inflated ordered probit (MIOP) models without correlated errors, and the ZIOPC and MIOPC models that incorporate correlated errors. |
|`opresults`; `iopresults`; `iopcresults`| Presents covariate estimates, the Variance-Covariance (VCV) matrix, and goodness-of-fit statistics (Log-Likelihood and AIC) of `opmod`, `iopmod`, and `iopcmod`. |
| `iopfit`; `iopcfit`| Computes the fitted probabilities from each estimated mode's object.|
| `vuong_opiop`;  `vuong_opiopc` | Calculates Vuong test statistic for comparing the performance of the OP with the ZiOP(C) and MiOP(C) models.|
|`split_effects`; `ordered_effects`| Estimates marginal effects of covariates in the split-stage and outcome-stage respectively.|
|`imnlmod` | fits baseline and other lower-category inflated MNL models.|
|`imnlresults` | Presents covariate estimates, Variance-Covariance (VCV) matrix, and goodness-of-fit statistics of `imnlmod`.|

## Dependencies
- scipy
- numpy
- pandas

## Installation

From [PyPi](https://pypi.org/project/ziopcpy/0.1.2/):

```sh
pip install IDCeMPy
```

From [GitHub](https://github.com/)

```sh
git clone https://github.com/hknd23/idcempy.git
cd idcempy
python setup.py install
```
On readthedocs you will find the installation guide, a complete overview of all the features included in **IDCeMPy**, and example scripts of
all the models.

## Using the Package

### Example 1: Zero-inflated Ordered Probit Models with Correlated Errors (ZiOPC)
We first illustrate how **IDCeMPy** can be used to estimate models when the ordered outcome variable presents "zero-inflation."
For that purpose we use data from the 2018 [National Youth Tobacco Dataset](https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/index.htm).  As mentioned above, **IDCeMPy** allows you to estimate "Zero-inflated" Ordered Probit models with and without correlated errors.

We demonstrate the use of a "Zero-inflated" Ordered Probit Model with correlated errors (ZMiOPC).  An example of the ZiOP model without correlated erros can be found in the documentation of the package.

First, import `IDCeMPy`, required packages, and dataset.

```python
from idcempy import zmiopc
import pandas as pd
import urllib
url = 'https://github.com/hknd23/idcempy/raw/main/data/tobacco_cons.csv'
data = pd.read_csv(url)
```

We now specify arrays of variable names (strings) X, Y, Z.

```python
X = ['age', 'grade', 'gender_dum']
Y = ['cig_count']
Z = ['gender_dum']
```

In addition, we define an array of starting parameters before estimating the `ziopc` model. If starting parameters are not specified, the function automatically generates them.

```python
pstart = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01])
```

The following line of code creates a ziopc regression object model.

```python
ziopc_tob = zmiopc.iopcmod('ziopc', pstartziopc, data, X, Y, Z, method='bfgs',
                    weights=1, offsetx=0, offsetz=0)
```

If you like to estimate your model without correlated errors, you only substitute the parameter 'ziopc' for 'ziop'.


The results of this example are stored in a class (`ZiopcModel`) with the following attributes:

* *coefs*: Model coefficients and standard errors
* *llik*: Log-likelihood
* *AIC*: Akaike information criterion
* *vcov*: Variance-covariance matrix

We, for example, can print out the covariate estimates, standard errors, *p* value and *t* statistics by typing:

```python
print(ziopc_tobb.coefs)
```

```python
                          Coef        SE     tscore             p       2.5%      97.5%

Split-stage
-----------------------
intercept              9.538072  3.470689   2.748178  5.992748e-03   2.735521  16.340623
gender_dum            -9.165963  3.420056  -2.680062  7.360844e-03 -15.869273  -2.462654

Second-stage
-----------------------
age                   -0.028606  0.008883  -3.220369  1.280255e-03  -0.046016  -0.011196
grade                  0.177541  0.010165  17.465452  0.000000e+00   0.157617   0.197465
gender_dum             0.602136  0.053084  11.343020  0.000000e+00   0.498091   0.706182
cut1                   1.696160  0.044726  37.923584  0.000000e+00   1.608497   1.783822
cut2                  -0.758095  0.033462 -22.655678  0.000000e+00  -0.823679  -0.692510
cut3                  -1.812077  0.060133 -30.134441  0.000000e+00  -1.929938  -1.694217
cut4                  -0.705836  0.041432 -17.036110  0.000000e+00  -0.787043  -0.624630
rho                   -0.415770  0.074105  -5.610526  2.017123e-08  -0.561017  -0.270524
```

Or the Akaike Information Criterion (AIC):

```python
print(ziopc_tobb.AIC)
```

```python
16061.716497590078
```

`split_effects` creates a dataframe with values of the change in predicted probabilities of the outome variable when 'gender_dum' equals 0, and when 'gender_dum' equals 1. The box plots below illustrate the change in predicted probablities using the values from the 'ziopc' dataframe.

```python
ziopcgender_split = zmiopc.split_effects(ziopc_tob, 1)
ziopcgender_split.plot.box(grid='False')
```

<p align="center">
   <img src="https://github.com/hknd23/idcempy/blob/main/graphics/ziopc_me.png?raw=true" width="500" height="300" />
</p>

Similarly `ordered_effects`, calculates the change in predicted probabilities of each outcome in the ordered stage.

```python
ziopcgender_ordered = zmiopc.ordered_effects(ziopc_tob, 2)
ziopcgender_ordered.plot.box(grid='False')
```

<p align="center">
   <img src="https://github.com/hknd23/idcempy/blob/main/graphics/ZiOPC_Order_Gender.png" width="500" height="300" />
</p>

Module `zmiopc` also provides function `vuong_opiopc` to compare the performace of the ZiOPC model with a standard Ordered Probit model (also available through `opmod`).

```python
op_tob = zmiopc.opmod(data, X, Y)
zmiopc.vuong_opiopc(op_tob,ziopc_tob)
```

```python
6.576246015382724
```

### Example 2: "Middle-inflated" Ordered Probit Models with Correlated Errors (MiOPC)
You can also use **IDCeMPy** to estimate "inflated" Ordered Probit models if your outcome variable presents inflation in the "middle" category. For the sake of consistency, we present below the code needed to estimate a "Middle-inflated" Ordered Probit Model with correlated errors. Data fot this example comes from Elgün and Tillman ([2007](https://journals.sagepub.com/doi/10.1177/1065912907305684)).

First, load the dataset.

```python
url= 'https://github.com/hknd23/idcempy/raw/main/data/EUKnowledge.dta'
data= pd.read_stata(url)
```

Now, define the lists with names of the covariates you would like to include in the split-stage (Z) and the second-stage (X) as well as the name of your "middle-inflated" outcome variable (Y).

```python
Y = ["EU_support_ET"]
X = ['Xenophobia', 'discuss_politics']
Z = ['discuss_politics', 'EU_Know_obj']
```

Run the model and print the results:

```python
miopc_EU = zmiopc.iopcmod('miopc', DAT, X, Y, Z)
```

```python
print(miopc_EU.coefs)

                              Coef    SE  tscore     p   2.5%  97.5%
Split-stage
---------------------------
int                         -0.129 0.021  -6.188 0.000 -0.170 -0.088
discuss_politics             0.192 0.026   7.459 0.000  0.142  0.243
EU_Know_obj                  0.194 0.027   7.154 0.000  0.141  0.248

Second-stage
---------------------------
Xenophobia                  -0.591 0.045 -13.136 0.000 -0.679 -0.502
discuss_politics            -0.029 0.021  -1.398 0.162 -0.070  0.012
cut1                        -1.370 0.044 -30.948 0.000 -1.456 -1.283
cut2                        -0.322 0.103  -3.123 0.002 -0.524 -0.120
rho                         -0.707 0.106  -6.694 0.000 -0.914 -0.500
```



<p align="center">
   <img src="https://github.com/hknd23/idcempy/raw/main/graphics/MiOPC_Split_EUKnow.png" width="500" height="300" />
</p>

`ordered_effects()` calculates the change in predicted probabilities of the outcome variable when the value of a covarariate changes. The box plots below display the change in predicted probabilities when Xenophobia increases one standard deviation from its mean value.

```python
xeno = zmiopc.ordered_effects(miopc_EU, 2)
xeno.plot.box(grid='False')
```

<p align="center">
   <img src="https://github.com/hknd23/idcempy/blob/main/graphics/MiOPC_Xenophobia.png" width="500" height="300" />
</p>

The performance of a MiOPC model can also be compared with an Ordered Probit with the Vuong test: 

```python
op_EU = zmiopc.opmod(DAT, X, Y)
zmiopc.vuong_opiopc(op_EU, miopc_EU)
```

```python
-10.435718518003675
```

### Example 3: Estimation of "inflated" Multinomial Logit Models
Unordered polytomous outcome variables sometimes present inflation in the baseline category, and not accounting for it could lead you to make faulty inferences.  But **IDCeMPy** has functions that make it easier for you to estimate Multinomial Logit Models that account for such inflation (iMNL).  This example shows how you can estimate iMNL models easily.
Data comes from Arceneaux and Kolodny ([2009](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-5907.2009.00399.x))


We begin by importing the `imnl` module and the 2004 Presidential Vote dataset.

```python
from zmiopc import imnl
url= 'https://github.com/hknd23/idcempy/raw/main/data/replicationdata.dta'
data= pd.read_stata(url)
```

Define the outcome variable (y, whose categories are numerically represented by 0, 1, and 2), covariates in the split-stage (z) and second-stage (x).

```python
x = ['educ', 'party7', 'agegroup2']
z = ['educ', 'agegroup2']
y = ['vote_turn']
```

```python
reference = [0, 1, 2]
inflatecat = "baseline"
```

The following line of code estimates the "inflated" Multinomial Logit Model (iMNL). Through the argument `reference`, users can select which category of the dependent variable as the baseline, or 'reference' category by placing it first. `imnlmod` can account for inflation in any of the three catergories. Argument `inflatecat` allows user to specify the inflated category. In this example, '0' is the baseline and inflated category.

```python
imnl_2004vote = imnl.imnlmod(data, x, y, z, reference, inflatecat)
```

Print the est

```python
                       Coef    SE  tscore     p    2.5%  97.5%
Split-stage
----------------------
intercept            -4.935 2.777  -1.777 0.076 -10.379  0.508
educ                  1.886 0.293   6.441 0.000   1.312  2.460
agegroup2             1.295 0.768   1.685 0.092  -0.211  2.800

Ordered-stage: 1
---------------------
intercept            -4.180 1.636  -2.556 0.011  -7.387 -0.974
educ                  0.334 0.185   1.803 0.071  -0.029  0.697
party7                0.454 0.057   7.994 0.000   0.343  0.566
agegroup2             0.954 0.248   3.842 0.000   0.467  1.441

Ordered-stage: 2
----------------------
intercept             0.900 1.564   0.576 0.565  -2.166  3.966
educ                  0.157 0.203   0.772 0.440  -0.241  0.554
party7               -0.577 0.058  -9.928 0.000  -0.691 -0.463
agegroup2             0.916 0.235   3.905 0.000   0.456  1.376
```

---
title: 'ZiopcPy: A Python Package for the Zero-Inflated Ordered Probit Models Without and With Correlated Errors'
authors:
- affiliation: 1
  name: Nguyen K. Huynh
- affiliation: 2
  name: Sergio Béjar
  orcid: 0000-0002-9352-3892
- affiliation: 3
  name: Nicolás Schmidt
  orcid: 0000-0001-5083-5792
- affiliation: 1
  name: Vineeta Yadav
- affiliation: 1
  name: Bumba Mukherjee
date: "02 February 2020"
output:
  html_document:
    df_print: paged
  pdf_document: default
bibliography: paper.bib

tags:
- Python

affiliations:
- index: 1
  name: Dept. of Political Science, Pennsylvania State University
- index: 2
  name: Dept. of Political Science, San Jose State University
- index: 3
  name: Dept. of Political Science. Universidad de la República, UY

---
# Summary


Ordered dependent variables that range from the lowest outcome category of zero to a higher category are employed in Biomedical research, Economics, Political Science, Psychology and Public Health. For example, in Public Health, self-reported ordered survey response measures on severity of depression are operationalized as 0 for none, 1 for mild, 2 for moderate, and 3 for severe [@crossley2002reliability; @baker2004self; @greene2015inflated]. Ordered survey response data on individual smoking behavior evaluated by economists range from 0 for “no smoking” to 3 for “smoking 20 or more cigarettes per day” [@harris2007zero]. Such ordered dependent variables are analyzed via standard ordered probit (OP) or ordered logit (OL) models. However, the OP (and OL) model cannot statistically account for the preponderance of zero observations that often exists in the lowest (i.e., “zero”) outcome category in ordered dependent variables, particularly when these zeros result from two distinct data generating processes [@harris2007zero; @bagozzi2015modeling]. An example of a “zero-inflated” ordered dependent variable in which the lowest outcome category has excessive zero observations produced by two data generating processes (d.g.p’s) is self-reported smoking behavior. Indeed, the high share of zeros observed in the “no smoking” outcome category is recorded as 0 for individuals in Regime 0 (“always-zero” group) who never smoke cigarettes and for those in Regime 1 who smoked previously but temporarily stopped smoking because of the high price of cigarettes[@harris2007zero: 1074]. 

Excessive zeros observed in the lowest outcome of the ordered depression severity response variable also emerges from two populations: one group that has never experienced depression, and a second group that experienced clinical depression in the past but not during the assessed time-period analyzed in the survey questionnaire. Failing to account for excessive zeros produced by distinct d.g.p’s in a zero-inflated ordinal dependent variable—as done by the OP and OL model—leads to model misspecification and biased estimates. To address this limitation, @harris2007zero developed the Zero-Inflated Ordered Probit (ZiOP) model that accounts for excessive zeros in ordered dependent variables that relate to two d.g.p’s. The ZiOP model does so by jointly estimating two latent equations: (1) the probit split-stage equation that estimates the effect of covariates on the probability of observations being in Regime 0 versus Regime 1, and (2) an OP outcome equation that estimates the effect of a second set of covariates on the probability of observing each ordered category, conditional on observations being in Regime 0. The stochastic error terms from these two latent equations are correlated in the ZiOPC model but are independent in the ZiOP model.  By jointly estimating the two aforementioned latent equations, the ZiOP(C) models avoid model misspecification since they statistically account for the preponderant share of zero observations in one’s zero-inflated ordinal dependent variable that results from two d.g.p’s. Our `ZiopcPy` Python package described below contains functions to fit the OP, ZiOP and ZiOPC models, and assess their performance.


# Statement of Need 

The ZiOP model without correlated errors is estimated in Limdep/NLogit and by using dale2018estimation `zioprobit` command in STATA 15. The `zioprobit` command, however, does not estimate the ZiOP with correlated errors even though the error term from the model’s two latent equations are often correlated [@harris2007zero; @bagozzi2015modeling]. There also does not exist any Python package that provides functions to estimate the ZiOP model with and without correlated errors, and assess these models using post-estimation commands. Our `ZiopcPy` Python package incorporates functions to estimate and evaluate the OP model, the ZiOP model without correlated errors, and ZiOPC model with correlated errors. The probit split-stage equation and OP outcome-stage equation are jointly estimated for both the ZiOP and ZiOPC models. By combining two probability distributions—via estimation of these two latent equations—that jointly produce the observed data, the ZiOP(C) models statistically account for the dual d.g.p. that generate excessive zero observations in zero-inflated ordered dependent variables. This feature avoids model misspecification and provides accurate estimates when evaluating the impact of covariates on zero-inflated ordered dependent variables. `ZiopcPy` also includes functions to compute the probability with which observations in the sample are in the Regime 0 versus Regime 1 group in the zero-inflated outcome category, assess the models’ goodness-of-fit, obtain the covariates marginal effects, and implement tests to perform model comparison. The ZiOP(C) models can evaluate the following zero-inflated ordered dependent variables: self-assessed health status or depression severity [@greene2015inflated], self-reported smoking behavior [@harris2007zero], interstate conflict escalation [@senese1997between; @bagozzi2015modeling], and state-perpetrated repression [@besley2009repression].


# ZiopcPy Python Package  

`ZiopcPy` contains the following functions to estimate the OP, ZiOP and ZiOPC models via Maximum Likelihood Estimation using Newton numerical optimization methods: 

* `opmod`; `iopmod`; `iopcmod`: `opmod` fits the standard OP model. `iopmod` and `iopcmod` fit the Zero-Inflated Ordered Probit model without and with correlated errors respectively.
* `opresults`; `iopresults`; `iopcresults`: Stores and presents the covariate estimates, variance-covariance (VCV) matrix, log-likelihood and AIC results from the OP, ZiOP, and ZiOPC model.   
* `iopfit`; `iopcfit`: Computes the fitted probabilities from the estimated `iopmod` and `iopcmod`.
* `vuong_opiop`; `vuong_opiopc:` Calculates the Vuong test statistic to compare the performance of the OP to the ZiOP and ZiOPC model. 

The covariate estimates, VCV matrix, and fitted probabilities from the estimated ZiOP(C) models in `ZiopcPy` can be used to calculate the marginal effect of the ZiOP(C) models’ (i) split-stage covariates on the probability of observations being in the always-zero versus Regime 1 group and (ii) outcome-stage covariates on the probability of each outcome category, conditional on observations being in the always-zero group. The Vuong test results evaluate the three estimated models’ performance and assesses whether the ZiOP(C) models are necessary to fit the relevant data.  To illustrate the `ZiopcPy` package's functionality, all the functions listed above are evaluated using the political violence ordered dependent variable in Besley and Persson’s (2009) data that is described in the package.

# Availability 
`ZiopcPy` is written in Python 3. Code and detailed installation instructions can be found at https://github.com/hknd23/ziopcpy. `ZiopcPy` is also available on [PyPi](https://pypi.org/project/ziopcpy/0.1.2/)

# References















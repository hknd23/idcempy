---
title: 'IDCeMPy: Python Package for Inflated Discrete Choice Models'

authors:
- affiliation: 1
  name: Nguyen K. Huynh
  orcid: 0000-0002-6234-7232
- affiliation: 2
  name: Sergio Béjar
  orcid: 0000-0002-9352-3892
- affiliation: 1
  name: Vineeta Yadav
- affiliation: 1
  name: Bumba Mukherjee
date: "21 June 2021"
output:
  html_document:
    df_print: paged
  pdf_document: default
bibliography: paper.bib

tags:
- Python, Inflated Ordered Probit Models, Generalized Inflated MNL Models

affiliations:
- index: 1
  name: Dept. of Political Science, Pennsylvania State University
- index: 2
  name: Dept. of Political Science, San Jose State University
  
---
# Summary

Scholars and Data Scientists often use discrete choice models to evaluate ordered dependent variables using the ordered probit model and unordered polytomous outcome measures via the multinomial logit (MNL) estimator [@greene2002nlogit; @JSSv074i10; @richards2018new]. These models, however, cannot account for the possibility that in many ordered and unordered polytomous choice outcomes, an excessive share of observations —stemming from two distinct data generating processes (d.g.p’s)— fall into a single category which is thus “inflated.” For instance, ordered outcome measures of self-reported smoking behavior that range from 0 for “no smoking” to 3 for “smoking 20 cigarettes or more daily” contain excessive observations in the zero (no smoking) category that includes individuals who never smoke cigarettes and those who smoked previously but temporarily stop smoking because of an increase in cigarette costs [@harris2007zero; @greene2015inflated]. The “indifference” middle-category in ordered measures of immigration attitudes is inflated since it includes respondents who are genuinely indifferent about immigration and those who select “indifference” because of social desirability reasons [@bagozzi2012mixture; @brown2020modelling]. The baseline category of unordered polytomous variables of Presidential vote choice is also often inflated as it includes non-voters who abstain from voting owing to temporary factors and routine non-voters who are disengaged from the political process [@campbell2008religion; @bagozzi2017distinguishing].  Inflated discrete choice models have been developed to address such category inflation in ordered and unordered polytomous outcome variables as failing to do so leads to model misspecification and incorrect inferences [@harris2007zero; @bagozzi2012mixture; @brown2020modelling].

`IDCeMPy` is an open-source Python package that enables researchers to fit three distinct sets of discrete choice models used by data scientists, economists, engineers, political scientists, and public health researchers: the Zero-Inflated Ordered Probit (ZiOP) model without and with correlated errors (ZiOPC model), Middle-Inflated Ordered Probit (MiOP) model without and with correlated errors (MiOPC), and Generalized-Inflated Multinomial Logit (GiMNL) models. Functions that fit the ZiOP(C) model in `IDCeMPy` evaluate zero-inflated ordered dependent variables that result from two d.g.p’s, while functions that fit the MiOP(C) models account for inflated middle-category ordered outcomes that emerge from distinct d.g.p’s. The functions in `IDCeMPy` that fit GiMNL models account for the large share and heterogeneous mixture of observations in the baseline and other lower outcome categories in unordered polytomous dependent variables. The primary location for the description of the functions that fit the models listed above is available at the [IDCeMPy package’s documentation website](https://idcempy.readthedocs.io/en/latest/).

# State of the Field

Software packages and code are available for estimating standard (non-inflated) discrete choice models. In the R environment, the packages `MASS` [@venables2002random] and `micEcon` [@henningsen2014micecon] fit binary and discrete choice models. The package `Rchoice` [@JSSv074i10] allows researchers to estimate binary and ordered probit and logit models as well as the Poisson model by employing various optimization routines. The proprietary LIMDEP package NLOGIT [@greene2002nlogit] fits conventional binary and ordered discrete choice models but is neither open-sourced nor freely available. The R packages `mlogit` [@croissant2012estimation] and `mnlogit` [@hasan2014fast] provide tools for working with conventional MNL models, while `gmnl` [@sarrias2017multinomial] and `PReMiuM` [@liverani2015premium] estimate MNL models that incorporate unit-specific heterogeneity. There exists proprietary LIMDEP software and R code —but not an R package— that fit few inflated ordered probit and MNL models [@harris2007zero; @bagozzi2012mixture; @bagozzi2017distinguishing]. Outside R, the Python package `biogeme` [@bierlaire2016pythonbiogeme] fits mixed logit and MNL models. Further, @dale2021estimation’s ZiOP STATA command (but not package) fits the Zero-Inflated Ordered Probit without correlated errors. @xia2019gidm’s `gidm` STATA command fits discrete choice models without correlated errors for inflated zero and other lower-category discrete outcomes. 

The R or LIMDEP software, along with the STATA commands listed above, are undoubtedly helpful. However, to our knowledge, there are no R or Python packages to fit a variety of statistical models that account for the excessive (i.e., “inflated”) share of observations in the baseline, and other higher categories of ordered and unordered polytomous dependent variables, which are commonly analyzed across the natural and social sciences. As discussed below, our Python package `IDCeMPy` thus fills an important lacuna by providing an array of functions that fit a substantial range of inflated discrete choice models applicable across various disciplines.

# Statement of Need 

Although our `IDCeMPy` package also fits standard discrete choice models, what makes it unique is that unlike existing software, it offers functions to fit and assess the performance of both Zero-Inflated and Middle-Inflated Ordered Probit (OP) models without and with correlated errors as well as a set of Generalized-Inflated MNL models. The models included in `IDCeMPy` account for the excessive proportion of observations in any given ordered or unordered outcome category by combining a single binary probit or logit split-stage equation with either an ordered probit outcome stage (for the Zero and Middle-Inflated OP models) or an MNL outcome-stage equation. Users can treat the error terms from the two equations in the Zero and Middle-Inflated OP models as independent or correlated in the package’s estimation routines. `IDCeMPy` also provides functions to assess each included model’s goodness-of-fit via the AIC statistics, extract the covariates’ marginal effects from each model, and conduct Vuong tests for comparing the performance between the standard and inflated discrete choice models. 

The functions in `IDCeMPy` use quasi-Newton optimization methods such as the Broyden-Fletcher-Goldfarb-Shanno algorithm for Maximum-Likelihood-Estimation (MLE), which facilitates convergence and estimation speed. Another feature is that the coefficients, standard errors, and confidence intervals obtained for each model estimated in `IDCeMPy` are in `pandas.DataFrame` [@mckinney-proc-scipy-2010] format and are stored as class attribute `.coefs`. This allows for easy export to csv or excel, which makes it easier for users to perform diagnostic tests and extract marginal effects. `IDCeMPy` is thus essential as it provides a much-needed unified software package to fit statistical models to account for category inflation in several ordered and unordered outcome variables used across fields as diverse as Economics, Engineering, Marketing, Political Science, Public Health, Sociology, and Transportation research. Users can employ the wide range of statistical models in `IDCeMPy` to assess:

- Zero-inflation in self-reported smoking behavior [@harris2007zero], demand for health treatment [@greene2015inflated], and accident injury-severity [@fountas2018analysis].

- Middle-category inflation in ordered measures of monetary policy [@brown2020modelling] and European Union (EU) membership attitudes [@elgun2007exposure].

- Inflated unordered polytomous outcomes such as transportation choice, environmental policy and consumer demand [@richards2018new], and Presidential vote choice [@campbell2008religion].

# Functionality and Applications

`IDCeMPy` contains the functions listed below to estimate via MLE the following inflated discrete choice models listed earlier:

* `opmod`; `iopmod`; `iopcmod`: Fits the ordered probit model, the Zero-Inflated (ZIOP) and Middle-Inflated ordered probit (MIOP) models without correlated errors, and the ZIOPC and MIOPC models that incorporate correlated errors.

* `opresults`; `iopresults`; `iopcresults`: Presents covariate estimates, Variance-Covariance (VCV) matrix, Log-Likelihood, and AIC statistics of the object models.

* `iopfit`; `iopcfit`: Computes fitted probabilities from each estimated model’s objects.

* `vuong_opiop`; `vuong_opiopc`: Calculates Vuong test statistic for comparing the performance of the OP with the ZiOP(C) and MiOP(C) models.

* `split_effects`; `ordered_effects`: Estimates marginal effects of covariates in the split-stage and outcome-stage respectively. 

* `mnlmod`; `gimnlmod`: Fits MNL model and Generalized-Inflated MNL models.

* `mnlresults`; `gimnlresults`; `vuong_gimnl`: Presents covariate estimates, VCV matrix, Log-Likelihood, and AIC statistics of `mnlmod`; `gimnlmod`. Vuong test statistic for comparing MNL to GIMNL models obtained from `vuong_gimnl`. 


Details about the functionality summarized above are available at the [package’s documentation website](https://idcempy.readthedocs.io/en/latest/), which is open-source and hosted by [ReadTheDocs](https://readthedocs.org/). The features of the functions in `IDCeMPy` that fit the, 

(i)	ZiOP(C) models are presented using the ordered self-reported tobacco consumption dependent variable from the [2018 National
Youth Tobacco Dataset](https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/index.htm), 

(ii)	MiOP(C) models are illustrated using the ordered EU support outcome variable from @elgun2007exposure. 

(iii)	GiMNL models are evaluated using the unordered polytomous Presidential vote choice dependent variable from @campbell2008religion. 


# Availability and Installation

`IDCeMPy` is an Open-source software made available under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0). It can be installed from [PyPi](https://pypi.org/project/idcempy/) or from its [GitHub repository](https://github.com/hknd23/idcempy). 

# References

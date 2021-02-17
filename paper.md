---
title: 'IDCeMPy: Python Package for Inflated Discrete Choice Models'

authors:
- affiliation: 1
  name: Nguyen K. Huynh
- affiliation: 2
  name: Sergio Béjar
  orcid: 0000-0002-9352-3892
- affiliation: 1
  name: Vineeta Yadav
- affiliation: 1
  name: Bumba Mukherjee
date: "16 February 2021"
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
Inflated discrete choice models have been recently developed to
address category inflation in ordered and unordered 
polytomous choice variables [@harris2007zero; @bagozzi2012mixture; @bagozzi2017distinguishing].
`IDCeMPy` is a comprehensive Python package that allows users 
to easily fit three distinct sets of inflated discrete choice 
models: Zero-Inflated Ordered Probit (ZIOP), Middle-Inflated Ordered Probit (MIOP), 
and Generalized Inflated Multinomial Logit (GIMNL) models. While the ZIOP model
permits careful evaluation of zero-inflated ordered choice outcomes
that results from two  data generating processes (hereafter, “d.g.p’s”),
the MIOP model accounts for ordered choice outcomes in which the 
inflated middle-category emerges from distinct d.g.p’s. 
The GIMNL models account for the large proportion—and heterogeneous 
mixture—of observations in the baseline and other unrodered outcome 
categories within MNL models that evaluate multiple
unordered polytomous outcomes. `IDCeMPy` thus provides users with 
versatile tools that can help them make more accurate inferences when working with inflated
ordered and unordered polytomous outcome variables. It can be 
applied to data in Economics, Engineering, Political Science, Psychology, and Public Health. 

# Statement of Need
Scholars and Data Scientists often use discrete choice models to evaluate ordered outcomes using the ordered 
probit model and unordered polytomous measures via the multinomial logit (MNL) estimator. These models, 
however, cannot account for the fact that in many ordered and unordered polytomous choice situations, 
an excessive share of observations—stemming from two distinct d.g.p’s—fall into a single category which 
is thus “inflated.” For instance, ordered outcome measures of self-reported smoking behavior that range 
from 0 for “no smoking” to 3 for “smoking 20 cigarettes or more daily” contain excessive observations 
in the zero (no smoking) category that includes two types of non-smokers: individuals who never smoke 
cigarettes, and those who smoked previously but temporarily stopped smoking because of their high price 
[@harris2007zero; @greene2015inflated].  

In ordered choice measures such as immigration attitudes, it is the middle-category of “indifference” 
that is inflated since it includes respondents who are truly indifferent about immigration and those 
who select “indifference” because of social desirability reasons 
[@bagozzi2012mixture; @brown2020modelling]. Further, in unordered polytomous variables of vote choice, 
for example, the baseline category is frequently inflated as it includes non-voters who abstain from 
voting in an election owing to temporary factors and routine non-voters who are disengaged from 
the political process [@campbell2008religion; @bagozzi2017distinguishing]. Failing to account for such category inflation in discrete choice measures leads to model misspecification, biased estimates, and incorrect inferences. 

@dale2018estimation's ZiOP STATA command fits the Zero-Inflated Ordered Probit without
correlated errors, while @xia2019gidm's gidm STATA command fits discrete choice models 
without correlated errors for inflated zero and other lower-category discrete outcomes. 
But there does not exist a comprehensive (e.g., Python) package that offers functions to 
estimate and assess the performance of inflated ordered probit models with correlated errors, 
the Middle-Inflated Ordered Probit, and inflated MNL models. 

# Package Architecture
`IDCeMPy` provides functions to fit the Zero-Inflated Ordered Probit (ZIOP) model without 
and with correlated errors (ZIOPC model), and the Middle-Inflated Ordered Probit (MIOP) model
without and with correlated errors (MIOPC). These models account for the inflated share of 
observations in either the zero or middle-category by combining a single binary “split-stage” 
probit equation with an ordered probit “outcome-stage” equation. Users can treat the error terms 
from these two equations as independent or correlated in the package’s estimation routines. 
`IDCeMPy` also includes functions to fit Generalized Inflated MNL models (GIMNL)—combining a logit split-stage equation,
and a MNL outcome-stage equation—to account for the preponderant and heterogeneous share of 
observations in the baseline or other outcome categories in unordered polytomous outcome measures. 
Combining two probability distributions by estimating two equations in each aforementioned 
model statistically addresses the inflated share of observations in an ordered or unordered 
choice category that results from distinct d.g.p’s. This ensures that each inflated discrete 
choice model in `IDCeMPy` avoids model misspecification and provides accurate estimates when 
evaluating inflated ordered and unordered polytomous dependent variables. `IDCeMPy`
also provides functions to assess each included model’s goodness-of-fit, extract marginal effects, 
and conduct tests for model comparison. The ZIOP(C) models can assess, for instance, zero-inflation 
in ordered outcome measures of self-reported smoking behavior [@harris2007zero], 
demand for health treatment [@greene2015inflated], and accident injury-severity 
outcomes [@fountas2018analysis]. The MIOP(C) models can address middle-category 
inflation in, for example, ordered measures like monetary policy [@brown2020modelling] and attitudes 
towards European Union (EU) membership [@bagozzi2012mixture]. The GIMNL models can 
evaluate inflated unordered polytomous outcome measures such as voter choice 
[@campbell2008religion] and consumer demand [@richards2018new].    

IDCeMPy contains the functions listed below to estimate the aforementioned inflated discrete choice models via MLE using Newton numerical optimization methods: 

* `opmod`; `iopmod`; `iopcmod`: Fits the ordered probit model, the Zero-Inflated (ZIOP) and Middle-Inflated ordered probit (MIOP) models without correlated errors, and the ZIOPC and MIOPC models that incorporate correlated errors.

* `opresults`; `iopresults`; `iopcresults`: Presents covariate estimates, Variance-Covariance (VCV) matrix, Log-Likelihood and AIC statistics of the object models.

* `iopfit`; `iopcfit`: Computes fitted probabilities from each estimated model’s objects.

* `vuong_opiop`; `vuong_opiopc`: Calculates Vuong test statistic for comparing the performance of the OP with the ZiOP(C) and MiOP(C) models.

* `split_effects`; `ordered_effects`: Estimates marginal effects of covariates in the split-stage and outcome-stage respectively. 

* `mnlmod`;`gimnlmod`: Fits MNL model and Generalized-Inflated MNL models.

* `mnlresults`;`gimnlresults`; `vuong_gimnl`: Presents covariate estimates, VCV matrix, Log-Likelihood and AIC statistics of `mnlmod`;`gimnlmod`. Vuong test statistic for comparing MNL to GIMNL models obtained from `vuong_gimnl`. 

The functions in IDCeMPy that fit the (i) ZIOP(C) models are presented using the ordered self-reported 
tobacco consumption dependent variable from the [2018 National Youth Tobacco Dataset](https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/index.htm), (ii) MIOP(C) models 
are illustrated using the ordered EU support outcome variable 
in @elgun2007exposure's data, and (iii) GIMNL models are evaluated using the unordered-polytomous Presidential vote choice dependent variable in @campbell2008religion's data. 
These three datasets are described in the package.

# Availability 
`IDCeMPy` is an Open-source software made available under the MIT license that can be installed from [PyPi](https://pypi.org/) or its [GitHub repository](https://github.com/hknd23/idcempy). 









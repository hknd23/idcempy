"""Classes and Functions for the biml module."""
import numpy as np
from numpy import *
# ZiOPC model converges extremely
# faster with import * rather than import as np.
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize


class BimnlModel:
    """Store model results from :py:func:`imnlmod`."""

    def __init__(self, modeltype, reference, inflatecat, llik,
                 coef, aic, vcov, data, xs, zs,
                 x_, yx_, z_, ycatu, xstr, ystr, zstr):
        """Store model results, goodness-of-fit tests, and other information.

        :param modeltype: Type of IMNL Model (bimnl3).
        :param reference: Order of categories. The order category will be
        the first element.
        :param llik: Log-Likelihood.
        :param coef: Model coefficients.
        :param aic: Model Akaike information .
        :param vcov: Variance-Covariance matrix.
            (optimized as inverted Hessian matrix)
        :param data: Full dataset.
        :param zs: Inflation stage estimates (Gammas).
        :param xs: Ordered probit estimates (Betas).
        :param ycatu: Number of categories in the Dependent Variable (DV).
        :param x_: X Data.
        :param yx_: Y (DV) data.
        :param z_: Z Data.
        :param xstr: list of string for x names.
        :param ystr: list of string for y names.
        :param zstr: list of string for z names.

        """
        self.modeltype = modeltype
        self.reference = reference
        self.inflatecat = inflatecat
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.inflatecat = zs
        self.multinom = xs
        self.ycat = ycatu
        self.X = x_
        self.Y = yx_
        self.Z = z_
        self.xstr = xstr
        self.ystr = ystr
        self.zstr = zstr


def bimnl3(pstart, x2, x3, y, z, reference):
    """
    Likelihood function for the baseline inflated three-category MNL model.

    :param pstart: starting parameters.
    :param x2: X covariates.
    :param x3: X covariates (should be identical to x2.
    :param y: Dependent Variable (DV).
    :param z: Inflation stage covariates.
    :param reference: order of categories (first category-baseline is inflated).
    """
    b2 = pstart[len(z.columns):(len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)):(len(pstart))]
    gamma = pstart[0:(len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = sum(np.log((1 - pz) + pz * p1) * (y == reference[0]) +
              np.log(pz * p2) * (y == reference[1]) +
              np.log(pz * p3) * (y == reference[2]))
    llik = -1 * sum(lik)
    return llik


def simnl3(pstart, x2, x3, y, z, reference):
    """
    Likelihood function for the second category inflated MNL model.

    :param pstart: starting parameters.
    :param x2: X covariates.
    :param x3: X covariates (should be identical to x2.
    :param y: Dependent Variable.
    :param z: Inflation stage covariates.
    :param reference: order of categories (second category is inflated).
    """
    b2 = pstart[len(z.columns):(len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)):(len(pstart))]
    gamma = pstart[0:(len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = sum(np.log(pz * p1) * (y == reference[0]) +
              np.log((1 - pz) + pz * p2) * (y == reference[1]) +
              np.log(pz * p3) * (y == reference[2]))
    llik = -1 * sum(lik)
    return llik


def timnl3(pstart, x2, x3, y, z, reference):
    """
    Likelihood function for the third category inflated MNL model.

    :param pstart: starting parameters.
    :param x2: X covariates.
    :param x3: X covariates (should be identical to x2.
    :param y: Dependent Variable (DV).
    :param z: Inflation stage covariates.
    :param reference: order of categories (third category is inflated).
    """
    b2 = pstart[len(z.columns):(len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)):(len(pstart))]
    gamma = pstart[0:(len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = sum(np.log(pz * p1) * (y == reference[0]) +
              np.log(pz * p2) * (y == reference[1]) +
              np.log((1 - pz) + pz * p3) * (y == reference[2]))
    llik = -1 * sum(lik)
    return llik


def imnlresults(model, data, x, y, z, modeltype, reference, inflatecat):
    """
    Produce estimation results, part of :py:func:`imnlmod`

    :param model: object model estimated.
    :param data: dataset.
    :param x: Multinomial Logit stage covariates.
    :param y: Dependent Variable (DV).
    :param z: Spplit-stage covariates.
    :param modeltype: type of inflated MNL model.
    :param reference: order of categories.
    :param inflatecat: inflated category.
    """
    varlist = np.unique(y + z + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how='any')
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    z_ = datasetnew[z]
    z_.insert(0, 'int', np.repeat(1, len(z_)))
    x_.insert(0, 'int', np.repeat(1, len(x_)))
    names = list()
    if modeltype == 'bimnl3':
        x2 = x_
        x3 = x_
        for s in range(z_.shape[1]):
            names.append("Inflation: " + z_.columns[s])
        for s in range(x2.shape[1]):
            names.append(str(reference[1]) + ": " + x2.columns[s])
        for s in range(x3.shape[1]):
            names.append(str(reference[2]) + ": " + x3.columns[s])
        xs = model.x[(z_.shape[1]):(z_.shape[1] + x2.shape[1] + x3.shape[1])]
    zs = model.x[0:(z_.shape[1])]
    ses = np.sqrt(np.diag(model.hess_inv))
    tscore = model.x / ses
    pval = (1 - (norm.cdf(abs(tscore)))) * 2
    lci = model.x - 1.96 * ses
    uci = model.x + 1.96 * ses
    coef = pd.DataFrame({'Coef': model.x, 'SE': ses, 'tscore': tscore,
                         'p': pval, '2.5%': lci, '97.5%': uci}, names)
    aic = -2 * (-model.fun) + 2 * (len(coef))
    llik = -1 * model.fun
    model = BimnlModel(modeltype, reference, inflatecat, llik, coef, aic,
                       model.hess_inv, datasetnew,
                       xs, zs, x_, yx_, z_,
                       yncat, x, y, z)
    return model


def imnlmod(data, x, y, z, reference, inflatecat,
            method='BFGS', pstart=None):
    """
    Estimate inflated Multinomial Logit model.

    :param data: dataset.
    :param x: MNL stage covariates.
    :param y: Dependent Variable. Variable needs to be in factor form,
    with a number from 0-2 representing each category.
    :param z: Inflation stage covariates.
    :param reference: order of categories.
    :param inflatecat: inflated category.
    :param method: Optimization method.  Default is 'BFGS'
    :param pstart: Starting parameters. Number of parameter n = 
    """
    varlist = np.unique(y + z + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how='any')
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    if yncat == 3:
        modeltype = 'bimnl3'
    else:
        raise Exception("Function only supports Dependent Variable with 3 "
                        "categories.")
    z_ = datasetnew[z]
    z_.insert(0, 'int', np.repeat(1, len(z_)))
    x_.insert(0, 'int', np.repeat(1, len(x_)))
    if modeltype == 'bimnl3':
        x2 = x_
        x3 = x_
        if pstart is None:
            pstart = np.repeat(.01, (len(x2.columns) + len(x3.columns)
                                     + len(z_.columns)))
        if inflatecat == "baseline":
            model = minimize(bimnl3, pstart,
                             args=(x2, x3, yx_, z_, reference),
                             method=method,
                             options={'gtol': 1e-6,
                                      'disp': True, 'maxiter': 500})
        elif inflatecat == "second":
            model = minimize(simnl3, pstart,
                             args=(x2, x3, yx_, z_, reference),
                             method=method,
                             options={'gtol': 1e-6,
                                      'disp': True, 'maxiter': 500})
        elif inflatecat == "third":
            model = minimize(timnl3, pstart,
                             args=(x2, x3, yx_, z_, reference),
                             method=method,
                             options={'gtol': 1e-6,
                                      'disp': True, 'maxiter': 500})
    results = imnlresults(model, data, x, y, z, modeltype, reference, inflatecat)
    return results

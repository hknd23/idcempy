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

    def __init__(self, modeltype, order, inflatecat, llik,
                 coef, aic, vcov, data, xs, zs,
                 x_, yx_, z_, ycatu, xstr, ystr, zstr):
        """Store model results, goodness-of-fit tests, and other information.

        :param modeltype: Type of IMNL Model (bimnl3)
        :param order: Order of categories. The order category will be
        the first element
        :param llik: Log-Likelihood
        :param coef: Model coefficients
        :param aic: Model Akaike information criterion
        :param vcov: Variance-Covariance matrix
            (optimized as inverted Hessian matrix)
        :param data: Full dataset
        :param zs: Inflation stage estimates (Gammas)
        :param xs: Ordered probit estimates (Betas)
        :param ycatu: Number of DV categories
        :param x_: X Data
        :param yx_: Y (DV) data
        :param z_: Z Data
        :param xstr: list of string for x names
        :param ystr: list of string for y names
        :param zstr: list of string for z names

        """
        self.modeltype = modeltype
        self.order = order
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


def bimnl3(pstart, x2, x3, y, z, order):
    """
    Likelihood function for the three-category BIMNL model.

    :param pstart: starting parameters
    :param x2: X covariates
    :param x3: X covariates (should be identical to x2
    :param y: Dependent Variable
    :param z: Inflation stage covariates
    :param order: order of categories
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
    lik = sum(np.log((1 - pz) + pz * p1) * (y == order[0]) +
              np.log(pz * p2) * (y == order[1]) +
              np.log(pz * p3) * (y == order[2]))
    llik = -1 * sum(lik)
    return llik


def simnl3(pstart, x2, x3, y, z, order):
    """
    Likelihood function for the three-category BIMNL model.

    :param pstart: starting parameters
    :param x2: X covariates
    :param x3: X covariates (should be identical to x2
    :param y: Dependent Variable
    :param z: Inflation stage covariates
    :param order: order of categories
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
    lik = sum(np.log(pz * p1) * (y == order[0]) +
              np.log((1 - pz) + pz * p2) * (y == order[1]) +
              np.log(pz * p3) * (y == order[2]))
    llik = -1 * sum(lik)
    return llik


def timnl3(pstart, x2, x3, y, z, order):
    """
    Likelihood function for the three-category BIMNL model.

    :param pstart: starting parameters
    :param x2: X covariates
    :param x3: X covariates (should be identical to x2
    :param y: Dependent Variable
    :param z: Inflation stage covariates
    :param order: order of categories
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
    lik = sum(np.log(pz * p1) * (y == order[0]) +
              np.log(pz * p2) * (y == order[1]) +
              np.log((1 - pz) + pz * p3) * (y == order[2]))
    llik = -1 * sum(lik)
    return llik


def imnlresults(model, data, x, y, z, modeltype, order, inflatecat):
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
            names.append("Z " + z_.columns[s])
        for s in range(x2.shape[1]):
            names.append("x2 " + x2.columns[s])
        for s in range(x3.shape[1]):
            names.append("x3 " + x3.columns[s])
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
    model = BimnlModel(modeltype, order, inflatecat, llik, coef, aic,
                       model.hess_inv, datasetnew,
                       xs, zs, x_, yx_, z_,
                       yncat, x, y, z)
    return model


def imnlmod(data, x, y, z, order, inflatecat,
            method='BFGS', pstart=None):
    """
    Estimate inflatecatd Multinomial Logit model.

    :param data: dataset
    :param x: MNL stage covariates
    :param y: Dependent Variable. Variable needs to be in factor form,
    with a number from 0-2 representing each category
    :param z: Inflation stage covariates
    
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
        print("Function only supports 3 categories.")
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
                             args=(x2, x3, yx_, z_, order),
                             method=method,
                             options={'gtol': 1e-6,
                                      'disp': True, 'maxiter': 500})
        elif inflatecat == "second":
            model = minimize(simnl3, pstart,
                             args=(x2, x3, yx_, z_, order),
                             method=method,
                             options={'gtol': 1e-6,
                                      'disp': True, 'maxiter': 500})
        elif inflatecat == "third":
            model = minimize(timnl3, pstart,
                             args=(x2, x3, yx_, z_, order),
                             method=method,
                             options={'gtol': 1e-6,
                                      'disp': True, 'maxiter': 500})
    results = imnlresults(model, data, x, y, z, modeltype, order, inflatecat)
    return results

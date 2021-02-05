"""Classes and Functions for the biml module."""
import numpy as np
from numpy import *
# ZiOPC model converges extremely
# faster with import * rather than import as np.
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize


class BimnlModel:
    """Store model results from :py:func:`bimnlmod`."""

    def __init__(self, modeltype, baseline, llik,
                 coef, aic, vcov, data, xs, zs,
                 x_, yx_, z_, ycatu, xstr, ystr, zstr):
        """Store model results, goodness-of-fit tests, and other information.

        :param modeltype: Type of Iop Model (bimnl3 or bimnl4)
        :param baseline: Baseline Category
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
        self.baseline = baseline
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.inflate = zs
        self.multinom = xs
        self.ycat = ycatu
        self.X = x_
        self.Y = yx_
        self.Z = z_
        self.xstr = xstr
        self.ystr = ystr
        self.zstr = zstr


def bimnl3(pstart, x2, x3, y, z, baseline, data):
    n = len(data)
    ycat = y.astype('category')
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
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
    lik = (sum(np.log((1 - pz) + pz * p1) * (y == baseline[0])) +
           sum(np.log(pz * p2) * (y == baseline[1])) +
           sum(np.log(pz * p3) * (y == baseline[2])))
    llik = -1 * sum(lik)
    return llik


def bimnl4(pstart, x2, x3, x4, y, z, baseline, data):
    n = len(data)
    ycat = y.astype('category')
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    b2 = pstart[len(z.columns):(len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)):
                (len(z.columns) + len(x2.columns) + len(x3.columns))]
    b4 = pstart[(len(z.columns) + len(x2.columns) + len(x3.columns)):
                (len(pstart))]
    gamma = pstart[0:(len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    xb4 = x4.dot(b4)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3) + np.exp(xb4))
    p2 = p1 * exp(xb2)
    p3 = p1 * exp(xb3)
    p4 = p1 * exp(xb4)
    lik = (sum(np.log((1 - pz) + pz * p1) * (y == baseline[0])) +
           sum(np.log(pz * p2) * (y == baseline[1])) +
           sum(np.log(pz * p3) * (y == baseline[2])) +
           sum(np.log(pz * p4) * (y == baseline[3])))
    llik = -1 * sum(lik)
    return llik


def bimnlresults(model, data, x, y, z, modeltype):  # CHECK AND EDITS
    """Produce estimation results, part of :py:func:`iopmod`.

    :param model: model object created from minimization
    :param data: dataset
    :param x: Ordered stage variables
    :param y: : DV
    :param z: : Inflation stage variables
    :param modeltype: : ZiOP or MiOP model
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
    for s in range(z_.shape[1]):
        names.append("Z " + z_.columns[s])
    for s in range(x_.shape[1]):
        names.append("X " + x_.columns[s])
    zs = model.x[yncat - 1:(yncat + z_.shape[1] - 1)]
    xs = model.x[(yncat + z_.shape[1] - 1):(
            yncat + z_.shape[1] + x_.shape[1] - 1)]
    ses = np.sqrt(np.diag(model.hess_inv))
    tscore = model.x / ses
    pval = (1 - (norm.cdf(abs(tscore)))) * 2
    lci = model.x - 1.96 * ses
    uci = model.x + 1.96 * ses
    coef = pd.DataFrame({'Coef': model.x, 'SE': ses, 'tscore': tscore,
                         'p': pval, '2.5%': lci, '97.5%': uci}, names)
    aic = -2 * (-model.fun) + 2 * (len(coef))
    llik = -1 * model.fun
    results = BimnlModel(modeltype, llik, coef, aic, model.hess_inv, datasetnew,
                         xs, zs, x_, yx_, z_, yncat, x, y, z)
    return results


def bimnlmod(data, x, y, z, baseline,
             method='BFGS', pstart=None):
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
    elif yncat == 4:
        modeltype = 'bimnl4'
    else:
        print("Function only supports 3 or 4 categories.")
    z_ = datasetnew[z]
    z_.insert(0, 'int', np.repeat(1, len(z_)))
    x_.insert(0, 'int', np.repeat(1, len(x_)))
    if modeltype == 'bimnl3':
        x2 = x_
        x3 = x_
        if pstart is None:
            pstart = np.repeat(.01, (len(x2.columns) + len(x3.columns)
                                     + len(z_.columns)))
        model = minimize(bimnl3, pstart,
                         args=(x2, x3, yx_, z_, baseline, datasetnew),
                         method=method,
                         options={'gtol': 1e-6,
                                  'disp': True, 'maxiter': 500})
    elif modeltype == 'bimnl4':
        x2 = x_
        x3 = x_
        x4 = x_
        if pstart is None:
            pstart = np.repeat(.01, (len(x2.columns) + len(x3.columns)
                                     + len(x4.columns) + len(z_.columns)))
        model = minimize(bimnl4, pstart,
                         args=(x2, x3, x4, yx_, z_, baseline, datasetnew),
                         method=method,
                         options={'gtol': 1e-6,
                                  'disp': True, 'maxiter': 500})
    results = bimnlresults(model, data, x, y, z, modeltype, baseline)
    return results


def bimnlresults(model, data, x, y, z, modeltype, baseline):
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
    elif modeltype == 'bimnl4':
        x2 = x_
        x3 = x_
        x4 = x_
        for s in range(z_.shape[1]):
            names.append("Z " + z_.columns[s])
        for s in range(x2.shape[1]):
            names.append("x2 " + x2.columns[s])
        for s in range(x3.shape[1]):
            names.append("x3 " + x3.columns[s])
        for s in range(x4.shape[1]):
            names.append("x4 " + x4.columns[s])
        xs = model.x[(z_.shape[1]):(z_.shape[1] + x2.shape[1] +
                                    x3.shape[1] + x4.shape[1])]
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
    model = BimnlModel(modeltype, baseline, llik, coef, aic, model.hess_inv,
                       datasetnew, xs, zs, x_, yx_, z_, yncat, x, y, z)
    return model

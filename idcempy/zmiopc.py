"""Classes and Functions for the ziopcpy module."""
import numpy as np

# ZiOPC model converges extremely
# faster with import * rather than import as np.
import pandas as pd
from numpy import *
from scipy.optimize import minimize
from scipy.stats import mvn, norm


class OpModel:
    """Store model results from :py:func:`opmod`."""

    def __init__(self, llik, coef, aic, vcov, data, xs, ts, x_, yx_, yncat,
                 xstr, ystr):
        """Store model results, goodness-of-fit tests, and other information.

        :param llik: Log-Likelihood.
        :param coef: Model coefficients.
        :param aic: Model Akaike information criterion.
        :param vcov: Variance-Covariance matrix.
            (optimized as inverted Hessian matrix)
        :param data: Full dataset.
        :param ts: Cutpoints for ordered probit.
        :param xs: Ordered probit estimates (Betas).
        :param yncat: Number of categories in the outcome variable.
        :param x_: X (Covariates) Data.
        :param yx_: Y (Dependent Variable) data.
        :param xstr: list of strings for variable(s) in the outcome stage (x).
        :param ystr: list of strings for outcome variable name (y).
        """
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.cutpoints = ts
        self.ordered = xs
        self.ycat = yncat
        self.X = x_
        self.Y = yx_
        self.xstr = xstr
        self.ystr = ystr


class IopModel:
    """Store model results from :py:func:`iopmod`."""

    def __init__(
            self,
            modeltype,
            llik,
            coef,
            aic,
            vcov,
            data,
            xs,
            zs,
            ts,
            x_,
            yx_,
            z_,
            yncat,
            xstr,
            ystr,
            zstr,
    ):
        """Store model results, goodness-of-fit tests, and other information.

        :param modeltype: Type of Iop Model (ziop or miop).
        :param llik: Log-Likelihood.
        :param coef: Model coefficients
        :param aic: Model Akaike information criterion.
        :param vcov: Variance-Covariance matrix.
            (optimized as inverted Hessian matrix)
        :param data: Full dataset.
        :param ts: Cutpoints for ordered probit.
        :param zs: Inflation stage estimates (Gammas).
        :param xs: Ordered probit estimates (Betas).
        :param yncat: Number of categories in the Dependent Variable (DV).
        :param x_: X Data.
        :param yx_: Y (DV) data.
        :param z_: Z Data.
        :param xstr: list of strings for x names.
        :param ystr: list of strings for y names.
        :param zstr: list of strings for z names.

        """
        self.modeltype = modeltype
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.cutpoints = ts
        self.inflate = zs
        self.ordered = xs
        self.ycat = yncat
        self.X = x_
        self.Y = yx_
        self.Z = z_
        self.xstr = xstr
        self.ystr = ystr
        self.zstr = zstr


class IopCModel:
    """Store model results from :py:func:`iopcmod`."""

    def __init__(
            self,
            modeltype,
            llik,
            coef,
            aic,
            vcov,
            data,
            xs,
            zs,
            ts,
            x_,
            yx_,
            z_,
            rho,
            yncat,
            xstr,
            ystr,
            zstr,
    ):
        """Store model results, goodness-of-fit tests, and other information.

        :param modeltype: Type of IopC Model (ziopc or miopc).
        :param llik: Log-Likelihood.
        :param coef: Model coefficients.
        :param AIC: Model Akaike information criterion.
        :param vcov: Variance-Covariance matrix.
            (optimized as inverted Hessian matrix)
        :param data: Full dataset.
        :param ts: Cutpoints for ordered probit.
        :param zs: Inflation stage estimates (Gammas).
        :param xs: Ordered probit estimates (Betas).
        :param rho: Rho.
        :param yncat: Number of categories in Dependent Variable (DV).
        :param x_: X Data.
        :param yx_: Y (DV) data.
        :param z_: Z Data.
        :param xstr: list of strings for x names.
        :param ystr: list of strings for y names.
        :param zstr: list of strings for z names.
        """
        self.modeltype = modeltype
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.cutpoints = ts
        self.inflate = zs
        self.ordered = xs
        self.ycat = yncat
        self.X = x_
        self.Y = yx_
        self.Z = z_
        self.r = rho
        self.xstr = xstr
        self.ystr = ystr
        self.zstr = zstr


class FittedVals:
    """Store fitted values for iOP models."""

    def __init__(self, responsefull, responseordered, responseinflation,
                 linear):
        """Store different type of equation in each attribute.

        :param responsefull: responsefull
        :param responseordered: responseordered
        :param responseinflation: responseinflation
        :param linear: linear
        """
        self.responsefull = responsefull
        self.responseordered = responseordered
        self.responseinflation = responseinflation
        self.linear = linear


def op(pstart, x, y, data, weights, offsetx):
    """Calculate likelihood function for Ordered Probit Model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x: Ordered stage variables.
    :type x: pandas dataframe
    :param y: DV.
    :type y: pandas dataframe
    :param data: dataset.
    :type data: pandas dataframe
    :param weights: weights.
    :type weights: int
    :param offsetx: offset for X.
    :type offsetx: int
    """
    n = len(data)
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((len(data), yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    tau = np.repeat(1.0, yncat)
    for j in range(yncat - 1):
        if j == 0:
            tau[j] = pstart[j]
        else:
            tau[j] = tau[j - 1] + np.exp(pstart[j])
    beta = pstart[(yncat - 1): len(pstart)]
    xb = x.dot(beta) + offsetx
    cprobs = np.zeros((n, yncat))
    probs = np.zeros((n, yncat))
    for i in range(yncat - 1):
        cprobs[:, i] = norm.cdf(tau[i] - xb)
    probs[:, 0] = cprobs[:, 0]
    for i in range(1, yncat - 1):
        probs[:, i] = cprobs[:, i] - cprobs[:, (i - 1)]
    probs[:, yncat - 1] = 1 - cprobs[:, (yncat - 2)]
    lik = np.zeros((n, yncat))
    for k in range(n):
        for j in range(yncat):
            lik[k, j] = v[k, j] * probs[k, j]
    likk = np.log(lik[lik != 0])
    llik = -1 * sum(likk * weights)
    return llik


def ziop(pstart, x, y, z, data, weights, offsetx, offsetz):
    """Calculate likelihood function for Zero-inflated Model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x: Ordered stage variables. Data subsetted to selected
        variables.
    :type x: pandas dataframe
    :param y: Dependent Variable (DV). Data subsetted to selected
        variables.
    :type y: pandas dataframe
    :param z: Inflation stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param data: Dataset with missing values listwise deleted.
    :type data: pandas dataframe
    :param weights: weights.
    :type weights: float
    :param offsetx: offset for X.
    :type offsetx: float
    :param offsetz: offset for z.
    :type offsetz: float
    """
    n = len(data)
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((len(data), yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    tau = np.repeat(1.0, yncat)
    for j in range(yncat - 1):
        if j == 0:
            tau[j] = pstart[j]
        else:
            tau[j] = tau[j - 1] + np.exp(pstart[j])
    beta = pstart[(yncat + len(z.columns) - 1): len(pstart)]
    gamma = pstart[(yncat - 1): (yncat + len(z.columns) - 1)]
    zg = z.dot(gamma) + offsetz
    xb = x.dot(beta) + offsetx
    cprobs = np.zeros((n, yncat))
    probs = np.zeros((n, yncat))
    for i in range(yncat - 1):
        cprobs[:, i] = norm.cdf(tau[i] - xb)
    probs[:, 0] = (cprobs[:, 0]) * norm.cdf(zg) + (1 - norm.cdf(zg))
    for i in range(1, yncat - 1):
        probs[:, i] = (cprobs[:, i] - cprobs[:, (i - 1)]) * norm.cdf(zg)
    probs[:, yncat - 1] = (1 - cprobs[:, (yncat - 2)]) * norm.cdf(zg)
    lik = np.zeros((n, yncat))
    for k in range(n):
        for j in range(yncat):
            lik[k, j] = v[k, j] * probs[k, j]
    likk = np.log(lik[lik != 0])
    llik = -1 * sum(likk * weights)
    return llik


def ziopc(pstart, x, y, z, data, weights, offsetx, offsetz):
    """Calculate likelihood function for Zero-inflated Correlated-Errors Model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x: Ordered stage variables. Data subsetted to selected
        variables.
    :type x: pandas dataframe
    :param y: Dependent Variable (DV). Data subsetted to selected
        variables.
    :type y: pandas dataframe
    :param z: Inflation stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param data: Dataset with missing values listwise deleted.
    :type data: pandas dataframe
    :param weights: weights.
    :type weights: float
    :param offsetx: offset for X.
    :type offsetx: float
    :param offsetz: offset for z.
    :type offsetz: float
    """
    n = len(data)
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((len(data), yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    tau = np.repeat(1.0, yncat)
    for j in range(yncat - 1):
        if j == 0:
            tau[j] = pstart[j]
        else:
            tau[j] = tau[j - 1] + np.exp(pstart[j])
    beta = pstart[(yncat + len(z.columns) - 1): len(pstart) - 1]
    gamma = pstart[(yncat - 1): (yncat + len(z.columns) - 1)]
    x_beta = x.dot(beta)
    # correlation
    rho = pstart[len(pstart) - 1]
    cprobs = np.zeros((len(x_beta), yncat))
    probs = np.zeros((len(x_beta), yncat))
    cut = np.zeros((len(x_beta), yncat))
    cutb = np.zeros((len(x_beta), yncat))
    zg = z.dot(gamma) + offsetz
    xb = x.dot(beta) + offsetx
    means = np.array([0, 0])
    lower = np.array([-inf, -inf])
    sigma = np.array([[1, rho], [rho, 1]])
    nsigma = np.array([[1, -rho], [-rho, 1]])
    for i in range(yncat - 1):
        cprobs[:, i] = norm.cdf(tau[i] - xb)
        cut[:, i] = tau[i] - xb
        cutb[:, i] = xb - tau[i]
    upperb = np.zeros((len(x_beta), 2))
    upper = np.zeros((len(x_beta), 2))
    for j in range(n):
        upperb[j, :] = [zg[j], cutb[j, yncat - 2]]
        upper[j, :] = [zg[j], cut[j, 0]]
        probs[j, yncat - 1] = mvn.mvnun(lower, upperb[j], means, sigma)[0]
        probs[j, 0] = (1 - norm.cdf(zg[j])) + mvn.mvnun(lower, upper[j],
                                                        means, nsigma)[0]
    for j in range(n):
        for i in range(1, yncat - 1):
            probs[j, i] = (
                    mvn.mvnun(lower, [zg[j],
                                      cut[j, i]], means, nsigma)[0]
                    - mvn.mvnun(lower, [zg[j],
                                        cut[j, i - 1]], means, nsigma)[0])
    lik = np.zeros((n, yncat))
    for k in range(n):
        for j in range(yncat):
            lik[k, j] = v[k, j] * probs[k, j]
    likk = np.log(lik[lik != 0])
    llik = -1 * sum(likk * weights)
    return llik


def miop(pstart, x, y, z, data, weights, offsetx, offsetz):
    """
    Likelihood function for Middle-inflated Ordered Probit Model
    "without" correlated errors.
    Number of outcomes must be odd.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x: Ordered stage variables. Data subsetted to selected
        variables.
    :type x: pandas dataframe
    :param y: Dependent Variable (DV). Data subsetted to selected
        variables.
    :type y: pandas dataframe
    :param z: Inflation stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param data: Dataset with missing values listwise deleted.
    :type data: pandas dataframe
    :param weights: weights.
    :type weights: float
    :param offsetx: offset for X.
    :type offsetx: float
    :param offsetz: offset for z.
    :type offsetz: float
    """
    n = len(data)
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((len(data), yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    tau = np.repeat(1.0, yncat)
    for j in range(yncat - 1):
        if j == 0:
            tau[j] = pstart[j]
        else:
            tau[j] = tau[j - 1] + np.exp(pstart[j])
    beta = pstart[(yncat + len(z.columns) - 1): len(pstart)]
    gamma = pstart[(yncat - 1): (yncat + len(z.columns) - 1)]
    zg = z.dot(gamma) + offsetz
    xb = x.dot(beta) + offsetx
    cprobs = np.zeros((n, yncat))
    probs = np.zeros((n, yncat))
    for i in range(yncat - 1):
        cprobs[:, i] = norm.cdf(tau[i] - xb)
    probs[:, 0] = (cprobs[:, 0]) * norm.cdf(zg)
    for i in range(1, yncat - 1):
        if i == np.median(range(yncat)):
            probs[:, i] = (1 - norm.cdf(zg)) + (
                    norm.cdf(zg) * (cprobs[:, i] - cprobs[:, (i - 1)])
            )
        else:
            probs[:, i] = norm.cdf(zg) * (cprobs[:, i] - cprobs[:, (i - 1)])
    probs[:, yncat - 1] = (1 - cprobs[:, (yncat - 2)]) * norm.cdf(zg)
    lik = np.zeros((n, yncat))
    for k in range(n):
        for j in range(yncat):
            lik[k, j] = v[k, j] * probs[k, j]
    likk = np.log(lik[lik != 0])
    llik = -1 * sum(likk * weights)
    return llik


def miopc(pstart, x, y, z, data, weights, offsetx, offsetz):
    """
    Likelihood function for Middle-inflated Correlated-Errors Model.
    Number of outcomes must be odd.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x: Ordered stage variables. Data subsetted to selected
        variables.
    :type x: pandas dataframe
    :param y: Dependent Variable (DV). Data subsetted to selected
        variables.
    :type y: pandas dataframe
    :param z: Inflation stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param data: Dataset with missing values listwise deleted.
    :type data: pandas dataframe
    :param weights: weights.
    :type weights: float
    :param offsetx: offset for X.
    :type offsetx: float
    :param offsetz: offset for z.
    :type offsetz: float
    """
    n = len(data)
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((len(data), yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    tau = np.repeat(1.0, yncat)
    for i in range(yncat - 1):
        if i == 0:
            tau[i] = pstart[i]
        else:
            tau[i] = tau[i - 1] + np.exp(pstart[i])
    beta = pstart[(yncat + len(z.columns) - 1): len(pstart) - 1]
    gamma = pstart[(yncat - 1): (yncat + len(z.columns) - 1)]
    x_beta = x.dot(beta)
    rho = pstart[len(pstart) - 1]
    cprobs = np.zeros((len(x_beta), yncat))
    probs = np.zeros((len(x_beta), yncat))
    cutpoint = np.zeros((len(x_beta), yncat))
    cutpointb = np.zeros((len(x_beta), yncat))
    zg = z.dot(gamma) + offsetz
    xb = x.dot(beta) + offsetx
    means = np.array([0, 0])
    lower = np.array([-inf, -inf])
    sigma = np.array([[1, rho], [rho, 1]])
    nsigma = np.array([[1, -rho], [-rho, 1]])
    for i in range(yncat - 1):
        cprobs[:, i] = norm.cdf(tau[i] - xb)
        cutpoint[:, i] = tau[i] - xb
        cutpointb[:, i] = xb - tau[i]
    upperb = np.zeros((len(x_beta), 2))
    upper = np.zeros((len(x_beta), 2))
    for j in range(n):
        upperb[j, :] = [zg[j], cutpointb[j, yncat - 2]]
        upper[j, :] = [zg[j], cutpoint[j, 0]]
        probs[j, yncat - 1] = mvn.mvnun(lower, upperb[j], means, sigma)[0]
        probs[j, 0] = mvn.mvnun(lower, upper[j], means, nsigma)[0]
    for i in range(n):
        for j in range(1, yncat - 1):
            if j == np.median(range(yncat)):
                probs[i, j] = (
                        (1 - norm.cdf(zg[i]))
                        + mvn.mvnun(lower, [zg[i], cutpoint[i, j]], means,
                                    nsigma)[0]
                        - mvn.mvnun(lower, [zg[i], cutpoint[i, j - 1]], means,
                                    nsigma)[0]
                )
            else:
                probs[i, j] = (
                        mvn.mvnun(lower, [zg[i], cutpoint[i, j]], means,
                                  nsigma)[0]
                        - mvn.mvnun(lower, [zg[i], cutpoint[i, j - 1]], means,
                                    nsigma)[0]
                )
    lik = np.zeros((n, yncat))
    for k in range(n):
        for j in range(yncat):
            lik[k, j] = v[k, j] * probs[k, j]
    likk = np.log(lik[lik != 0])
    llik = -1 * sum(likk * weights)
    return llik


def opresults(model, data, x, y):
    """Produce estimation results, part of :py:func:`opmod`.

    :param model: model object created from minimization.
    :param data: dataset.
    :param x: Independent variables.
    :param y: : Dependent Variable.
    """
    varlist = np.unique(y + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    names = list()
    for s in range(1, yncat):
        names.append("cut" + str(s))
    for s in range(x_.shape[1]):
        names.append(x_.columns[s])
    ts = model.x[0: yncat - 1]
    xs = model.x[(yncat - 1): (yncat + x_.shape[1] - 1)]
    ses = np.sqrt(np.diag(model.hess_inv))
    tscore = model.x / ses
    pval = (1 - (norm.cdf(abs(tscore)))) * 2
    lci = model.x - 1.96 * ses
    uci = model.x + 1.96 * ses
    coef = pd.DataFrame(
        {
            "Coef": model.x,
            "SE": ses,
            "tscore": tscore,
            "p": pval,
            "2.5%": lci,
            "97.5%": uci,
        },
        names,
    )
    aic = -2 * (-model.fun) + 2 * (len(coef))
    results = OpModel(
        model.fun, coef, aic, model.hess_inv, datasetnew, xs, ts, x_, yx_,
        yncat, x, y
    )
    return results


def opmod(data, x, y, pstart=None, method="BFGS", weights=1, offsetx=0):
    """Estimate Ordered Probit model and return :class:`OpModel` class object.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param data: full dataset.
    :type x: list of str
    :param y: Dependent Variable (DV).
    :type y: list of str
    :param method: method for optimization, default 'BFGS'. For other
        available methods, see scipy.optimize.minimize documentation.
    :param weights: weights.
    :param offsetx: offset for X.
    :return: OpModel
    """
    varlist = np.unique(y + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    if pstart is None:
        pstart = np.repeat(0.01, ((yncat - 1) + len(x_.columns)))
    model = minimize(
        op,
        pstart,
        args=(x_, yx_, datasetnew, weights, offsetx),
        method=method,
        options={"gtol": 1e-6, "disp": True, "maxiter": 500},
    )
    results = opresults(model, data, x, y)
    return results


def iopresults(model, data, x, y, z, modeltype):
    """Produce estimation results, part of :py:func:`iopmod`.

    :param model: model object created from minimization.
    :param data: dataset.
    :param x: Ordered stage variables.
    :param y: : Dependent Variable (DV).
    :param z: : Inflation stage variables.
    :param modeltype: : 'ziop' or 'miop' model.
    """
    varlist = np.unique(y + z + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    z_ = datasetnew[z]
    z_.insert(0, "int", np.repeat(1, len(z_)))
    names = list()
    for s in range(1, yncat):
        names.append("cut" + str(s))
    for s in range(z_.shape[1]):
        names.append("Inflation: " + z_.columns[s])
    for s in range(x_.shape[1]):
        names.append("Ordered: " + x_.columns[s])
    ts = model.x[0: yncat - 1]
    zs = model.x[yncat - 1: (yncat + z_.shape[1] - 1)]
    xs = model.x[
         (yncat + z_.shape[1] - 1): (yncat + z_.shape[1] + x_.shape[1] - 1)]
    ses = np.sqrt(np.diag(model.hess_inv))
    tscore = model.x / ses
    pval = (1 - (norm.cdf(abs(tscore)))) * 2
    lci = model.x - 1.96 * ses
    uci = model.x + 1.96 * ses
    coef = pd.DataFrame(
        {
            "Coef": model.x,
            "SE": ses,
            "tscore": tscore,
            "p": pval,
            "2.5%": lci,
            "97.5%": uci,
        },
        names,
    )
    aic = -2 * (-model.fun) + 2 * (len(coef))
    results = IopModel(
        modeltype,
        model.fun,
        coef,
        aic,
        model.hess_inv,
        datasetnew,
        xs,
        zs,
        ts,
        x_,
        yx_,
        z_,
        yncat,
        x,
        y,
        z,
    )
    return results


def iopcresults(model, data, x, y, z, modeltype):
    """Produce estimation results, part of :py:func:`ziopc  mod`.

    :param model: model object created from minimization.
    :param data: dataset.
    :param x: Ordered stage variables.
    :param y: : Dependent Variable (DV).
    :param z: : Inflation stage variables.
    :param modeltype: : Type of model. Options are: 'ziopc' or 'miopc'
    """
    varlist = np.unique(y + z + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    z_ = datasetnew[z]
    z_.insert(0, "int", np.repeat(1, len(z_)))
    names = list()
    for s in range(1, yncat):
        names.append("cut" + str(s))
    for s in range(z_.shape[1]):
        names.append("Inflation: " + z_.columns[s])
    for s in range(x_.shape[1]):
        names.append("Ordered: " + x_.columns[s])
    names.append("rho")
    ts = model.x[0: yncat - 1]
    zs = model.x[yncat - 1: (yncat + z_.shape[1] - 1)]
    xs = model.x[
         (yncat + z_.shape[1] - 1): (yncat + z_.shape[1] + x_.shape[1] - 1)]
    rho = model.x[-1]
    ses = np.sqrt(np.diag(model.hess_inv))
    tscore = model.x / ses
    pval = (1 - (norm.cdf(abs(tscore)))) * 2
    lci = model.x - 1.96 * ses
    uci = model.x + 1.96 * ses
    coef = pd.DataFrame(
        {
            "Coef": model.x,
            "SE": ses,
            "tscore": tscore,
            "p": pval,
            "2.5%": lci,
            "97.5%": uci,
        },
        names,
    )
    aic = -2 * (-model.fun) + 2 * (len(coef))
    results = IopCModel(
        modeltype,
        model.fun,
        coef,
        aic,
        model.hess_inv,
        datasetnew,
        xs,
        zs,
        ts,
        x_,
        yx_,
        z_,
        rho,
        yncat,
        x,
        y,
        z,
    )
    return results


def iopmod(
        modeltype,
        data,
        x,
        y,
        z,
        pstart=None,
        method="BFGS",
        weights=1,
        offsetx=0,
        offsetz=0,
):
    """Estimate ZiOP model and return :class:`IopModel` class object as output.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param data: full dataset.
    :type x: list of str.
    :param y: Dependent Variable (DV).
    :type y: list of str.
    :param z: Inflation stage variables.
    :type z: list of str.
    :param modeltype: must be one of "ziop" or 'miop'.
    :param method: method for optimization, default 'BFGS'. For other
        available methods, see scipy.optimize.minimize documentation.
    :param weights: weights.
    :param offsetx: offset for X.
    :param offsetz: offset for Z.
    :return: IopModel
    """
    types = ["ziop", "miop"]
    if modeltype in types:
        varlist = np.unique(y + z + x)
        dataset = data[varlist]
        datasetnew = dataset.dropna(how="any")
        x_ = datasetnew[x]
        y_ = datasetnew[y]
        yx_ = y_.iloc[:, 0]
        yncat = len(np.unique(yx_))
        z_ = datasetnew[z]
        z_.insert(0, "ones", np.repeat(1, len(z_)))
        if pstart is None:
            pstart = np.repeat(0.01, (
                    (yncat - 1) + len(x_.columns) + len(z_.columns)))
        if modeltype == "ziop":
            model = minimize(
                ziop,
                pstart,
                args=(x_, yx_, z_, datasetnew, weights, offsetx, offsetz),
                method=method,
                options={"gtol": 1e-6, "disp": True, "maxiter": 500},
            )
        elif modeltype == "miop":
            if len(np.unique(y_.astype("category").iloc[:, 0])) % 2 == 1:
                model = minimize(
                    miop,
                    pstart,
                    args=(x_, yx_, z_, datasetnew, weights, offsetx, offsetz),
                    method=method,
                    options={"gtol": 1e-6, "disp": True, "maxiter": 500},
                )
            else:
                raise Exception("miop requires odd number of categories.")
        results = iopresults(model, data, x, y, z, modeltype)
        return results
    else:
        raise Exception("type must be ziop or miop")


def iopcmod(
        modeltype,
        data,
        x,
        y,
        z,
        pstart=None,
        method="BFGS",
        weights=1,
        offsetx=0,
        offsetz=0,
):
    """Estimate an iOP model (ZiOP or MiOP) and return :class:`IopcModel`.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param data: dataset.
    :type x: list of str
    :param y: Dependent Variable (DV).
    :type y: list of str
    :param z: Inflation stage variables.
    :type z: list of str
    :param modeltype: Type of model to be estimated ("ziopc" or 'miopc').
    :param method: method for optimization, default 'BFGS'.  For other
        available methods, see scipy.optimize.minimize documentation.
    :param weights: weights.
    :param offsetx: offset for X.
    :param offsetz: offset for Z.
    :return: IopCModel
    """
    types = ["ziopc", "miopc"]
    if modeltype in types:
        varlist = np.unique(y + z + x)
        dataset = data[varlist]
        datasetnew = dataset.dropna(how="any")
        datasetnew = datasetnew.reset_index(drop=True)
        x_ = datasetnew[x]
        y_ = datasetnew[y]
        yx_ = y_.iloc[:, 0]
        yncat = len(np.unique(yx_))
        z_ = datasetnew[z]
        z_.insert(0, "ones", np.repeat(1, len(z_)))
        if pstart is None:
            pstart = np.repeat(
                0.01, ((yncat - 1) + len(x_.columns) + len(z_.columns) + 1)
            )
        if modeltype == "ziopc":
            model = minimize(
                ziopc,
                pstart,
                args=(x_, yx_, z_, datasetnew, weights, offsetx, offsetz),
                method=method,
                options={"gtol": 1e-6, "disp": True, "maxiter": 500},
            )
        elif modeltype == "miopc":
            if len(np.unique(y_.astype("category").iloc[:, 0])) % 2 == 1:
                model = minimize(
                    miopc,
                    pstart,
                    args=(x_, yx_, z_, datasetnew, weights, offsetx, offsetz),
                    method=method,
                    options={"gtol": 1e-6, "disp": True, "maxiter": 500},
                )
            else:
                raise Exception("miopc requires odd number of categories.")
                return
        results = iopcresults(model, data, x, y, z, modeltype)
        return results
    else:
        raise Exception("type must be ziopc or miopc")


def iopfit(model):
    """Calculate probabilities from :py:func:`iopmod`.

    :param model: :class:IopModel object from :py:func:`iopmod`.
    :return: :class:FittedVals object with fitted values.
    """
    zg = model.Z.dot(model.inflate)
    xb = model.X.dot(model.ordered)
    cprobs = np.zeros((model.ycat - 1, 1))
    n = len(model.data)
    probs = np.zeros((n, model.ycat))
    cprobs[0, 0] = model.cutpoints[0]
    if model.modeltype == "ziop":
        for j in range(1, model.ycat - 1):
            cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
        probs[:, model.ycat - 1] = (norm.cdf(zg)) * (
                1 - norm.cdf(cprobs[model.ycat - 2, 0] - xb)
        )
        probs[:, 0] = (1 - norm.cdf(zg)) + (norm.cdf(zg)) * (
            norm.cdf(cprobs[0, 0] - xb)
        )
        for j in range(1, model.ycat - 1):
            probs[:, j] = (norm.cdf(zg)) * ((norm.cdf(cprobs[j, 0] - xb))
                                            - (norm.cdf(cprobs[j - 1, 0]
                                                        - xb)))
    elif model.modeltype == "miop":
        for j in range(1, model.ycat - 1):
            cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
        probs[:, model.ycat - 1] = (norm.cdf(zg)) * (
                1 - norm.cdf(cprobs[model.ycat - 2, 0] - xb)
        )
        probs[:, 0] = norm.cdf(zg) * norm.cdf(cprobs[0, 0] - xb)

        for i in range(1, model.ycat - 1):
            if i == np.median(range(model.ycat)):
                probs[:, i] = (1 - norm.cdf(zg)) + (
                        norm.cdf(zg)
                        * (norm.cdf(cprobs[j, 0] - xb)
                           - norm.cdf(cprobs[j - 1, 0] - xb)))
            else:
                probs[:, i] = norm.cdf(zg) * (
                        norm.cdf(cprobs[:, i] - xb)
                        - norm.cdf(cprobs[j - 1, 0] - xb))
    probsordered = np.zeros((n, model.ycat))
    probsordered[:, model.ycat - 1] = 1 - norm.cdf(
        cprobs[model.ycat - 2, 0] - xb)
    probsordered[:, 0] = norm.cdf(cprobs[0, 0] - xb)
    for j in range(1, model.ycat - 1):
        probsordered[:, j] = (norm.cdf(cprobs[j, 0] - xb)) - (
            norm.cdf(cprobs[j - 1, 0] - xb)
        )
    probsinfl = np.zeros((n, 1))
    probsinfl[:, 0] = 1 - norm.cdf(zg)
    probslin = pd.DataFrame({"ZG": zg, "XB": xb})
    fitted = FittedVals(probs, probsordered, probsinfl, probslin)
    return fitted


def iopcfit(model):
    """Calculate fitted probabilities from :py:func:`iopcmod`.

    :param model: :class:`IopCModel` object from :py:func:`iopcmod`.
    :return: :class:`FittedVals` object with fitted values.
    """
    zg = model.Z.dot(model.inflate)
    xb = model.X.dot(model.ordered)
    cprobs = np.zeros((model.ycat - 1, 1))
    n = len(model.data)
    probs = np.zeros((n, model.ycat))
    cprobs[0, 0] = model.cutpoints[0]
    rho = model.coefs.iloc[-1, 0]
    means = np.array([0, 0])
    lower = np.array([-inf, -inf])
    sigma = np.array([[1, rho], [rho, 1]])
    nsigma = np.array([[1, -rho], [-rho, 1]])
    if model.modeltype == "ziopc":
        for j in range(1, model.ycat - 1):
            cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
        for i in range(n):
            probs[i, model.ycat - 1] = mvn.mvnun(
                lower, [zg[i], (xb[i] - cprobs[model.ycat - 2][0])], means,
                sigma
            )[0]
            probs[i, 0] = (1 - norm.cdf(zg[i])) + mvn.mvnun(
                lower, [zg[i], (cprobs[0][0] - xb[i])], means, nsigma
            )[0]
        for i in range(n):
            for j in range(1, model.ycat - 1):
                probs[i, j] = (
                                  mvn.mvnun(lower,
                                            [zg[i], (cprobs[j][0] - xb[i])],
                                            means, nsigma)[0]
                              ) - (
                                  mvn.mvnun(
                                      lower,
                                      [zg[i], (cprobs[j - 1][0] - xb[i])],
                                      means, nsigma
                                  )[0]
                              )
    elif model.modeltype == "miopc":
        for j in range(1, model.ycat - 1):
            cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
        for i in range(n):
            probs[i, model.ycat - 1] = mvn.mvnun(
                lower, [zg[i], (xb[i] - cprobs[model.ycat - 2][0])], means,
                sigma
            )[0]
            probs[i, 0] = mvn.mvnun(
                lower, [zg[i], (cprobs[0][0] - xb[i])], means, nsigma
            )[0]
        for i in range(n):
            for j in range(1, model.ycat - 1):
                if j == np.median(range(model.ycat)):
                    probs[i, j] = (1 - norm.cdf(zg[i])) + (
                            (mvn.mvnun(
                                lower, [zg[i], (cprobs[j][0] - xb[i])],
                                means, nsigma)[0])
                            - (mvn.mvnun(lower,
                                         [zg[i], (cprobs[j - 1][0] -
                                                  xb[i])],
                                         means, nsigma)[0]))
                else:
                    probs[i, j] = (mvn.mvnun(lower,
                                             [zg[i], (cprobs[j][0] - xb[i])],
                                             means, nsigma)[0]) - (
                                      mvn.mvnun(lower, [zg[i],
                                                        (cprobs[j - 1][0] -
                                                         xb[i])],
                                                means, nsigma)[0])

    # ordered
    probsordered = np.zeros((n, model.ycat))
    probsordered[:, model.ycat - 1] = 1 - norm.cdf(
        cprobs[model.ycat - 2, 0] - xb)
    probsordered[:, 0] = norm.cdf(cprobs[0, 0] - xb)
    for j in range(1, model.ycat - 1):
        probsordered[:, j] = norm.cdf(cprobs[j, 0] - xb) - (
            norm.cdf(cprobs[j - 1, 0] - xb)
        )

    probsinfl = np.zeros((n, 1))
    probsinfl[:, 0] = 1 - norm.cdf(zg)
    probslin = pd.DataFrame({"zg": zg, "xb": xb})
    fitted = FittedVals(probs, probsordered, probsinfl, probslin)
    return fitted


def vuong_opiop(opmodel, iopmodel):
    """Run the Vuong test to compare the performance of the OP and iOP model.

    :param opmodel: The OP model from :class:`OpModel`.
    :param iopmodel: The ZiOP model from :class:`IopModel`.
    :return: vuongopiop: Result of the Vuong test
    """
    n1 = len(opmodel.data)
    y = iopmodel.Y
    # can also y = opmodel.Y when 2 models have the same length
    x = iopmodel.X
    # can also x = opmodel.X when 2 models have the same length
    cuts_op = np.repeat(0, len(opmodel.cutpoints)).astype(float)
    xop = opmodel.ordered
    cuts_op[0] = opmodel.cutpoints[0]
    for i in range(1, len(opmodel.cutpoints)):
        cuts_op[i] = cuts_op[i - 1] + np.exp(opmodel.cutpoints[i])
    xbop = pd.DataFrame(index=np.arange(n1), columns=np.arange(len(x.columns)))
    for j in range(len(x.columns)):
        xbop.iloc[:, j] = xop[j] * x.iloc[:, j]
    xbop_sum = xbop.sum(axis=1)
    fitttediop = iopfit(iopmodel).responsefull
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((n1, yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    m = np.zeros(n1)
    probs = np.zeros((n1, yncat))
    probs[:, 0] = norm.cdf(cuts_op[0] - xbop_sum) / fitttediop[:, 0]
    probs[:, yncat - 1] = (1 - norm.cdf(
        cuts_op[yncat - 2] - xbop_sum)) / fitttediop[:, yncat - 1]
    for i in range(1, yncat - 1):
        probs[:, i] = (norm.cdf(cuts_op[i] - xbop_sum)
                       - norm.cdf(cuts_op[i - 1]
                                  - xbop_sum)) / fitttediop[:, i]
    m = np.zeros((n1, yncat))
    for k in range(n1):
        for j in range(yncat):
            m[k, j] = v[k, j] * probs[k, j]
    m2 = m[m != 0]
    mlog = np.log(m2)
    diffmsq = (mlog - np.mean(mlog)) ** 2
    sumdms = sum(diffmsq)
    vuongopiop = (np.sqrt(n1) * (1 / n1) * sum(mlog)) / (
        np.sqrt((1 / n1) * sumdms))
    return vuongopiop


def vuong_opiopc(opmodel, iopcmodel):
    """Run the Vuong test to compare the performance of the OP and iOPC model.

    :param opmodel: The OP model from :class:`OpModel`.
    :param iopcmodel: The iOPC model from :class:`IopCModel`.
    :return: vuongopiopc: Result of the Vuong test
    """
    n1 = len(opmodel.data)
    y = iopcmodel.Y
    x = iopcmodel.X
    cuts_op = np.repeat(0, len(opmodel.cutpoints)).astype(float)
    xop = opmodel.ordered
    cuts_op[0] = opmodel.cutpoints[0]
    for i in range(1, len(opmodel.cutpoints)):
        cuts_op[i] = cuts_op[i - 1] + np.exp(opmodel.cutpoints[i])
    xbop = pd.DataFrame(index=np.arange(n1), columns=np.arange(len(x.columns)))
    for j in range(len(x.columns)):
        xbop.iloc[:, j] = xop[j] * x.iloc[:, j]
    xbop_sum = xbop.sum(axis=1)
    fitttedziopc = iopcfit(iopcmodel).responsefull
    ycat = y.astype("category")
    ycatu = np.unique(ycat)
    yncat = len(ycatu)
    y0 = np.sort(ycatu)
    v = np.zeros((n1, yncat))
    for j in range(yncat):
        v[:, j] = y == y0[j]
    probs = np.zeros((n1, yncat))
    probs[:, 0] = norm.cdf(cuts_op[0] - xbop_sum) / fitttedziopc[:, 0]
    probs[:, yncat - 1] = (1 - norm.cdf(
        cuts_op[yncat - 2] - xbop_sum)) / fitttedziopc[:, yncat - 1]
    for i in range(1, yncat - 1):
        probs[:, i] = (norm.cdf(cuts_op[i] - xbop_sum) - norm.cdf(
            cuts_op[i - 1] - xbop_sum)) / fitttedziopc[:, i]
    m = np.zeros((n1, yncat))
    for k in range(n1):
        for j in range(yncat):
            m[k, j] = v[k, j] * probs[k, j]
    m2 = m[m != 0]
    mlog = np.log(m2)
    diffmsq = (mlog - np.mean(mlog)) ** 2
    sumdms = sum(diffmsq)
    vuongopiopc = (np.sqrt(n1) * (1 / n1) * sum(mlog)) / (
        np.sqrt((1 / n1) * sumdms))
    return vuongopiopc


def split_effects(model, inflvar, nsims=10000):
    """Calculate change in probability of being 0 in the split-probit stage.

    This function calculates the predicted probabilities
    when there is change in value of a variable in the split-probit equation.
    The chosen dummy variable is changed from 0 to 1,
    and chosen numerical variable is mean value + 1 standard deviation.
    Other variables are kept at 0 or mean value
    (Note: the current version of the function
    recognize ordinal variables as numerical).

    :param model: :class:`IopModel` or :class:`IopCModel`.
    :param inflvar: Number representing the location of variable
        in the split-probit equation.
        (attribute .inflate of :class:`IopModel` or :class:`IopCModel`)
    :type inflvar: int
    :param nsims: number of simulated observations, default is 10000.
    :type nsims: int
    :return: changeprobs: a dataframe of the predicted
        probabilities when there is change in the variable (1)
        versus original values (0).
    """
    estimate = model.coefs.iloc[:, 0]
    vcov = model.vcov
    model_z = model.Z
    zsim1 = np.zeros(len(model_z.columns))
    zsim1[0] = 1
    zsima = np.zeros(len(model_z.columns))
    zsima[0] = 1
    for j in range(1, len(model_z.columns)):
        if (
                max(model_z.iloc[:, j]) == 1
                and min(model_z.iloc[:, j]) == 0
                and len(np.unique(model_z.iloc[:, j])) == 2
        ):
            zsim1[j] = 0
        else:
            zsim1[j] = np.mean(model_z.iloc[:, j])
    for j in range(1, len(model_z.columns)):
        if (
                max(model_z.iloc[:, j]) == 1
                and min(model_z.iloc[:, j]) == 0
                and len(np.unique(model_z.iloc[:, j])) == 2
        ):
            zsima[j] = 1
        else:
            zsima[j] = np.mean(model_z.iloc[:, j]) + np.std(model_z.iloc[:, j])
    zsim2 = zsim1.copy()
    zsim2[inflvar] = zsima[inflvar]
    np.random.seed(1)
    probs1 = np.zeros(nsims)
    probs2 = np.zeros(nsims)
    for i in range(nsims):
        gsim = np.random.multivariate_normal(estimate, vcov)
        gsim2 = gsim[model.ycat - 1: model.ycat - 1 + len(model.inflate)]
        zg1 = zsim1.dot(gsim2)
        zg2 = zsim2.dot(gsim2)
        probs1[i] = norm.cdf(zg1)
        probs2[i] = norm.cdf(zg2)
    name = model.coefs.index[model.ycat - 1 + inflvar]
    changeprobs = pd.DataFrame({name.replace("Inflation: ", "") + "= 0": probs1,
                                name.replace("Inflation: ", "") + "= 1": probs2}
                               )
    return changeprobs


def ordered_effects(model, ordvar, nsims=10000):
    """Calculate the changes in probability in each outcome in OP stage.

    This function calculates predicted probabilities
    when there is change in value of a variable
    in the ordered probit equation.
    The chosen dummy variable is changed from 0 to 1,
    and chosen numerical variable is mean value + 1 standard deviation.
    Other variables are kept at 0 or mean value
    (Note: the current version of the function
    recognize ordinal variables as numerical).

    :param model: :class:`IopModel` or :class:`IopCModel`.
    :param ordvar: Number representing the location of variable
        in the ordered probit equation.
        (attribute .ordered of :class:`IopModel` or :class:`IopCModel`)
    :type ordvar: int
    :param nsims: number of simulated observations, default is 10000.
    :type nsims: int
    :return: changeprobs: a dataframe of the predicted
        probabilities when there is change in the variable for each outcome (1)
        versus original values (0).
    """
    estimate = model.coefs.iloc[:, 0]
    vcov = model.vcov
    model_x = model.X
    xsim1 = np.zeros(len(model_x.columns))
    xsima = np.zeros(len(model_x.columns))
    for j in range(len(model_x.columns)):
        if (
                max(model_x.iloc[:, j]) == 1
                and min(model_x.iloc[:, j]) == 0
                and len(np.unique(model_x.iloc[:, j])) == 2
        ):
            xsim1[j] = 0
        else:
            xsim1[j] = np.mean(model_x.iloc[:, j])
    for j in range(len(model_x.columns)):
        if (
                max(model_x.iloc[:, j]) == 1
                and min(model_x.iloc[:, j]) == 0
                and len(np.unique(model_x.iloc[:, j])) == 2
        ):
            xsima[j] = 1
        else:
            xsima[j] = np.mean(model_x.iloc[:, j]) + np.std(model_x.iloc[:, j])
    xsim2 = xsim1.copy()
    xsim2[ordvar] = xsima[ordvar]
    np.random.seed(1)
    probsordered1 = np.zeros(model.ycat)
    probsordered2 = np.zeros(model.ycat)
    cprobs = np.zeros((model.ycat - 1, 1))
    cprobs[0, 0] = model.cutpoints[0]
    for j in range(1, model.ycat - 1):
        cprobs[j, 0] = cprobs[j - 1, 0] + np.exp(model.cutpoints[j])
    probs1 = pd.DataFrame(index=np.arange(nsims),
                          columns=np.arange(model.ycat))
    probs2 = pd.DataFrame(index=np.arange(nsims),
                          columns=np.arange(model.ycat))
    name = model.coefs.index[model.ycat - 1 + len(model.inflate) + ordvar]
    probs1 = probs1.add_suffix(": " + name.replace("Ordered: ", "") + " = 0")
    probs2 = probs2.add_suffix(": " + name.replace("Ordered: ", "") + " = 1")
    for i in range(nsims):
        bsim = np.random.multivariate_normal(estimate, vcov)
        bsim2 = bsim[model.ycat - 1 + len(model.inflate):
                     model.ycat - 1 + len(model.inflate) + len(model.ordered)]
        xb1 = xsim1.dot(bsim2)
        xb2 = xsim2.dot(bsim2)
        probsordered1[model.ycat - 1] = 1 - norm.cdf(
            cprobs[model.ycat - 2, 0] - xb1)
        probsordered1[0] = norm.cdf(cprobs[0, 0] - xb1)
        for j in range(1, model.ycat - 1):
            probsordered1[j] = norm.cdf(cprobs[j, 0] - xb1) - (
                norm.cdf(cprobs[j - 1, 0] - xb1)
            )
        probsordered2[model.ycat - 1] = 1 - norm.cdf(
            cprobs[model.ycat - 2, 0] - xb2)
        probsordered2[0] = norm.cdf(cprobs[0, 0] - xb2)
        for j in range(1, model.ycat - 1):
            probsordered2[j] = norm.cdf(cprobs[j, 0] - xb2) - (
                norm.cdf(cprobs[j - 1, 0] - xb2)
            )
        probs1.iloc[i:, ] = probsordered1
        probs2.iloc[i:, ] = probsordered2
    changeprobs = pd.DataFrame(index=np.arange(nsims),
                               columns=np.arange(2 * model.ycat))
    newnames = list(np.repeat("", model.ycat * 2))
    for j in range(0, 2 * model.ycat, 2):
        changeprobs.iloc[:, j] = probs1.iloc[:, round(j / 2)]
    for j in range(1, 2 * model.ycat, 2):
        changeprobs.iloc[:, j] = probs2.iloc[:, round((j - 1) / 2)]
    for j in range(0, 2 * model.ycat, 2):
        newnames[j] = list(probs1.columns)[round(j / 2)]
    for j in range(1, 2 * model.ycat, 2):
        newnames[j] = list(probs2.columns)[round((j - 1) / 2)]
    changeprobs.columns = newnames
    return changeprobs
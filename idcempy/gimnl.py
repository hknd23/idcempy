"""Classes and Functions for the gimnl module."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class GimnlModel:
    """Store model results from :py:func:`gimnlmod`."""

    def __init__(
            self,
            modeltype,
            reference,
            inflatecat,
            llik,
            coef,
            aic,
            vcov,
            data,
            xs,
            zs,
            x_,
            yx_,
            z_,
            ycatu,
            xstr,
            ystr,
            zstr,
    ):
        """Store model results, goodness-of-fit tests, and other information
        for the Generalized Inflated Multinomial Logit Model.

        :param modeltype: Type of GIMNL Model (bimnl3, simnl3, or timnl3),
        indicating the inflated category.
        :param reference: Order of categories. The order category will be
        the first element.
        :param llik: Log-Likelihood.
        :param coef: Model coefficients.
        :param aic: Model Akaike information .
        :param vcov: Variance-Covariance matrix.
            (optimized as inverted Hessian matrix)
        :param data: Full dataset used for estimation, with missing values
        dropped.
        :param zs: Inflation stage estimates (Gammas).
        :param xs: Multinomial Logit probit estimates (Betas).
        :param ycatu: Number of categories in the Dependent Variable (DV).
        :param x_: X Data.
        :param yx_: Y (DV) data.
        :param z_: Z Data.
        :param xstr: list of string for variable names in the MNL stage.
        :param ystr: list of string for dependent variable name .
        :param zstr: list of string for variable names in the Logit Split
        stage.
        """
        self.modeltype = modeltype
        self.reference = reference
        self.inflatecat = inflatecat
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.split = zs
        self.multinom = xs
        self.ycat = ycatu
        self.X = x_
        self.Y = yx_
        self.Z = z_
        self.xstr = xstr
        self.ystr = ystr
        self.zstr = zstr


class MnlModel:
    """Store model results from :py:func:`gimnlmod`."""

    def __init__(
            self,
            modeltype,
            reference,
            llik,
            coef,
            aic,
            vcov,
            data,
            xs,
            x_,
            yx_,
            ycatu,
            xstr,
            ystr,
    ):
        """Store model results, goodness-of-fit tests, and other information
        for the Multinomial Logit Model.

        :param modeltype: Type of Model (mnl3).
        :param reference: Order of categories. The order category will be
        the first element.
        :param llik: Log-Likelihood.
        :param coef: Model coefficients.
        :param aic: Model Akaike information .
        :param vcov: Variance-Covariance matrix.
            (optimized as inverted Hessian matrix)
        :param data: Full dataset used in estimation, with missing values
        dropped.
        :param xs: Multinomial Logit estimates (Betas).
        :param ycatu: Number of categories in the Dependent Variable (DV).
        :param x_: X Data.
        :param yx_: Y (DV) data.
        :param xstr: list of string for x names.
        :param ystr: list of string for y names.
        """
        self.modeltype = modeltype
        self.reference = reference
        self.llik = llik
        self.coefs = coef
        self.AIC = aic
        self.vcov = vcov
        self.data = data
        self.multinom = xs
        self.ycat = ycatu
        self.X = x_
        self.Y = yx_
        self.xstr = xstr
        self.ystr = ystr


def mnl3(pstart, x2, x3, y, reference):
    """
    Likelihood function for the baseline inflated three-category MNL model.

    :param pstart: starting parameters.
    :param x2: X (Multinomial Logit) covariates.
    :param x3: X (Multinomial Logit) covariates (should be identical to x2).
    :param y: Dependent Variable (DV).
    :param reference: order of categories.
    """
    b2 = pstart[0: len(x2.columns)]
    b3 = pstart[len(x2.columns): (len(pstart))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = np.sum(
        np.log(p1) * (y == reference[0])
        + np.log(p2) * (y == reference[1])
        + np.log(p3) * (y == reference[2])
    )
    llik = -1 * np.sum(lik)
    return llik


def bimnl3(pstart, x2, x3, y, z, reference):
    """
    Likelihood function for the baseline inflated three-category MNL model.

    :param pstart: starting parameters.
    :param x2: X (Multinomial Logit) covariates.
    :param x3: X (Multinomial Logit) covariates (should be identical to x2.
    :param y: Dependent Variable (DV).
    :param z: Logit Split stage covariates.
    :param reference: order of categories (first category/baseline inflated).
    """
    b2 = pstart[len(z.columns): (len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)): (len(pstart))]
    gamma = pstart[0: (len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = np.sum(
        np.log((1 - pz) + pz * p1) * (y == reference[0])
        + np.log(pz * p2) * (y == reference[1])
        + np.log(pz * p3) * (y == reference[2])
    )
    llik = -1 * np.sum(lik)
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
    b2 = pstart[len(z.columns): (len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)): (len(pstart))]
    gamma = pstart[0: (len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = np.sum(
        np.log(pz * p1) * (y == reference[0])
        + np.log((1 - pz) + pz * p2) * (y == reference[1])
        + np.log(pz * p3) * (y == reference[2])
    )
    llik = -1 * np.sum(lik)
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
    b2 = pstart[len(z.columns): (len(z.columns) + len(x2.columns))]
    b3 = pstart[(len(z.columns) + len(x2.columns)): (len(pstart))]
    gamma = pstart[0: (len(z.columns))]
    xb2 = x2.dot(b2)
    xb3 = x3.dot(b3)
    zg = z.dot(gamma)
    pz = 1 / (1 + np.exp(-zg))
    p1 = 1 / (1 + np.exp(xb2) + np.exp(xb3))
    p2 = p1 * np.exp(xb2)
    p3 = p1 * np.exp(xb3)
    lik = np.sum(
        np.log(pz * p1) * (y == reference[0])
        + np.log(pz * p2) * (y == reference[1])
        + np.log((1 - pz) + pz * p3) * (y == reference[2])
    )
    llik = -1 * np.sum(lik)
    return llik


def gimnlresults(model, data, x, y, z, modeltype, reference, inflatecat):
    """
    Produce estimation results, part of :py:func:`gimnlmod`.

    Store estimates, model AIC, and other information to
    :py:class:`GimnlModel`.

    :param model: object model estimated.
    :param data: dataset with missing values omitted.
    :param x: Multinomial Logit stage covariates.
    :param y: Dependent Variable (DV).
    :param z: Spplit-stage covariates.
    :param modeltype: type of inflated MNL model.
    :param reference: order of categories.
    :param inflatecat: inflated category.
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
    x_.insert(0, "int", np.repeat(1, len(x_)))
    names = list()
    if modeltype == "bimnl3":
        x2 = x_
        x3 = x_
        for s in range(z_.shape[1]):
            names.append("Inflation: " + z_.columns[s])
        for s in range(x2.shape[1]):
            names.append(str(reference[1]) + ": " + x2.columns[s])
        for s in range(x3.shape[1]):
            names.append(str(reference[2]) + ": " + x3.columns[s])
        xs = model.x[(z_.shape[1]): (z_.shape[1] + x2.shape[1] + x3.shape[1])]
    zs = model.x[0: (z_.shape[1])]
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
    llik = -1 * model.fun
    model = GimnlModel(
        modeltype,
        reference,
        inflatecat,
        llik,
        coef,
        aic,
        model.hess_inv,
        datasetnew,
        xs,
        zs,
        x_,
        yx_,
        z_,
        yncat,
        x,
        y,
        z,
    )
    return model


def mnlresults(model, data, x, y, modeltype, reference):
    """
    Produce estimation results, part of :py:func:`mnlmod`.

    Store estimates, model AIC, and other information to :py:class:`MnlModel`.

    :param model: object model estimated.
    :param data: dataset.
    :param x: Multinomial Logit stage covariates.
    :param y: Dependent Variable (DV).
    :param modeltype: type of inflated MNL model.
    :param reference: order of categories.
    """
    varlist = np.unique(y + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    x_.insert(0, "int", np.repeat(1, len(x_)))
    names = list()
    if modeltype == "mnl3":
        x2 = x_
        x3 = x_
        for s in range(x2.shape[1]):
            names.append(str(reference[1]) + ": " + x2.columns[s])
        for s in range(x3.shape[1]):
            names.append(str(reference[2]) + ": " + x3.columns[s])
        xs = model.x[0: (x2.shape[1] + x3.shape[1])]
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
    llik = -1 * model.fun
    model = MnlModel(
        modeltype,
        reference,
        llik,
        coef,
        aic,
        model.hess_inv,
        datasetnew,
        xs,
        x_,
        yx_,
        yncat,
        x,
        y,
    )
    return model


def gimnlmod(data, x, y, z, reference, inflatecat, method="BFGS", pstart=None):
    """
    Estimate three-category inflated Multinomial Logit model.

    :param data: dataset.
    :param x: MNL stage covariates.
    :param y: Dependent Variable. Variable needs to be in factor form,
            with a number from 0-2 representing each category.
    :param z: Inflation stage covariates.
    :param reference:  List specifying the order of categories (e.g [0, 1,
    2], [2, 1, 0]. etc...). The parameter inflatecat then specifies which
    category in the list inflated.
    :param inflatecat: inflated category. One of "baseline" for the first,
    "second" for the second,
         or "third" for the third in reference list to specify the inflated
         category.
    :param method: Optimization method.  Default is 'BFGS'. For other
    available methods, see scipy.optimize.minimize documentation.
    :param pstart: Starting parameters.
    """
    varlist = np.unique(y + z + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    if yncat == 3:
        modeltype = "bimnl3"
    else:
        raise Exception(
            "Function only supports Dependent Variable with 3 " "categories."
        )
    z_ = datasetnew[z]
    z_.insert(0, "int", np.repeat(1, len(z_)))
    x_.insert(0, "int", np.repeat(1, len(x_)))
    if modeltype == "bimnl3":
        x2 = x_
        x3 = x_
        if pstart is None:
            pstart = np.repeat(
                0.01, (len(x2.columns) + len(x3.columns) + len(z_.columns))
            )
        if inflatecat == "baseline":
            model = minimize(
                bimnl3,
                pstart,
                args=(x2, x3, yx_, z_, reference),
                method=method,
                options={"gtol": 1e-6, "disp": True, "maxiter": 500},
            )
        elif inflatecat == "second":
            model = minimize(
                simnl3,
                pstart,
                args=(x2, x3, yx_, z_, reference),
                method=method,
                options={"gtol": 1e-6, "disp": True, "maxiter": 500},
            )
        elif inflatecat == "third":
            model = minimize(
                timnl3,
                pstart,
                args=(x2, x3, yx_, z_, reference),
                method=method,
                options={"gtol": 1e-6, "disp": True, "maxiter": 500},
            )
    results = gimnlresults(model, data, x, y, z, modeltype, reference,
                           inflatecat)
    return results


def mnlmod(data, x, y, reference, method="BFGS", pstart=None):
    """
    Estimate three-category Multinomial Logit model.

    :param data: dataset.
    :param x: MNL stage covariates.
    :param y: Dependent Variable. Variable needs to be in factor form,
         with a number from 0-2 representing each category.
    :param reference: order of categories. List specifying the order of
    categories (e.g [0, 1, 2], [2, 1, 0]. etc...)
    :param method: Optimization method.  Default is 'BFGS'. For other
    available methods, see scipy.optimize.minimize documentation.
    :param pstart: Starting parameters.
    """
    varlist = np.unique(y + x)
    dataset = data[varlist]
    datasetnew = dataset.dropna(how="any")
    datasetnew = datasetnew.reset_index(drop=True)
    x_ = datasetnew[x]
    y_ = datasetnew[y]
    yx_ = y_.iloc[:, 0]
    yncat = len(np.unique(yx_))
    if yncat == 3:
        modeltype = "mnl3"
    else:
        raise Exception(
            "Function only supports Dependent Variable with 3 " "categories."
        )
    x_.insert(0, "int", np.repeat(1, len(x_)))
    if modeltype == "mnl3":
        x2 = x_
        x3 = x_
        if pstart is None:
            pstart = np.repeat(
                0.01, (len(x2.columns) + len(x3.columns))
            )
            model = minimize(
                mnl3,
                pstart,
                args=(x2, x3, yx_, reference),
                method=method,
                options={"gtol": 1e-6, "disp": True, "maxiter": 500},
            )
    results = mnlresults(model, data, x, y, modeltype, reference)
    return results


def vuong_gimnl(modelmnl, modelgimnl):
    """
    Run the Vuong test to compare the performance of the MNL and GIMNL model.

    For the function to run properly, the models need to have the same X
    covariates and same number of observations.

    :param modelmnl: A :class:`MnlModel` object.
    :param modelgimnl: A :class:`GimnlModel` object.
    """
    xb2_mnl = modelmnl.X.dot(modelmnl.multinom[0: len(modelmnl.X.columns)])
    xb3_mnl = modelmnl.X.dot(modelmnl.multinom[len(modelmnl.X.columns):
                                               len(modelmnl.multinom)])
    xb2_gimnl = modelgimnl.X.dot(
        modelgimnl.multinom[0: len(modelgimnl.X.columns)])
    xb3_gimnl = modelgimnl.X.dot(modelgimnl.multinom[len(modelgimnl.X.columns):
                                                     len(modelgimnl.multinom)])
    zg_gimnl = modelgimnl.Z.dot(modelgimnl.split)
    mnldenom = 1 + np.exp(xb2_mnl) + np.exp(xb3_mnl)
    gimnldenom = 1 + np.exp(xb2_gimnl) + np.exp(xb3_gimnl)
    p1mnl = 1 / mnldenom
    p2mnl = np.exp(xb2_mnl) / mnldenom
    p3mnl = np.exp(xb3_mnl) / mnldenom
    if modelgimnl.inflatecat == "baseline":
        p1gimnl = ((1 / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl)))
                   + (1 - (1 / (1 + np.exp(-zg_gimnl)))))
        p2gimnl = (np.exp(xb2_gimnl) / gimnldenom) * (
                1 / (1 + np.exp(-zg_gimnl)))
        p3gimnl = (np.exp(xb3_gimnl) / gimnldenom) * (
                1 / (1 + np.exp(-zg_gimnl)))
    elif modelgimnl.inflatecat == "second":
        p1gimnl = ((1 / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl))))
        p2gimnl = ((np.exp(xb2_gimnl) / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl)))
                   + (1 - (1 / (1 + np.exp(-zg_gimnl)))))
        p3gimnl = ((np.exp(xb3_gimnl) / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl))))
    elif modelgimnl.inflatecat == "third":
        p1gimnl = ((1 / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl))))
        p2gimnl = ((np.exp(xb2_gimnl) / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl))))
        p3gimnl = ((np.exp(xb3_gimnl) / gimnldenom)
                   * (1 / (1 + np.exp(-zg_gimnl)))
                   + (1 - (1 / (1 + np.exp(-zg_gimnl)))))
    m = np.zeros(len(modelgimnl.X))
    reference = modelgimnl.reference
    y = modelgimnl.Y
    for i in range(len(m)):
        if y[i] == reference[0]:
            m[i] = np.log(p1mnl[i] / p1gimnl[i])
        elif y[i] == reference[0]:
            m[i] = np.log(p2mnl[i] / p2gimnl[i])
        elif y[i] == reference[2]:
            m[i] = np.log(p3mnl[i] / p3gimnl[i])
    diffmsq = (m - np.mean(m)) ** 2
    sumdms = sum(diffmsq)
    vuong = (np.sqrt(len(m)) * (1 / len(m)) * sum(m)) / (
        np.sqrt((1 / len(m)) * sumdms))
    return vuong

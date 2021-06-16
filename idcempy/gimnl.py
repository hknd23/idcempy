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
        :param data: Model data used for estimation, subsetted to selected
            variables and missing values listwise deleted.
        :param zs: Inflation stage estimates (Gammas).
        :param xs: Multinomial Logit probit estimates (Betas).
        :param ycatu: Number of categories in the Dependent variable (DV).
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
        :param data: Model data used for estimation, subsetted to selected
            variables and missing values listwise deleted.
        :param xs: Multinomial Logit estimates (Betas).
        :param ycatu: Number of categories in the Dependent variable (DV).
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
    Calculate likelihood function for the baseline inflated three-category MNL
    model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x2: Multinomial Logit variables. Data subsetted to selected
        variables.
    :type x2: pandas dataframe
    :param x3: Multinomial Logit variables (should be identical to x2.
    :type x3: pandas dataframe
    :param y: Dependent variable (DV). Data subsetted to selected
        variable.
    :type y: pandas dataframe
    :param reference: list of order of categories.
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
    Calculate likelihood function for the baseline inflated three-category
    MNL model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x2: Multinomial Logit variables. Data subsetted to selected
        variables.
    :type x2: pandas dataframe
    :param x3: Multinomial Logit variables (should be identical to x2.
    :type x3: pandas dataframe
    :param y: Dependent variable (DV). Data subsetted to selected
        variable.
    :type y: pandas dataframe
    :param z: Logit Split stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param reference: list of order of categories (first element will be the
        inflated category).
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
    Calculate likelihood function for the second category inflated
    three-category MNL model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x2: Multinomial Logit variables. Data subsetted to selected
        variables.
    :type x2: pandas dataframe
    :param x3: Multinomial Logit variables (should be identical to x2.
    :type x3: pandas dataframe
    :param y: Dependent variable (DV). Data subsetted to selected
        variable.
    :type y: pandas dataframe
    :param z: Logit Split stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param reference: list of order of categories (second element will be the
        inflated category).
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
    Calculate likelihood function for the third category inflated
    three-category MNL model.

    :param pstart: A list of starting parameters.
    :type pstart: list
    :param x2: Multinomial Logit variables. Data subsetted to selected
        variables.
    :type x2: pandas dataframe
    :param x3: Multinomial Logit variables (should be identical to x2.
    :type x3: pandas dataframe
    :param y: Dependent variable (DV). Data subsetted to selected
        variable.
    :type y: pandas dataframe
    :param z: Logit Split stage variables. Data subsetted to selected
        variables.
    :type z: pandas dataframe
    :param reference: list of order of categories (third element will be the
        inflated category).
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

    :param model: Estimation results.
    :param data: Model data used for estimation, subsetted to selected
        variables and missing values listwise deleted.
    :param x: Multinomial Logit stage variables.
    :param y: Dependent variable (DV).
    :param z: Spplit-stage variables.
    :param modeltype: type of inflated MNL model. One of 'bimnl3', 'simnl3',
        or 'timnl3'.
    :param reference: List of order of categories. First element is the
        baseline/ reference category.
    :param inflatecat: Tnflated category. One of 'baseline', 'second',
        or 'third'.
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

    :param model: Estimation results.
    :param data: Model data used for estimation, subsetted to selected
        variables and missing values listwise deleted.
    :param x: Multinomial Logit variables.
    :param y: Dependent variable (DV).
    :param modeltype: three-category MNL model ('mnl3').
    :param reference: List of order of categories. First element is the
        baseline/ reference category.
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

    :param data: Full dataset.
    :type data: pandas.dataframe
    :param x: MNL stage variables. The elements must match column
    names of ``data``.
    :type x: list of str
    :param y: Dependent Variable. Values should be integers,
            with a number from 0-2 representing each category. The element
            must match column names of ``data``.
    :type y: list of str
    :param z: Inflation stage variables. The elements must match column
    names of ``data``.
    :type z: list of str
    :param reference:  List of three elements specifying the order of
        categories (e.g [0, 1, 2], [2, 1, 0]. etc...). The first element is
        the baseline/reference category. The parameter
        inflatecat then specifies which category in the list inflated.
    :type reference: list of int
    :param inflatecat: A string specifying the inflated category. One of
        "baseline" for the first, "second" for the second, or "third" for the
        third in reference list to specify the inflated category.
    :param method: Optimization method.  Default is 'BFGS'. For other
        available methods, see scipy.optimize.minimize documentation.
    :param pstart: A list of starting parameters.
    :type pstart: list
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

    :param data: Full dataset.
    :type data: pandas.dataframe
    :param x: MNL stage variables. The elements must match column
    names of ``data``
    :type x: list of str
    :param y: Dependent variable. Values should be integers,
            with a number from 0-2 representing each category.
    :param reference:  List of three elements specifying the order of
        categories (e.g [0, 1, 2], [2, 1, 0]. etc...). The first element is
        the baseline/reference category.
    :type reference: list of int
    :param method: Optimization method.  Default is 'BFGS'. For other
        available methods, see scipy.optimize.minimize documentation.
    :param pstart: A list of starting parameters.
    :type pstart: list
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

import numpy as np
import pandas as pd
from idcempy import zmiopc

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Xsmall = ['logGDPpc', 'parliament', 'disaster']
Z = ['logGDPpc', 'parliament']
Y = ['rep_civwar_DV']
data = DAT

pstartziop = [-1.31, .32, 2.5, -.21, .2, -0.2, -0.4, 0.2, .9, -.4]

pstartziopc = [-1.31, .32, 2.5, -.21,
               .2, -0.2, -0.4, 0.2, .9, -.4, .1]


def test_ziopc_coefs():
    ziopc_JCR = zmiopc.iopcmod('ziopc',
                               data, X, Y, Z, pstart=pstartziopc)
    assert len(ziopc_JCR.coefs) == len(pstartziopc)


def test_ziop_coefs():
    ziop_JCR = zmiopc.iopmod('ziop', data, X, Y, Z, pstart=pstartziop)
    assert len(ziop_JCR.coefs) == len(pstartziop)


def test_iopc_fitted():
    ziopc_JCR = zmiopc.iopcmod('ziopc',
                               data, X, Y, Z, pstart=pstartziopc)
    fitttedziopc = zmiopc.iopcfit(ziopc_JCR)
    assert len(fitttedziopc.responseordered) == len(fitttedziopc.responsefull)


def test_iop_fitted():
    ziop_JCR = zmiopc.iopmod('ziop', data, X, Y, Z, pstart=pstartziop)
    fitttedziop = zmiopc.iopfit(ziop_JCR)
    assert len(fitttedziop.responseordered) == len(fitttedziop.responsefull)


# OP Model
pstartop = np.array([-1, 0.3, -0.2, -0.5, 0.2, .9, -.4])

DAT = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/bp_exact_for_analysis.dta")
X = ['logGDPpc', 'parliament', 'disaster', 'major_oil', 'major_primary']
Y = ['rep_civwar_DV']
data = DAT
JCR_OP = zmiopc.opmod(pstartop, data, X, Y)

# MiOP Examples
dataeu = pd.read_stata("C:/Users/Nguyen/Box/Summer 20/EUKnowledge.dta")

Y = ["EU_support_ET"]
X = ['Xenophobia', 'discuss_politics']
Z = ['discuss_politics', 'EU_Know_obj']
yvar = np.unique(dataeu[Y])


def test_miopc_coefs():
    miopc_EU = zmiopc.iopcmod('miopc',
                              dataeu, X, Y, Z)
    assert len(miopc_EU.coefs) == len(X + Z) + len(yvar) + 1


def test_miop_coefs():
    miop_EU = zmiopc.iopmod('miop', dataeu, X, Y, Z)
    assert len(miop_EU.coefs) == len(X + Z) + len(yvar)

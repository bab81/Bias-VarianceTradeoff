import numpy as np
from numpy.matlib import repmat
import matplotlib
import matplotlib.pyplot as plt
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

OFFSET = 1.75
X, y = toydata(OFFSET, 1000)

# Visualize the generated data
ind1 = y == 1
ind2 = y == 2
plt.figure(figsize=(10,6))
plt.scatter(X[ind1, 0], X[ind1, 1], c='r', marker='o', label='Class 1')
plt.scatter(X[ind2, 0], X[ind2, 1], c='b', marker='o', label='Class 2')
plt.legend();
#plt.show();


def computeybar(xTe, OFFSET):
    """
    function [ybar]=computeybar(xTe, OFFSET);

    computes the expected label 'ybar' for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    ybar : a nx1 vector of the expected labels for vectors xTe
    noise:
    """
    n, d = xTe.shape
    ybar = np.zeros(n)

    # Feel free to use the following function to compute p(x|y)
    # By default, mean is 0 and std. deviation is 1.
    normpdf = lambda x, mu, sigma: np.exp(-0.5 * np.power((x - mu) / sigma, 2)) / (np.sqrt(2 * np.pi) * sigma)

    # step 1: calculate 𝑝(𝐱|𝑦=1) = 𝑝(𝑥1|𝑦=1)𝑝(𝑥2|𝑦=1)

    # step 1a: 𝑝(𝑥1|𝑦=1)
    pofx1_givenyis1 = normpdf(xTe[:, 0], 0, 1)
    # step 1b: 𝑝(𝑥2|𝑦=1)
    pofx2_givenyis1 = normpdf(xTe[:, 1], 0, 1)

    # complete step 1: calculate 𝑝(𝐱|𝑦=1) = 𝑝(𝑥1|𝑦=1)𝑝(𝑥2|𝑦=1)
    pofx_givenyis1 = np.multiply(pofx1_givenyis1, pofx2_givenyis1)

    # step 2: calculate 𝑝(𝐱|𝑦=2) = 𝑝(𝑥1|𝑦=2)𝑝(𝑥2|𝑦=2)

    # step 2a: 𝑝(𝑥1|𝑦=2)
    pofx1_givenyis2 = normpdf(xTe[:, 0], OFFSET, 1)
    # step 2b: 𝑝(𝑥2|𝑦=2)
    pofx2_givenyis2 = normpdf(xTe[:, 1], OFFSET, 1)

    # complete step 2: calculate 𝑝(𝐱|𝑦=2) = 𝑝(𝑥1|𝑦=2)𝑝(𝑥2|𝑦=2)
    pofx_givenyis2 = np.multiply(pofx1_givenyis2, pofx2_givenyis2)

    # step 3
    numerator = pofx_givenyis1 + 2 * pofx_givenyis2
    denominator = pofx_givenyis1 + pofx_givenyis2

    # step 4
    ybar = numerator / denominator

    return ybar


OFFSET = 3;
xTe = np.array([
    [0.45864, 0.71552],
    [2.44662, 1.68167],
    [1.00345, 0.15182],
    [-0.10560, -0.48155],
    [3.07264, 3.81535],
    [3.13035, 2.72151],
    [2.25265, 3.78697]])
yTe = np.array([1, 2, 1, 1, 2, 2, 2])

ybar = computeybar(xTe, OFFSET)


# biasvariancedemo

OFFSET = 1.75
# how big is the training set size N
Nsmall = 75
# how big is a really big data set (approx. infinity)
Nbig = 7500
# how many models do you want to average over
NMODELS = 100
# What regularization constants to evaluate
depths = [0, 1, 2, 3, 4, 5, 6, np.inf]

# we store
Ndepths = len(depths)
lbias = np.zeros(Ndepths)
lvariance = np.zeros(Ndepths)
ltotal = np.zeros(Ndepths)
lnoise = np.zeros(Ndepths)
lsum = np.zeros(Ndepths)
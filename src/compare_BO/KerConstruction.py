"""
@Tong
26-01-2018
decipher kernel string
"""
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
import numpy as np


def str2ker(str):
    k1 = C(1.0) * RBF(length_scale=1)
    k2 = C(1.0) * RationalQuadratic(length_scale=1)
    k4 = DotProduct(sigma_0=1)
    k3 = C(1.0) * ExpSineSquared(length_scale=1, periodicity=1)
    k5 = WhiteKernel(1.0)
    map = {"s": k1, "r": k2, "p": k3, "l": k4}

    #  if basic kernel
    if len(str) == 1:
        ker = map[str]
    else:

        # if composite kernel
        ker = []
        factor = map[str[0]]
        op = str[1]
        for i in range(2, len(str), 2):

            # if the operator is *, use @ covProd to continue costructing the
            #  factor
            if op == '*':
                factor = factor * map[str[i]]

                # the end?
                if i == len(str) - 1:
                    if not ker:
                        ker = factor
                    else:
                        ker = ker + factor
                else:
                    op = str[i + 1]
                # if the oprator is +, combine current factor with ker then form a
                #  new factor
            else:
                if not ker:
                    ker = factor
                else:
                    ker = ker + factor

                factor = map[str[i]]

                # % the end?
                if i == len(str) - 1:
                    ker = ker + factor
                else:
                    op = str[i + 1]
    ker = ker + k5
    return ker


# extend the kernel space from a certain kernel
# return the strings of children kernels
def ker_extend(cur, base):
    children = []
    for i in range(len(base)):
        children.append(cur + '+' + base[i])
        children.append(cur + '*' + base[i])
    return children


# parameters for hyper prior distribution, use the value in BO for model selection paper
def prior_params(str):
    se = np.array([[0.4, 0.1], [0.7, 0.7]])
    rq = np.array([[0.4, 0.05, 0.1], [0.7, 0.7, 0.7]])
    per = np.array([[0.4, 0.1, 2], [0.7, 0.7, 0.7]])
    lin = np.array([[0.4], [0.7]])
    noise = np.array([[0.1], [1]])

    map = {"s": se, "r": rq, "p": per, "l": lin}
    mu = []
    sigma = []
    for i in range(0, len(str), 2):
        mu.extend(map[str[i]][0, :])
        sigma.extend(map[str[i]][1, :])
    mu.extend(noise[0, :])
    sigma.extend(noise[1, :])
    mu = np.array(mu)
    sigma = np.array(sigma)
    return mu, sigma


def kernel_space(base, depth):
    """
    :param base:
    :type: list
    :param depth:
    :return:
    :type: 1d array
    """

    space = base.copy()  # ATTENTION
    level_start = 0
    level_end = len(space)
    for i in range(depth - 1):
        for j in range(level_start, level_end):
            c = ker_extend(space[j], base)
            space.extend(c)

        level_start = level_end
        level_end = len(space)
    return np.array(space)

# compisite_kernel = str2ker('s*r*p') + WhiteKernel(1.0)
# print(compisite_kernel)

# prior_params('s*r*p')
# kernel_space(['s', 'r', 'p'], 4)

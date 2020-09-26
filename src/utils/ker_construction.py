from gpflow.kernels import RBF, RationalQuadratic as RQ, Periodic as Per, Linear as Lin, White
import numpy.random as rnd
import numpy as np


def map_base_kernel(s, dim, init_hyper_fixed):
    if init_hyper_fixed:
        # TODO: ARD
        # if dim == 1:
        if s == 's':
            k = RBF(dim, lengthscales=1, variance=0.5)
        elif s == 'r':
            k = RQ(dim, lengthscales=1, variance=0.5, alpha=0.5)
        elif s == 'p':
            k = Per(dim, period=1, lengthscales=0.1, variance=0.5)
        else:
            k = Lin(dim, variance=0.5)
        # else:
        #     if s == 's':
        #         k = RBF(dim, lengthscales=1 * np.ones(dim), variance=0.5)
        #     elif s == 'r':
        #         k = RQ(dim, lengthscales=1 * np.ones(dim), variance=0.5, alpha=0.5)
        #     elif s == 'p':
        #         k = Per(dim, period=1, lengthscales=0.1, variance=0.5)
        #     else:
        #         k = Lin(dim, variance=0.5 * np.ones(dim))
    else:
        if dim == 1:
            # this is for reusing hypers of trained models
            if s == 's':
                k = RBF(dim, lengthscales=rnd.ranf() * 5)
            elif s == 'r':
                k = RQ(dim, lengthscales=rnd.ranf() * 5)
            elif s == 'p':
                k = Per(dim, period=rnd.ranf() * 5, lengthscales=rnd.ranf() * 5)
            else:
                k = Lin(dim, variance=rnd.ranf() * 10)
        else:
            if s == 's':
                k = RBF(dim, lengthscales=rnd.ranf(dim) * 5)
            elif s == 'r':
                k = RQ(dim, lengthscales=rnd.ranf(dim) * 5)
            elif s == 'p':
                k = Per(dim, period=rnd.ranf() * 5, lengthscales=rnd.ranf() * 5)
            else:
                k = Lin(dim, variance=rnd.ranf(dim) * 10)
    return k


# if bracket=True, operation * will NOT multiply the current one with all the previous ones
def str2ker(ker_str, dim, init_hyper_fixed, bracket=None):
    if bracket == True:
        # if base kernel
        if len(ker_str) == 1:
            ker = map_base_kernel(ker_str, dim, init_hyper_fixed)
        else:
            ker = map_base_kernel(ker_str[0], dim, init_hyper_fixed)
            for i in range(1, len(ker_str) - 1, 2):
                op = ker_str[i]
                factor = map_base_kernel(ker_str[i + 1], dim, init_hyper_fixed)
                if op == '*':
                    ker = ker * factor
                else:
                    ker = ker + factor
    else:
        #  if base kernel
        if len(ker_str) == 1:
            ker = map_base_kernel(ker_str, dim, init_hyper_fixed)
        else:

            # if composite kernel
            ker = []
            factor = map_base_kernel(ker_str[0], dim, init_hyper_fixed)
            op = ker_str[1]
            for i in range(2, len(ker_str), 2):

                # if the operator is *, use @ covProd to continue constructing the
                #  factor
                if op == '*':
                    factor = factor * map_base_kernel(ker_str[i], dim, init_hyper_fixed)

                    # the end?
                    if i == len(ker_str) - 1:
                        if not ker:
                            ker = factor
                        else:
                            ker = ker + factor
                    else:
                        op = ker_str[i + 1]
                    # if the operator is +, combine current factor with ker then form a
                    #  new factor
                else:
                    if not ker:
                        ker = factor
                    else:
                        ker = ker + factor

                    factor = map_base_kernel(ker_str[i], dim, init_hyper_fixed)

                    # % the end?
                    if i == len(ker_str) - 1:
                        ker = ker + factor
                    else:
                        op = ker_str[i + 1]
    return ker


# * will multiply the current one with all the previous components
def str2ker_bracket(ker_str, dim, init_hyper_fixed):
    # if base kernel
    if len(ker_str) == 1:
        ker = map_base_kernel(ker_str, dim, init_hyper_fixed)
    else:
        ker = map_base_kernel(ker_str[0], dim, init_hyper_fixed)
        for i in range(1, len(ker_str) - 1, 2):
            op = ker_str[i]
            factor = map_base_kernel(ker_str[i + 1], dim, init_hyper_fixed)
            if op == '*':
                ker = ker * factor
            else:
                ker = ker + factor
    return ker

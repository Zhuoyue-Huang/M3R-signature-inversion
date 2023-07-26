"""Signature inversion adapted from a existing package of orthogonal polynomials generator. Link: https://github.com/j-jith/orthopoly#id1"""
import scipy.integrate as integrate
import numpy as np
import numpy.polynomial.polynomial as nppoly
from free_lie_algebra import *
from numbers import Number
import unittest

class OrthoPoly(object):

    def __init__(self, measure, **kwargs):
        self.measure = measure
        self.measure_args = kwargs.get('margs', None)

        self.poly = []
        self.order = None
        self.intlims = kwargs.get('intlims', [-1, 1])

        self.alpha = None
        self.beta = None

    def gen_poly(self, n):
        self.order = n

        # zeroth polynomial
        self.poly = [nppoly.polyone]
        alpha = [self._get_alpha(self.poly[0])]
        beta = [self.integrate(self.poly[0], self.intlims[0], self.intlims[1])]

        # first polynomial
        self.poly.append(nppoly.polymulx(self.poly[0]))
        self.poly[1] = nppoly.polyadd(self.poly[1], -alpha[0]*self.poly[0])
        alpha.append(self._get_alpha(self.poly[1]))
        beta.append(self._get_beta(self.poly[1], self.poly[0]))

        # reccurence relation for other polynomials
        for i in range(2, n+1):
            p_i = nppoly.polymulx(self.poly[i-1])
            p_i = nppoly.polyadd(p_i, -alpha[i-1] * self.poly[i-1])
            p_i = nppoly.polyadd(p_i, -beta[i-1] * self.poly[i-2])

            self.poly.append(p_i)

            alpha.append(self._get_alpha(self.poly[i]))
            beta.append(self._get_beta(self.poly[i], self.poly[i-1]))
        
        self.alpha = alpha
        self.beta = beta

        # normalise polynomials
        self.poly[0] = self.poly[0]/np.sqrt(beta[0])
        for i in range(1, len(self.poly)):
            # self.poly[i][self.poly[i]!=0] = self.normalise(self.poly[i][self.poly[i]!=0], beta[:i+1])
            self.poly[i] = self.poly[i] / np.prod(np.sqrt(beta[:i+1]))
    
    def a(self, path, N):
        lim1, lim2 = self.intlims
        a_arr = np.zeros(N)
        if self.order is None or self.order-1<N:
            self.gen_poly(N-1)
        if callable(path):
            for n in range(N):
                a_arr[n] = self.integrate(self.poly[n], lim1, lim2, path)
            return a_arr
        else:
            t_grid = np.linspace(lim1, lim2, len(path))
            if self.measure_args:
                for n in range(N):
                    a_arr[n] = np.trapz(path * nppoly.polyval(t_grid, self.poly[n]) * 
                                        self.measure(t_grid, self.measure_args), t_grid)
            else:
                for n in range(N):
                    a_arr[n] = np.trapz(path * nppoly.polyval(t_grid, self.poly[n]) * 
                                        self.measure(t_grid), t_grid)
            return a_arr

    def path_eval(self, t, path, N):
        recon = self.a(path, N) @ self.eval(t)[:N]
        return recon

    def eval(self, t, **kwargs):
        n = kwargs.get('n', None)
        
        if n is None:
            if isinstance(t, Number):
                y = np.zeros(self.order+1)
            else:
                y = np.zeros((self.order+1, len(t)))
            for i in range(self.order+1):
                y[i] = nppoly.polyval(t, self.poly[i])
            return y
        else:
            return nppoly.polyval(t, self.poly[n])

    def integrate(self, p, lim1, lim2, other=None):
        if other is None:
            other = lambda t: 1.0
        if self.measure_args:
            return integrate.quad(lambda t: other(t) * nppoly.polyval(t, p) *
                    self.measure(t, self.measure_args), lim1, lim2)[0]
        else:
            return integrate.quad(lambda t: other(t) * nppoly.polyval(t, p) *
                    self.measure(t), lim1, lim2)[0]

    def normalise(self, p, current_beta):
        return np.sign(p)*np.exp(np.log(np.abs(p))-np.sum(np.log(current_beta))/2)

    def _get_alpha(self, p):
        p2 = nppoly.polypow(p, 2)
        xp2 = nppoly.polymulx(p2)

        return (self.integrate(xp2, self.intlims[0], self.intlims[1]) /
                self.integrate(p2, self.intlims[0], self.intlims[1]))

    def _get_beta(self, p, p0):
        p2 = nppoly.polypow(p, 2)
        p02 = nppoly.polypow(p0, 2)

        return (self.integrate(p2, self.intlims[0], self.intlims[1]) /
                self.integrate(p02, self.intlims[0], self.intlims[1]))


class Sig2path(object):
    def __init__(self, measure, **kwargs):
        self.measure = measure
        self.measure_args = kwargs.get('margs', None)
        self.poly_class = OrthoPoly(measure, **kwargs)

        self.intlims = kwargs.get('intlims', [-1, 1])
        self.length = kwargs.get('length', 100)
        self.t_grid = kwargs.get('t_grid', np.linspace(self.intlims[0], self.intlims[1], self.length))
        self.intlims = [self.t_grid[0], self.t_grid[-1]]
        self.length = len(self.t_grid)

    def recover(self, t, path, N):
        if self.poly_class.order is None or self.poly_class.order-1<N:
            self.poly_class.gen_poly(N)
        s = self.sig(path, N)
        sum_polynomial = nppoly.polyzero
        for n, p in enumerate(self.poly_class.poly[:N]):
            sum_polynomial = nppoly.polyadd(sum_polynomial, self._a_sig(s, n)*p)
        return nppoly.polyval(t, sum_polynomial)

    def sig(self, path, N):
        if self.measure_args:
            path_aug_time = np.c_[self.t_grid, path*self.measure(self.t_grid, self.measure_args)]
        else:
            path_aug_time = np.c_[self.t_grid, path*self.measure(self.t_grid)]
        return signature_of_path_iisignature(path_aug_time, N+2)

    def _l(self, n):
        if n==0:
            A0 = self.poly_class.poly[0][0]
            return A0 * word2Elt('21') 
        elif n==1:
            A1, B1 = self.poly_class.poly[1][::-1]
            return (A1*self.intlims[0]+B1)*word2Elt('21') + A1*(word2Elt('211')+word2Elt('121'))
        else:
            An = 1 / self.poly_class.beta[n]**0.5
            Bn = -self.poly_class.alpha[n-1] / self.poly_class.beta[n]**0.5
            Cn = -(self.poly_class.beta[n-1] / self.poly_class.beta[n])**0.5
            l_prev, l_prev_prev = self._l(n-1), self._l(n-2)
            l_n = An*rightHalfShuffleProduct(word2Elt('1'), l_prev)\
                  + (An*self.intlims[0]+Bn)*l_prev\
                  + Cn*l_prev_prev
            return l_n

    def _a_sig(self, sig, n):
        return dotprod(self._l(n), sig)

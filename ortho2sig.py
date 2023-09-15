"""Signature inversion adapted from a existing package of orthogonal polynomials generator. Link: https://github.com/j-jith/orthopoly#id1"""
import scipy.integrate as integrate
import numpy as np
import numpy.polynomial.polynomial as nppoly
from free_lie_algebra import *
from numbers import Number

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
        """Generate orthonormal polynomials up to degree n."""
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
            self.poly[i] = self.poly[i] / np.prod(np.sqrt(beta[:i+1]))
    
    def a(self, path, N):
        """Calculate the coefficients of orthogonal polynomials up to degree n for a path."""
        lim1, lim2 = self.intlims
        a_arr = np.zeros(N+1)
        if self.order is None or self.order<N:
            self.gen_poly(N)
        if callable(path):
            for n in range(N):
                a_arr[n] = self.integrate(self.poly[n], lim1, lim2, path)
            return a_arr
        else:
            t_grid = np.linspace(lim1, lim2, len(path))
            if self.measure_args:
                for n in range(N+1):
                    a_arr[n] = np.trapz(path * nppoly.polyval(t_grid, self.poly[n]) * 
                                        self.measure(t_grid, self.measure_args), t_grid)
            else:
                for n in range(N+1):
                    a_arr[n] = np.trapz(path * nppoly.polyval(t_grid, self.poly[n]) * 
                                        self.measure(t_grid), t_grid)
            return a_arr

    def path_eval(self, t, path, N):
        """Reconstruct path at t by orthogonal polynomials up to degree N."""
        recon = self.a(path, N) @ self.eval(t)[:N+1]
        return recon

    def eval(self, t, **kwargs):
        """Evaluate t for orthogonal polynomials up to its order."""
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
        """Implement inner product."""
        if other is None:
            other = lambda t: 1.0
        if self.measure_args:
            return integrate.quad(lambda t: other(t) * nppoly.polyval(t, p) *
                    self.measure(t, self.measure_args), lim1, lim2)[0]
        else:
            return integrate.quad(lambda t: other(t) * nppoly.polyval(t, p) *
                    self.measure(t), lim1, lim2)[0]

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
        self.intlims = kwargs.get('intlims', [-1, 1])
        self.length = kwargs.get('length', 100)
        self.t_grid = kwargs.get('t_grid', np.linspace(self.intlims[0], self.intlims[1], self.length))
        self.intlims = [self.t_grid[0], self.t_grid[-1]]
        self.length = len(self.t_grid)

        self.measure = measure
        self.measure_args = kwargs.get('margs', None)
        self.poly_class = OrthoPoly(measure, intlims=self.intlims, margs=self.measure_args)
        depth = kwargs.get('depth', None)
        if depth:
            self.poly_class.gen_poly(depth)

    def path2path(self, t, path, N, dim=1):
        """Reconstruct path based on signature via orthogonal polynomials"""
        if self.poly_class.order is None or self.poly_class.order-1<N:
            self.poly_class.gen_poly(N)
        s = self.sig(path, N)
        return self.sig2path(t, N, s, dim=dim)

    def sig2path(self, t, N, sig, dim=1, return_func=False):
        """Invert signature via orthogonal polynomials."""
        if self.poly_class.order is None or self.poly_class.order-1<N:
            self.poly_class.gen_poly(N)
        if isinstance(t, Number):
            recon = np.zeros(dim)
        else:
            recon = np.zeros((dim, len(t)))
        if return_func:
            recon = []
        for i in range(dim):
            sum_polynomial = nppoly.polyzero
            coeff_arr = self._a_sig(sig, N, dim=i+1)
            for n, p in enumerate(self.poly_class.poly[:N+1]):
                sum_polynomial = nppoly.polyadd(sum_polynomial, coeff_arr[n]*p)
            if return_func:
                recon.append(sum_polynomial)
            else:
                recon[i] = nppoly.polyval(t, sum_polynomial)
        if return_func:
            return recon
        else:
            if dim==1:
                return recon[0]
            else:
                return recon.T

    def sig(self, path, N):
        """Compute N+2 truncated signature of time-augmented weighted path."""
        if self.measure_args:
            path_aug_time = np.c_[self.t_grid, path*self.measure(self.t_grid, self.measure_args)]
        else:
            path_aug_time = np.c_[self.t_grid, path*self.measure(self.t_grid)]
        return signature_of_path_iisignature(path_aug_time, N+2)

    def _l(self, n, dim=1, l_dict=None):
        if n==0:
            A0 = self.poly_class.poly[0][0]
            return A0 * word2Elt(f'{dim+1}1') 
        elif n==1:
            A1, B1 = self.poly_class.poly[1][::-1]
            return (A1*self.intlims[0]+B1)*word2Elt(f'{dim+1}1') + A1*(word2Elt(f'{dim+1}11')+word2Elt(f'1{dim+1}1'))
        else:
            An = 1 / self.poly_class.beta[n]**0.5
            Bn = -self.poly_class.alpha[n-1] / self.poly_class.beta[n]**0.5
            Cn = -(self.poly_class.beta[n-1] / self.poly_class.beta[n])**0.5
            if l_dict:
                l_prev, l_prev_prev = l_dict[n-1], l_dict[n-2]
            else:
                l_prev, l_prev_prev = self._l(n-1, dim=dim), self._l(n-2, dim=dim)
            l_n = An*rightHalfShuffleProduct(word2Elt('1'), l_prev)\
                  + (An*self.intlims[0]+Bn)*l_prev\
                  + Cn*l_prev_prev
            return l_n

    def _a_sig(self, sig, N, dim=1):
        l_dict = {}
        coeff_arr = np.zeros(N+1)
        for n in range(N+1):
            l_dict[n] = self._l(n, dim=dim, l_dict=l_dict)
            coeff_arr[n] = dotprod(l_dict[n], sig)
        return coeff_arr

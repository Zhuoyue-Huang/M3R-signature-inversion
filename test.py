import unittest
import numpy as np
import numpy.polynomial.polynomial as nppoly
from ortho2sig import OrthoPoly, Sig2path

# test orthogonal polynomials
def make_test_case(measure, **kwargs):
    class TestOrthoPoly(unittest.TestCase):
        def setUp(self):
            self.measure = measure
            self.ortho_poly = OrthoPoly(measure=self.measure, **kwargs)

        def test_orthogonality(self):
            self.ortho_poly.gen_poly(10)
            for i in range(len(self.ortho_poly.poly)):
                for j in range(i+1, len(self.ortho_poly.poly)):
                    # check if integral of p_i * p_j * w is close to zero
                    integral = self.ortho_poly.integrate(nppoly.polymul(self.ortho_poly.poly[i],
                                                                        self.ortho_poly.poly[j]),
                                                         self.ortho_poly.intlims[0],
                                                         self.ortho_poly.intlims[1])
                    self.assertAlmostEqual(integral, 0, places=7)

        def test_normality(self):
            for p in self.ortho_poly.poly:
                # check if square root of integral of p_i^2 * w is close to 1
                norm = np.sqrt(self.ortho_poly.integrate(nppoly.polypow(p, 2),
                                                         self.ortho_poly.intlims[0],
                                                         self.ortho_poly.intlims[1]))
                self.assertAlmostEqual(norm, 1, places=7)

    TestOrthoPoly.__name__ = f'TestOrthoPolyWithMeasure{measure.__name__}'
    return TestOrthoPoly

Measure1 = lambda z: 1.0
intlims1 = [-1, 1]
Measure2 = lambda z: np.exp(-z**2/2)
intlims2 = [-1, 1]

test_case_1 = make_test_case(Measure1, intlims=intlims1)
test_case_2 = make_test_case(Measure2, intlims=intlims2)

if __name__ == '__main__':
    unittest.main()

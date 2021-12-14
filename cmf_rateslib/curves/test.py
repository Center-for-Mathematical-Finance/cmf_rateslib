from typing import List
import unittest
import numpy as np
import interpolator as interp
from zero_curve import ZeroCurve
from base_curve import BaseZeroCurve
import matplotlib.pyplot as plt

class Test_TestZeroCurve(unittest.TestCase):
    def test_zero_curve(self):
        maturities = np.array([1/4, 1/3, 1/2, 1.])
        rates = np.array([6.25, 6.35, 6.5, 6.7])
        zc = ZeroCurve(maturities, rates, rates_format="percent")        
        bc = BaseZeroCurve(maturities, rates)        

        expiry = np.array([3/4, 3/8, 2/3])
        res1 = bc.zero_rate(expiry)
        res2 = zc.zero_rate(expiry)
        self.assertEqual(np.linalg.norm(res1 - res2), 0)

        res1 = bc.df(expiry)
        res2 = zc.df(expiry)
        self.assertEqual(np.linalg.norm(res1 - res2), 0)

    def test_fwd_rateInterpolation(self):
        maturities = np.array([1/4, 1/3, 1/2, 1.])
        rates = np.array([0.625, 0.635, 0.65, 0.67])
        tenor = np.array([1/5, 1/4])
        expiry = np.array([3/7, 4/7])

        #test continuous compounding, linear interpolation
        zc = ZeroCurve(maturities, rates)
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)

        rate1 = np.exp(r1 * expiry + r12 * tenor)
        rate2 = np.exp(r2 * (expiry + tenor))
        np.testing.assert_array_almost_equal(rate2, rate1)

        #test discret compounding, linear interpolation
        zc = ZeroCurve(maturities, rates, compound="discret")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)

        rate1 = (1 + r1) ** (expiry) * (1+r12) ** tenor
        rate2 = (1 + r2) ** (tenor + expiry)
        np.testing.assert_array_almost_equal(rate2, rate1)

        #test discret compounding, quadratic interpolation
        zc = ZeroCurve.changeInterp(zc, "quadratic")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)

        rate1 = (1 + r1) ** (expiry) * (1+r12) ** tenor
        rate2 = (1 + r2) ** (tenor + expiry)
        np.testing.assert_array_almost_equal(rate2, rate1)

        #test continuous compounding, quadratic interpolation
        zc = ZeroCurve(maturities, rates, interp_method="quadratic")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)


        rate1 = np.exp(r1 * expiry + r12 * tenor)
        rate2 = np.exp(r2 * (expiry + tenor))
        np.testing.assert_array_almost_equal(rate2, rate1)

    def test_fwd_dfInterpolation(self):
        maturities = np.array([1/4, 1/3, 1/2, 1.])
        rates = np.array([0.625, 0.635, 0.65, 0.67])
        tenor = np.array([1/5, 1/4])
        expiry = np.array([3/7, 4/7])

        #test continuous compounding, linear interpolation
        zc = ZeroCurve(maturities, rates, interp_method="log_linear")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)

        rate1 = np.exp(r1 * expiry + r12 * tenor)
        rate2 = np.exp(r2 * (expiry + tenor))
        np.testing.assert_array_almost_equal(rate2, rate1)

        #test discret compounding, linear interpolation
        zc = ZeroCurve(maturities, rates, compound="discret", interp_method="log_linear")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)

        rate1 = (1 + r1) ** (expiry) * (1+r12) ** tenor
        rate2 = (1 + r2) ** (tenor + expiry)
        np.testing.assert_array_almost_equal(rate2, rate1)

        #test discret compounding, quadratic interpolation
        zc = ZeroCurve.changeInterp(zc, "log_quadratic")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)

        rate1 = (1 + r1) ** (expiry) * (1+r12) ** tenor
        rate2 = (1 + r2) ** (tenor + expiry)
        np.testing.assert_array_almost_equal(rate2, rate1)

        #test continuous compounding, quadratic interpolation
        zc = ZeroCurve(maturities, rates, interp_method="log_quadratic")
        r2 = zc.zero_rate(expiry + tenor)
        r1 = zc.zero_rate(expiry)
        r12 = zc.fwd_rate(expiry, tenor)


        rate1 = np.exp(r1 * expiry + r12 * tenor)
        rate2 = np.exp(r2 * (expiry + tenor))
        np.testing.assert_array_almost_equal(rate2, rate1)
    
    def test_rollDown(self):
        maturities = np.array([1/4, 1/3, 1/2, 1.])
        rates = np.array([0.625, 0.635, 0.65, 0.67])
        expiry = np.array([3/7, 4/7])
        t = 0.25
        
        #check if rates1(maturities)
        zc = ZeroCurve(maturities, rates)
        rolled_zc = zc.roll(t)

        res1 = zc.zero_rate(expiry)
        res2 = rolled_zc.zero_rate(expiry + t)
        np.testing.assert_array_almost_equal(res1, res2)

    def test_bump(self):
        rates1 = np.array([0.6,   0.63,  0.64, 0.645]) 
        rates2 = np.array([0.625, 0.633, 0.66, 0.67])
        rates3 = np.array([0.628, 0.63,  0.64, 0.675])
        rates4 = np.array([0.628, 0.635, 0.65, 0.67])
        
        rates = [rates1, rates2, rates3, rates4]
        maturities = np.array([1/4, 1/3, 1/2, 1.])

        zc_arr: List[ZeroCurve]
        zc_arr = []
        for i in range(0, 4):
            zc_arr.append(ZeroCurve(maturities, rates[i], interp_method="quadratic"))
        cov = ZeroCurve.get_cov_matrix(zc_arr, maturities)
        dr = {1: 0.01, 3: 0.06}
        zc = zc_arr[-1]
        zc.zero_rate(maturities)

        res = zc.PCAbump(dr, cov, adj=0.01)

        dr = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        res = zc.PCAbump(dr, cov, adj=0.1)

        res1 = res.get_rates()
        res2 = zc.get_rates() + np.array(list(dr.values()))
        np.testing.assert_array_almost_equal(res1, res2)
        

        

        

class Test_TestInterpolator(unittest.TestCase):

    def test_quadratic(self):
        xp = np.array([1., 2., 3.])
        yp = np.array([4., 7., 15.])
        x = np.array([1.7, 2.3])
        f = interp.Interpolator(xp, yp, "quadratic")
        res = f.interp(x)
        self.assertAlmostEqual(np.linalg.norm(res - np.array([6.1, 8.35])), 0)
    def test_log_quadratic(self):
        xp = np.array([1., 5., 8., 16., 23.])
        yp = np.exp(np.array([-1., 1., 2., 3., -2.]))
        x = np.linspace(1, 23, 100)
        f = interp.Interpolator(xp, yp, "log_quadratic")
        res = f.interp(x)
        h = 1
        self.assertEqual(0, 0)


    def test_linear(self):
        xp = np.array([1., 2., 3.])
        yp = np.array([1., 2., 3.])
        x = 0.5
        f = interp.Interpolator(xp, yp, "linear")
        res = f.interp(x)
        self.assertAlmostEqual(res, np.array([1/2]))

        x = np.array([0.5, 1, 1.5, 5])
        res = f.interp(x)
        self.assertAlmostEqual(np.linalg.norm(x - res), 0)

    def test_log_linear(self):
        xp = np.exp(np.array([1., 2., 3.]))
        yp = np.exp(np.array([1., 2., 3.]))
        x = 1.7
        f = interp.Interpolator(xp, np.exp(yp), "log_linear")
        res = f.interp(x)
        self.assertAlmostEqual(res, res)


if __name__ == "__main__":
    unittest.main()
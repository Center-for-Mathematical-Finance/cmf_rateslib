from ..curves.base_curve import BaseZeroCurve
import numpy as np

class Interpolation2Pow:
    
    def __init__(self, x, y, start_df = 1):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.start_df = start_df
        self.splines = self.get_splines()
        
    def get_splines(self):
        
        splines = [Interpolation2PowBlock(self.x[0], self.x[1], self.y[0], self.y[1], self.start_df)]
        for i in range(1, len(self.x)-1):
            splines.append(Interpolation2PowBlock(self.x[i], self.x[i+1], self.y[i], self.y[i+1], splines[-1].df1))
        return splines
    
    def __call__(self, x):
        if x < self.x[0] or x > self.x[-1]:
            raise ValueError('Error, x value is out of bounds!')
        for func in self.splines:
            if func.check(x):
                return func(x)
        

class Interpolation2PowBlock:
    
    def __init__(self, x0, x1, f0, f1, df0):
        
        self.x0 = x0
        self.x1 = x1
        self.f0 = f0
        self.f1 = f1
        self.df0 = df0
        
        self.a, self.b, self.c = self.get_coef()
        self.func = lambda x: self.a*x**2 + self.b*x + self.c
        self.df1 = 2*self.a*self.x1 + self.b
    
    def __call__(self, x):
        return self.func(x)
    
    def get_coef(self):
        a = (self.f1 - self.df0*self.x1 + self.df0*self.x0 - self.f0)/(self.x1**2 + self.x0**2 - 2*self.x1*self.x0)
        c = self.f0 + a*self.x0**2 - self.df0*self.x0 
        b = self.df0 - 2*a*self.x0
        return a, b, c
    
    def check(self, x):
        return x>=self.x0 and x<=self.x1
    
class InterpolationLinearBlock:
    
    def __init__(self, x0, x1, f0, f1):
        self.x0 = x0
        self.x1 = x1
        self.f0 = f0
        self.f1 = f1
        self.a, self.b = self.get_coef()
        self.func = lambda x: self.a*x + self.b
        
    def __call__(self, x):
        return self.func(x)
    
    def get_coef(self):
        a = (self.f1 - self.f0)/(self.x1 - self.x0)
        b = self.x0 - a*self.x0
        return a, b
    
    def check(self, x):
        return x>=self.x0 and x<=self.x1
    
class InterpolationLinear:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.splines = self.get_splines()
    
    def get_splines(self):
        res = []
        for i in range(1, len(self.x)):
            res.append(InterpolationLinearBlock(self.x[i-1], self.x[i], self.y[i-1], self.y[i]))
        return res
    
    def __call__(self, x):
        if x < self.x[0] or x > self.x[-1]:
            raise ValueError('Error, x value is out of bounds!')
        for func in self.splines:
            if func.check(x):
                return func(x)

class ZeroCurve(BaseZeroCurve):
    
    def __init__(self, maturities, rates, **kwarg):
        super().__init__(maturities, rates)
        if kwarg.get('df_for_quadratic') is None:
            kwarg['df_for_quadratic'] = 1
        self.quadratic_interpolator = Interpolation2Pow(maturities, rates ,kwarg['df_for_quadratic'])
        self.linear_interpolation = InterpolationLinear(maturities, rates)
    
    def interpolate(self, mode, power, expirity, tenor = None):
        if power not in [1, 2]:
            raise ValueError('Unknown type of interpolation!')
        if mode not in ['zero_rate', 'df', 'forward']:
            raise ValueError('Unknown type of IR')
        if mode == 'forward' and tenor is None:
            raise ValueError("Tenor can't be None if mode is 'forward'")
        if mode == 'zero_rate':
            if power == 1:
                return self.linear_interpolation(expirity)
            elif power == 2:
                return self.quadratic_interpolator(expirity)
        elif mode == 'df':
            if power == 1:
                return np.exp(- self.linear_interpolation(expirity) * expiry)
            elif power == 2:
                return np.exp(- self.quadratic_interpolator(expirity) * expiry)
        elif mode == 'forward':
            if power == 1:
                return -np.log((np.exp(- self.linear_interpolation(expirity) * expiry)/np.exp(- self.linear_interpolation(expirity+tenor) * (expiry+tenor)))) / tenor
            elif power == 2:
                return -np.log((np.exp(- self.quadratic_interpolator(expirity) * expiry)/np.exp(- self.quadratic_interpolator(expirity+tenor) * (expiry+tenor)))) / tenor


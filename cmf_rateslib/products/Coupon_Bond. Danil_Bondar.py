#!/usr/bin/env python
# coding: utf-8

# In[212]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# In[205]:


# frequency = 1, 2, 4 
# expiry in years
class Coupon_Bond_cont:
    def get_price(self, coupon, face_value,int_rate,expiry,freq=1,asof = 0):
        total_coupons_pv=self.get_coupons_pv(face_value,coupon,int_rate,expiry,freq,asof)
        face_value_pv=self.get_face_value_pv(face_value,int_rate,expiry,asof)
        result = total_coupons_pv + face_value_pv
        return result
    
    @staticmethod
    def get_face_value_pv(face_value,int_rate, expiry,asof = 0):
        Time = expiry - asof
        fvpv = face_value *np.exp(-(int_rate*Time))
        return fvpv
    
    def get_coupons_pv(self, face_value,coupon, int_rate, expiry,freq=1,asof = 0):
        pv = 0
        Time=int(expiry*freq)
        dates = self.get_payment_dates(expiry, freq,asof)
        for period in dates:
            if (period >= asof):
                pv += self.get_coupon_pv(coupon,int_rate,period,freq,asof)
        return pv 
    
    @staticmethod
    def get_coupon_pv(coupon, int_rate,expiry,freq,asof = 0):
        pv = coupon * np.exp(-(int_rate*(expiry-asof))/freq)
        return pv
    
    def get_ytm (self, bond_price, face_value, coupon, expiry, freq=1,estimate = 0.01,asof = 0):
        get_yield = lambda int_rate: self.get_price(coupon,face_value,int_rate,expiry,freq,asof)-bond_price
        return optimize.newton(get_yield,estimate)
    
    #@staticmethod
    def get_DVO1 (self,bond_price, face_value, coupon, expiry, freq=1,estimate = 0.01,asof = 0):
        DV=expiry*face_value*np.exp(-self.get_ytm (bond_price, face_value, coupon, expiry,asof)* (expiry))
        dates = self.get_payment_dates(expiry, freq,asof)
        t = 0
        for period in dates:
            if (period >= asof):
                DV += (t+1) * coupon * np.exp(-self.get_ytm (bond_price, face_value, coupon, expiry)* (t+1))
            t+=1
        return DV * 0.01
    
    def get_Macaulay_Duration (self,bond_price, face_value, coupon, expiry, freq=1,estimate = 0.01,asof = 0):
        return self.get_DVO1 (bond_price, face_value, coupon, expiry, freq,estimate, asof)/ bond_price
    
    def get_cashflows(self,face_value, coupon, expiry, freq=1,asof = 0):
        cashflow = []
        dates = self.get_payment_dates(expiry, freq,asof)
        
        for period in dates[:-1]:
            if (period >= asof):
                cashflow.append(coupon*face_value)
        cashflow.append((coupon+1)*face_value)
        return cashflow
    
    @staticmethod
    def get_payment_dates (expiry, freq=1,asof = 0):
        dates = []
        dates.append(1/freq)
        while (dates[-1]<expiry):
            dates.append(dates[-1]+(1/freq))
        return dates
    
    def CashFlows_plot(self, bond_price, face_value,int_rate, coupon, expiry, freq=1,asof = 0):
        plt.figure(figsize = (15,7))
        dates = self.get_payment_dates(expiry, freq,asof)
        flows = self.get_cashflows(face_value, coupon, expiry, freq,asof)
        plt.plot(dates, flows)
        plt.title('Cash Flows')
        plt.show()
      
    def accrued_interest(self, face_value, int_rate,coupon, expiry, freq=1,asof = 0):
        dates = self.get_payment_dates(expiry, freq,asof)
        i=0
        if asof in dates:
            return 0
        else:
            while (dates[i] < asof):
                Closest_coupon_time = dates[i]
                i+=1
            return (int_rate/365) * (face_value*coupon) * (asof - dates[i-1]) 
        
    def get_convexity (self, bond_price, face_value, int_rate,coupon, expiry, freq=1,asof = 0):
        price =  self.get_price(coupon, face_value,int_rate,expiry,freq)
        price_up =  self.get_price(coupon, face_value,int_rate+0.01,expiry,freq)
        price_down =  self.get_price(coupon, face_value,int_rate-0.01,expiry,freq)
        
        # Calculate convexity of the bond
        convexity = (price_up+price_down-price*2)/(price*0.01**2)

        # Calculate dollar convexity
        dollar_convexity = convexity * price * 0.01**2
        
        return dollar_convexity
    
    def get_clean_price (self, face_value, coupon,int_rate, expiry, freq=1,asof = 0):
        Present_value = self.get_price(coupon, face_value,int_rate,expiry,freq,asof)
        Accrued_interest = self.accrued_interest(face_value,int_rate, coupon, expiry, freq,asof)
        return Present_value - Accrued_interest 
    
    def get_dirty_price (self, face_value,int_rate, coupon, expiry, freq=1,asof = 0):
        return self.get_price(coupon, face_value,int_rate,expiry,freq,asof)


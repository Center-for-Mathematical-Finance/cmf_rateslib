
from ..curves.base_curve import BaseZeroCurve
import numpy as np


class ZeroCurve(BaseZeroCurve):
	def df(self, expiry,mode = 'continuous',period = 1,unit='years'):
		if not ((mode in ['continuous','discrete']) or unit in ['years','months','days']):
			raise ValueError("Incorrect mode or unit")
		if(mode=='continuous'):
			return np.exp(- self.zero_rate(expiry) * expiry)
		else:
			if unit=='years' :
				return (1+self.zero_rate(expiry)*period)**(-np.floor(expiry/period))
			if unit=='months':
				return (1 + self.zero_rate(expiry) *period/12) ** (-np.floor(expiry*12 / period))
			if unit =='days':
				return (1 + self.zero_rate(expiry) * period/252) ** (-np.floor(expiry *252 / period))

	def fwd_rate(self, expiry,tenor,mode = 'continuous',frequency = 1,unit='years'):
		if not ((mode in ['continuous','discrete']) or unit in ['years','months','days']):
			raise ValueError("Incorrect mode or unit")

		if (mode == 'continuous'):
			return -np.log((self.df(expiry)/self.df(expiry + tenor))) / tenor
		else:
			return (self.df(expiry,mode,frequency,unit)/self.df(expiry+tenor,mode,frequency,unit))**tenor - 1

	def roll(self,t,mode = 'zero_rate',pow=1):
		new_rates =np.array([])
		new_maturities = self._maturities+t
		n_points_to_extrapolate = ((new_maturities)>self._maturities[-1]).sum()
		new_rates = np.append(new_rates,self.interpolate(new_maturities[:-n_points_to_extrapolate],mode,pow))
		new_rates = np.append(new_rates,self.extrapolate(new_maturities[-n_points_to_extrapolate:],mode,pow))
		return ZeroCurve(new_maturities,new_rates)

	def _get_spline_coeffs(self,maturities,rates):
		beta = (rates[1] - rates[0]) / (maturities[1] - maturities[0])
		a = []
		b = []
		c = []
		for i in range(len(maturities)-1):
			c.append((rates[i + 1] - rates[i]) / (maturities[i + 1] - maturities[i]) ** 2 - beta / (maturities[i + 1] - maturities[i]))
			b.append(beta - 2 * c[i] * maturities[i])
			a.append(rates[i] - c[i] * maturities[i] ** 2 - b[i] * maturities[i])
			beta = b[i] + 2 * c[i] * maturities[i+1]
		return a,b,c

	def interpolate(self, expiries, mode = 'zero_rate', pow =1):
		expiries=np.array(expiries)
		if not(mode in ['zero_rate','log_df','forward_rate_3month',
						'forward_rate_6month','forward_rate_12month']) or not(pow in [1,2]):
			raise ValueError("Incorrect mode or power")
		if np.any((expiries<self._maturities[0])*(expiries>self._maturities[-1])):
			raise ValueError("Expiry is not in maturities range. Use extrapolation")

		if pow==1:
			if(mode=='zero_rate'):
				return self.zero_rate(expiries)
			if(mode=='log_df'):
				return np.exp(np.interp(expiries, self._maturities, np.log(self._rates)))
			if(mode=='forward_rate_3month'):
				pass 

		if pow==2:
			out_rates = []
			n=np.zeros(len(expiries))

			if (mode == 'zero_rate'):
				a,b,c = self._get_spline_coeffs(self._maturities,self._rates)

				for i in range(len(expiries)):
					n=0
					while (self._maturities[n] < expiries[i]):
						n = n + 1
					out_rates.append(a[n-1]+b[n-1]*expiries[i] + c[n-1]*expiries[i]**2)
				return np.array(out_rates)

			if (mode == 'log_df'):
				a, b, c = self._get_spline_coeffs(self._maturities, np.log(self._rates))

				for i in range(len(expiries)):
					n=0
					while (self._maturities[n] < expiries[i]):
						n = n + 1
					out_rates.append(np.exp((a[n - 1] + b[n - 1] * expiries[i] + c[n - 1] * expiries[i] ** 2)))

				return np.array(out_rates)

	def extrapolate(self, expiries, mode = 'zero_rate', pow =1):

		if not(mode in ['zero_rate','log_df','forward_rate_3month',
						'forward_rate_6month','forward_rate_12month']) or not(pow in [1,2]):
			raise ValueError("Incorrect mode or power")
		if np.any((expiries>=self._maturities[0])*(expiries<=self._maturities[-1])):
			raise ValueError("Expiry is in maturities range. Use interpolation")

		out_rates=[]

		if pow==1:
			if(mode=='zero_rate'):
				for e in expiries:
					if(e<self._maturities[0]):
						out_rates.append(self._rates[0] + (self._rates[0] - self._rates[1]) /
											(self._maturities[0] - self._maturities[1]) * (e - self._maturities[0]))
					else:
						out_rates.append(self._rates[-1] + (self._rates[-1] - self._rates[-2]) /
									 (self._maturities[-1] - self._maturities[-2]) * (e - self._maturities[-1]))
				return np.array(out_rates)

			if(mode=='log_df'):
				for e in expiries:
					if (e < self._maturities[0]):
						out_rates.append(np.exp((self._rates[0] + (self._rates[0] - self._rates[1]) /
										 (self._maturities[0] - self._maturities[1]) * (e - self._maturities[0]))))
					else:
						out_rates.append(np.exp((self._rates[-1] + (self._rates[-1] - self._rates[-2]) /
										 (self._maturities[-1] - self._maturities[-2]) * (e - self._maturities[-1]))))
				return np.array(out_rates)

		if pow == 2:
			out_rates = []
			if (mode == 'zero_rate'):
				a, b, c = self._get_spline_coeffs(self._maturities, self._rates)

				for i in range(len(expiries)):
					if(expiries[i]<self._maturities[0]):
						out_rates.append(a[0] + b[0] * expiries[i] + c[0] * expiries[i] ** 2)
					else:
						out_rates.append(a[-1] + b[-1] * expiries[i] + c[-1] * expiries[i] ** 2)
				return np.array(out_rates)

			if (mode == 'log_df'):
				a, b, c = self._get_spline_coeffs(self._maturities, np.log(self._rates))

				for i in range(len(expiries)):
					if(expiries[i]<self._maturities[0]):
						out_rates.append(np.exp((a[0] + b[0] * expiries[i] + c[0] * expiries[i] ** 2)))
					else:
						out_rates.append(np.exp((a[-1] + b[-1] * expiries[i] + c[-1] * expiries[i] ** 2)))

				return np.array(out_rates)












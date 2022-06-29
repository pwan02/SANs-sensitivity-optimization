## Author: Peng Wan
## Date: 08/22/2021

import networkx as nx
import numpy as np
import math
import random
import time
import statistics
import scipy.special as sc

from GenerateGraph import RmGraphGen
from scipy.stats import norm
from scipy.stats import erlang
from scipy.stats import gamma
from scipy.stats import uniform
from pert import PERT
from copy import copy
from scipy.stats import weibull_min
import matplotlib.pyplot as plt 
from IPython.display import display, Math, Latex
from collections import deque 

################################### Distribution Dictionary #########################################
## Normal Distribution:                        1
## exponential Distribution:                   2
## Triangle Distribution:                      3
## Truncated Normal Distribution unbounded:    4
## Uniform distribution:                       5
## Gamma distribution:                         6
## Truncated Normal Distribution bounded:      7
## Triangle Distribution location              8
## PERT Distribution location  (0,a,2a)        9
#####################################################################################################

class DistFunctions:
	def __init__(self,M_rv,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,B33):
		self.M_rv = M_rv
		self.M_mu = M_mu
		self.M_sigma = M_sigma
		self.M_unif_scal = M_unif_scal
		self.M_unif_loca = M_unif_loca
		### Edited on 09/02/2021 ###
		#self.M_tri_a = M_tri_a
		#self.M_tri_b = M_tri_b
		#self.M_tri_c = M_tri_c
		self.M_tri_a = M_tnormal_a
		self.M_tri_b = M_tnormal_b
		self.M_tri_c = M_mu
		self.M_gamma_shap = M_gamma_shap
		self.M_gamma_scal = M_gamma_scal
		self.M_tnormal_a = M_tnormal_a
		self.M_tnormal_b = M_tnormal_b
		self.M_theta = M_theta
		self.B33 = B33
		self.Mdst = Mdst


	### Density Function and CDF ###
	def pdf(self,ind_i,ind_j):
		rv = self.M_rv[in_i,ind_j]
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)
		if ind_distr == 2:
			rt = (1/mu)*np.exp((-1/mu)*rv)
		if ind_distr == 3 or ind_distr == 7:
			if rv < tri_a or rv > tri_b:
				rt = 0
			elif rv >= tri_a and rv <= tri_c:
				rt = (2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c))
			elif rv > tri_b:
				rt = 1
		if ind_distr == 4:
			rt = norm.pdf((rv-mu)/sigma)/(sigma*(norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma)))
		if ind_distr == 5:
			if rv < unif_loca or rv > unif_loca + unif_scal:
				rt = 0
			else:
				rt = 1/unif_scal
		if ind_distr == 6:
			rt = gamma.pdf(rv,gamma_shap,loc=0,scale=gamma_scal)
		if ind_distr == 8:
			if rv < 0 or rv > 2*theta:
				rt = 0
			elif rv >= 0 and rv <= theta:
				rt = rv/(theta**2)
			elif rv > theta and rv <= 2*theta:
				rt = (2*theta - rv)/(theta**2)
		if ind_distr == 9:
			if rv >= 0 and rv <= 2*theta:
				rt = (rv**2*(2*theta-rv)**2)/(self.B33*32*theta**5)
			else:
				rt = 0
		return rt

	def cdf(self,ind_i,ind_j):
		rv = self.M_rv[in_i,ind_j]
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = norm.cdf((rv-mu)/sigma)
		if ind_distr == 2:
			rt = 1 - np.exp((-1/mu)*rv)
		if ind_distr == 3 or ind_distr == 7:
			if rv <= tri_a or rv >= tri_b:
				rt = 0
			elif rv > tri_a and rv <= tri_c:
				rt = ((rv-tri_a)**2)/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv < tri_b:
				rt = 1 - ((tri_b-rv)**2)/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4:
			rt = (norm.cdf((rv - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma))/(norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma))
		if ind_distr == 5:
			if rv > unif_loca + unif_scal:
				rt = 1
			elif rv >= unif_loca and rv <= unif_loca + unif_scal:
				rt = (rv - unif_loca)/unif_scal 
			else:
				rt = 0
		if ind_distr == 6:
			rt = gamma.cdf(rv,gamma_shap,loc=0,scale=gamma_scal)
		if ind_distr == 8:
			if rv <= 0:
				rt = 0
			elif rv > 0 and rv <= theta:
				rt = 0.5*(rv/theta)**2
			elif rv > theta and rv < 2*theta:
				rt = 1 - 0.5*((2*theta - rv)/theta)**2
			elif rv >= 2*theta:
				rt = 1
        if ind_distr == 9:
        	pt = PERT(0,theta,2*theta)
        	rt = pt.cdf(rv)
		return rt

	def cdf2(self,rv,ind_i,ind_j):
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = norm.cdf((rv-mu)/sigma)
		if ind_distr == 2:
			rt = 1 - np.exp((-1/mu)*rv)
		if ind_distr == 3 or ind_distr == 7:
			if rv <= tri_a or rv >= tri_b:
				rt = 0
			elif rv > tri_a and rv <= tri_c:
				rt = ((rv-tri_a)**2)/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv < tri_b:
				rt = 1 - ((tri_b-rv)**2)/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4:
			rt = (norm.cdf((rv - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma))/(norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma))
		if ind_distr == 5:
			if rv > unif_loca + unif_scal:
				rt = 1
			elif rv >= unif_loca and rv <= unif_loca + unif_scal:
				rt = (rv - unif_loca)/unif_scal 
			else:
				rt = 0
		if ind_distr == 6:
			rt = gamma.cdf(rv,gamma_shap,loc=0,scale=gamma_scal)
		if ind_distr == 8:
			if rv <= 0:
				rt = 0
			elif rv > 0 and rv <= theta:
				rt = 0.5*(rv/theta)**2
			elif rv > theta and rv < 2*theta:
				rt = 1 - 0.5*((2*theta - rv)/theta)**2
			elif rv >= 2*theta:
				rt = 1
        if ind_distr == 9:
        	pt = PERT(0,theta,2*theta)
        	rt = pt.cdf(rv)
		return rt
	
	#### Derivative w.r.t. parameter of pdf and cdf ####
	### Derivative of parameters ###
	def pdf_dpara(self,ind_i,ind_j):
		rv = self.M_rv[in_i,ind_j]
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*((rv-mu)/(sigma**2))
		if ind_distr == 2:
			rt = ((-1)/(mu**2))*np.exp(rv/(-mu))+(rv/(mu)**3)*np.exp((-rv)/mu)
		if ind_distr == 3 or ind_distr == 7:
			if rv >= tri_a and rv <= tri_c:
				rt = (-2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a)**2)
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c)**2)
		if ind_distr == 4:
			Z = norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma)
			sigma_inv = 1/sigma
			dfmu = (1/(math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*((rv-mu)/(sigma**2))
			dFb = (-sigma_inv)*norm.pdf((tnormal_b - mu)/sigma)
			dFa = (-sigma_inv)*norm.pdf((tnormal_a - mu)/sigma)
			rt = sigma_inv*(dfmu/Z - (norm.pdf((rv - mu)/sigma)*(dFb - dFa))/(Z**2))
		if ind_distr == 8:
			if rv < 0 or rv > 2*theta:
				rt = 0
			elif rv >= 0 and rv <= theta:
				rt = -2*rv*(theta**(-3))
			elif rv > theta and rv <= 2*theta:
				rt = (2*(rv-theta))*(theta**(-3))
		if ind_distr == 9:
			if rv >= 0 and rv <= 2*theta:
				rt = (rv**2*(2*theta-rv)*(5*rv - 6*theta))/(self.B33*32*theta**6)
			else:
				rt = 0
		return rt

	def cdf_dpara(self,ind_i,ind_j):
		rv = self.M_rv[in_i,ind_j]
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = -(1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)
		if ind_distr == 2:
			rt = ((-1)/(mu**2))*np.exp(rv/(-mu))+(rv/(mu)**3)*np.exp((-rv)/mu)
		if ind_distr == 3 or ind_distr == 7:
			if rv >= tri_a and rv <= tri_c:
				rt = (-2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a)**2)
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c)**2)
		if ind_distr == 4:
			Z = norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma)
			sigma_inv = 1/sigma
			dfmu = (1/(math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*((rv-mu)/(sigma**2))
			dFb = (-sigma_inv)*norm.pdf((tnormal_b - mu)/sigma)
			dFa = (-sigma_inv)*norm.pdf((tnormal_a - mu)/sigma)
			rt = sigma_inv*(dfmu/Z - (norm.pdf((rv - mu)/sigma)*(dFb - dFa))/(Z**2))
        if ind_distr == 8:
			if rv < 0 or rv > 2*theta:
				rt = 0
			elif rv >= 0 and rv <= theta:
				rt = -rv**2*(theta**(-3))
			elif rv > theta and rv <= 2*theta:
				rt = 2*rv*(2*theta - rv)*(theta**(-3))
		if ind_distr == 9:
			if rv >= 0 and rv <= 2*theta:
				rt = -(1/8)*rv**3*(1-rv/(2*theta))**2*theta**(-4)
			else:
				rt = 0

		return rt

	### Derivative of x ###
	def pdf_dx(self,ind_i,ind_j):
		rv = self.M_rv[in_i,ind_j]
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*(-(rv-mu)/(sigma**2))
		if ind_distr == 2:
			rt = ((-1)/(mu**2))*np.exp(rv/(-mu))
		if ind_distr == 3 or ind_distr == 7:
			if rv >= tri_a and rv <= tri_c:
				rt = 2/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv <= tri_b:
				rt = (-2)/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4:
			Z = norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma)
			sigma_inv = 1/sigma
			dfx = (1/(math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*(-(rv-mu)/(sigma**2))
			rt = (sigma_inv*dfx)/Z 
        if ind_distr == 8:
			if rv < 0 or rv > 2*theta:
				rt = 0
			elif rv >= 0 and rv <= theta:
				rt = theta**(-2)
			elif rv > theta and rv <= 2*theta:
				rt = -theta**(-2)
		if ind_distr == 9:
			if rv >= 0 and rv <= 2*theta:
				rt = (4*rv*(2*theta-rv)*(theta - rv))/(self.B33*32*theta**5)
			else:
				rt = 0

		return rt

	def cdf_dx(self,ind_i,ind_j):
		rv = self.M_rv[in_i,ind_j]
		mu = self.M_mu[in_i,ind_j] 
		sigma = self.M_sigma[in_i,ind_j] 
		unif_scal = self.M_unif_scal[in_i,ind_j] 
		unif_loca = self.M_unif_loca[in_i,ind_j] 
		tri_a = self.M_tri_a[in_i,ind_j] 
		tri_b = self.M_tri_b[in_i,ind_j] 
		tri_c = self.M_tri_c[in_i,ind_j] 
		gamma_shap = self.M_gamma_shap[in_i,ind_j] 
		gamma_scal = self.M_gamma_scal[in_i,ind_j] 
		tnormal_a = self.M_tnormal_a[in_i,ind_j] 
		tnormal_b = self.M_tnormal_b[in_i,ind_j] 
		ind_distr = self.Mdst[in_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)
		if ind_distr == 2:
			rt = (1/mu)*np.exp((-1/mu)*rv)
		if ind_distr == 3 or ind_distr == 7:
			if rv < tri_a or rv > tri_b:
				rt = 0
			elif rv >= tri_a and rv <= tri_c:
				rt = (2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4:
			rt = norm.pdf((rv-mu)/sigma)/(sigma*(norm.cdf((tnormal_b - mu)/sigma) - norm.cdf((tnormal_a - mu)/sigma)))
		if ind_distr == 5:
			if rv < unif_loca or rv > unif_loca + unif_scal:
				rt = 0
			else:
				rt = 1/unif_scal
		if ind_distr == 6:
			rt = gamma.pdf(rv,gamma_shap,loc=0,scale=gamma_scal)
        if ind_distr == 8:
			if rv < 0 or rv > 2*theta:
				rt = 0
			elif rv >= 0 and rv <= theta:
				rt = rv*(theta**(-2))
			elif rv > theta and rv <= 2*theta:
				rt = (2*theta - rv)*(theta**(-2))
		if ind_distr == 9:
			if rv >= 0 and rv <= 2*theta:
				rt = (1/8)*((rv/theta)*(1-rv/(2*theta)))**2
			else:
				rt = 0

		return rt

	####### LR part #######
	def lr_part(self,i,j):
		theta = self.M_theta[i,j]
		rv = self.simu[i,j]
		if self.Mdst[i,j] == 1 or self.Mdst[i,j] == 7:
			lr_part = (self.simu[i,j] - self.M_mu[i,j])/(self.M_sigma[i,j]**2)
		if self.Mdst[i,j] == 2:
			lr_part = (self.simu[i,j]/self.M_mu[i,j] - 1)/self.M_mu[i,j]
		#### here triangle distribution takes the parameter c as a location parameter, i.e., both a, b, c changes the same amount ####
		if self.Mdst[i,j] == 3:
			if self.simu[i,j] >= self.M_tri_a[i,j] and self.simu[i,j] <= self.M_tri_c[i,j]:
				lr_part = (-1)/(self.simu[i,j] - self.M_tri_a[i,j])
			elif self.simu[i,j] > self.M_tri_c[i,j] and self.simu[i,j] <= self.M_tri_b[i,j]:
				lr_part = 1/(self.M_tri_b[i,j] - self.simu[i,j])
		if self.Mdst[i,j] == 4:
			lr_part = ((self.simu[i,j] - self.M_mu[i,j])/self.M_sigma[i,j]**2) + (self.pdf2(self.M_tnormal_b[i,j],i,j)-self.pdf2(self.M_tnormal_a[i,j],i,j))/((self.M_sigma[i,j])*(self.cdf2(self.M_tnormal_b[i,j],i,j) - self.cdf2(self.M_tnormal_a[i,j],i,j)))
		if self.Mdst[i,j] == 8:
			if rv < 0 or rv > 2*theta:
				lr_part = 0
			elif rv >= 0 and rv <= theta:
				lr_part = rv**(-1)
			elif rv > theta and rv <= 2*theta:
				lr_part = (rv - 2*theta)**(-1)
		if self.Mdst[i,j] == 9:
			if rv < 0 or rv > 2*theta::
				lr_part = 0
			else:
				lr_part = (4*(theta - rv))/(rv*(2*theta - rv))
        return lr_part


if __name__ == "__main__":
	gnwk = RmGraphGen(10,20)
	G = gnwk.generate_graph_am()
	print(G)


	
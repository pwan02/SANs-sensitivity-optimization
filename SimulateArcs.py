## Author: Peng Wan
## Date: 08/23/2021

import networkx as nx
import numpy as np
import math
import random
import time
import statistics
import scipy.special as sc

from NetGraph import RmGraphGen
from NetGraph import GraphInform

from scipy.stats import norm
from scipy.stats import erlang
from scipy.stats import gamma
from scipy.stats import uniform
from pert import PERT
import copy
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
## Triangle Distribution scale                 8
## PERT Distribution scale                     9
#####################################################################################################

class ParaGen:
	def __init__(self,Arc,Node,mu_b,mu_c,sigma_b,sigma_c,unif_loca_b,unif_loca_c,unif_scal_b,unif_scal_c,tri_left_b,tri_left_c,tri_right_b,tri_right_c,gamma_shap_b,gamma_shap_c,gamma_scal_b,gamma_scal_c,tnormal_left_b,tnormal_left_c,tnormal_right_b,tnormal_right_c,theta_b,theta_c,cost_b,cost_c,nonzero_list):
		self.Arc = Arc
		self.Node = Node
		self.mu_b = mu_b
		self.mu_c = mu_c
		self.sigma_b = sigma_b
		self.sigma_c = sigma_c
		self.unif_loca_b = unif_loca_b
		self.unif_loca_c = unif_loca_c
		self.unif_scal_b = unif_scal_b
		self.unif_scal_c = unif_scal_c
		self.tri_left_b = tri_left_b
		self.tri_left_c = tri_left_c
		self.tri_right_b = tri_right_b
		self.tri_right_c = tri_right_c
		self.gamma_shap_b = gamma_shap_b
		self.gamma_shap_c = gamma_shap_c
		self.gamma_scal_b = gamma_scal_b
		self.gamma_scal_c = gamma_scal_c
		self.tnormal_left_b = tnormal_left_b
		self.tnormal_left_c = tnormal_left_c
		self.tnormal_right_b = tnormal_right_b
		self.tnormal_right_c = tnormal_right_c
		self.nonzero_list = nonzero_list
		self.theta_b = theta_b
		self.theta_c = theta_c

	def paragenerate(self):
		n = self.Node
		m = self.Arc
		M_mu = np.zeros([n,n]) 
		M_sigma = np.zeros([n,n])
		M_unif_loca = np.zeros([n,n])
		M_unif_scal = np.zeros([n,n])
		M_tri_a = np.zeros([n,n])
		M_tri_b = np.zeros([n,n])
		M_tri_c = np.zeros([n,n])
		M_gamma_scal = np.zeros([n,n])
		M_gamma_shap = np.zeros([n,n])
		M_tnormal_a = np.zeros([n,n])
		M_tnormal_b = np.zeros([n,n])
		M_theta = np.zeros([n,n])

		for k in range(m):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			M_mu[i,j] = uniform.rvs(loc = self.mu_b,scale = self.mu_c)
			M_sigma[i,j] = uniform.rvs(loc = self.sigma_b,scale = self.sigma_c)*M_mu[i,j]
			M_unif_loca[i,j] = uniform.rvs(loc = self.unif_loca_b, scale = self.unif_loca_c)
			M_unif_scal[i,j] = uniform.rvs(loc = self.unif_scal_b, scale = self.unif_scal_c)
			M_tri_c[i,j] = M_mu[i,j]
			M_tri_a[i,j] = (1 - uniform.rvs(loc = self.tri_left_b, scale = self.tri_left_c))*M_mu[i,j]
			M_tri_b[i,j] = (1 + uniform.rvs(loc = self.tri_right_b, scale = self.tri_right_c))*M_mu[i,j]
			M_gamma_shap[i,j] = uniform.rvs(loc = self.gamma_shap_b, scale = self.gamma_shap_c)
			M_gamma_scal[i,j] = uniform.rvs(loc = self.gamma_scal_b, scale = self.gamma_scal_c)
			M_tnormal_a[i,j] = (1 - uniform.rvs(loc = self.tnormal_left_b, scale = self.tnormal_left_c))*M_mu[i,j]
			M_tnormal_b[i,j] = (1 + uniform.rvs(loc = self.tnormal_right_b, scale = self.tnormal_right_c))*M_mu[i,j]
			M_theta[i,j] = uniform.rvs(loc = self.theta_b,scale = self.theta_c)

		return M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta

	def cost_generate(self):
		cost = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			cost[i,j] = uniform.rvs(loc = self.cost_b, scale = self.cost_c)
		return cost

	## Generate the distribution Matrix
	def dstr_gen(self,sublist):
		m = self.Arc
		n = self.Node
		lst_len = len(sublist)
		Dst = np.zeros([n,n])
		for k in range(m):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if lst_len > 1:
				rand_num = random.randint(0,lst_len-1)
				Dst[i,j] = int(sublist[rand_num])
			elif lst_len == 1:
				Dst[i,j] = int(sublist[0])
		return Dst

	def generate_lowbd_matrix(self):
		M_lbd = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			M_lbd[i,j] = 1 - round(uniform.rvs(loc = 0.5, scale = 0.3),1)
		return M_lbd

	def generate_cost(self):
		cost_Mt = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			cost_Mt[i,j] = round(uniform.rvs(loc = 5, scale = 45))
		return cost_Mt


class DstrSimu:
	def __init__(self,Arc,Node,M_rv,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,nonzero_list):

		## Mdst is the matrix of distributions an n by n matrix with inputs integers ranges from 1 to 7 ##
		self.Arc = Arc
		self.Node = Node
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
		self.Mdst = Mdst
		self.nonzero_list = nonzero_list
		self.B33 = sc.beta(3,3)


	### Density Function and CDF ###
	def pdf(self,ind_i,ind_j):
		rv = self.M_rv[ind_i,ind_j]
		mu = self.M_mu[ind_i,ind_j] 
		sigma = self.M_sigma[ind_i,ind_j] 
		unif_scal = self.M_unif_scal[ind_i,ind_j] 
		unif_loca = self.M_unif_loca[ind_i,ind_j] 
		tri_a = self.M_tri_a[ind_i,ind_j] 
		tri_b = self.M_tri_b[ind_i,ind_j] 
		tri_c = self.M_tri_c[ind_i,ind_j] 
		gamma_shap = self.M_gamma_shap[ind_i,ind_j] 
		gamma_scal = self.M_gamma_scal[ind_i,ind_j] 
		tnormal_a = self.M_tnormal_a[ind_i,ind_j] 
		tnormal_b = self.M_tnormal_b[ind_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[ind_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)
		if ind_distr == 2:
			rt = (1/mu)*np.exp((-1/mu)*rv)
		if ind_distr == 3:
			if rv < tri_a or rv > tri_b:
				rt = 0
			elif rv >= tri_a and rv <= tri_c:
				rt = (2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4 or ind_distr == 7:
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
		rv = self.M_rv[ind_i,ind_j]
		mu = self.M_mu[ind_i,ind_j] 
		sigma = self.M_sigma[ind_i,ind_j] 
		unif_scal = self.M_unif_scal[ind_i,ind_j] 
		unif_loca = self.M_unif_loca[ind_i,ind_j] 
		tri_a = self.M_tri_a[ind_i,ind_j] 
		tri_b = self.M_tri_b[ind_i,ind_j] 
		tri_c = self.M_tri_c[ind_i,ind_j] 
		gamma_shap = self.M_gamma_shap[ind_i,ind_j] 
		gamma_scal = self.M_gamma_scal[ind_i,ind_j] 
		tnormal_a = self.M_tnormal_a[ind_i,ind_j] 
		tnormal_b = self.M_tnormal_b[ind_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[ind_i,ind_j] 

		if ind_distr == 1:
			rt = norm.cdf((rv-mu)/sigma)
		if ind_distr == 2:
			rt = 1 - np.exp((-1/mu)*rv)
		if ind_distr == 3:
			if rv <= tri_a or rv >= tri_b:
				rt = 0
			elif rv > tri_a and rv <= tri_c:
				rt = ((rv-tri_a)**2)/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv < tri_b:
				rt = 1 - ((tri_b-rv)**2)/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4 or ind_distr == 7:
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
		rv = self.M_rv[ind_i,ind_j]
		mu = self.M_mu[ind_i,ind_j] 
		sigma = self.M_sigma[ind_i,ind_j] 
		unif_scal = self.M_unif_scal[ind_i,ind_j] 
		unif_loca = self.M_unif_loca[ind_i,ind_j] 
		tri_a = self.M_tri_a[ind_i,ind_j] 
		tri_b = self.M_tri_b[ind_i,ind_j] 
		tri_c = self.M_tri_c[ind_i,ind_j] 
		gamma_shap = self.M_gamma_shap[ind_i,ind_j] 
		gamma_scal = self.M_gamma_scal[ind_i,ind_j] 
		tnormal_a = self.M_tnormal_a[ind_i,ind_j] 
		tnormal_b = self.M_tnormal_b[ind_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[ind_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*((rv-mu)/(sigma**2))
		if ind_distr == 2:
			rt = ((-1)/(mu**2))*np.exp(rv/(-mu))+(rv/(mu)**3)*np.exp((-rv)/mu)
		if ind_distr == 3:
			if rv >= tri_a and rv <= tri_c:
				rt = (-2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a)**2)
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c)**2)
		if ind_distr == 4 or ind_distr == 7:
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
		rv = self.M_rv[ind_i,ind_j]
		mu = self.M_mu[ind_i,ind_j] 
		sigma = self.M_sigma[ind_i,ind_j] 
		unif_scal = self.M_unif_scal[ind_i,ind_j] 
		unif_loca = self.M_unif_loca[ind_i,ind_j] 
		tri_a = self.M_tri_a[ind_i,ind_j] 
		tri_b = self.M_tri_b[ind_i,ind_j] 
		tri_c = self.M_tri_c[ind_i,ind_j] 
		gamma_shap = self.M_gamma_shap[ind_i,ind_j] 
		gamma_scal = self.M_gamma_scal[ind_i,ind_j] 
		tnormal_a = self.M_tnormal_a[ind_i,ind_j] 
		tnormal_b = self.M_tnormal_b[ind_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[ind_i,ind_j] 

		if ind_distr == 1:
			rt = -(1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)
		if ind_distr == 2:
			rt = ((-1)/(mu**2))*np.exp(rv/(-mu))+(rv/(mu)**3)*np.exp((-rv)/mu)
		if ind_distr == 3:
			if rv >= tri_a and rv <= tri_c:
				rt = (-2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a)**2)
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c)**2)
		if ind_distr == 4 or ind_distr == 7:
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
		rv = self.M_rv[ind_i,ind_j]
		mu = self.M_mu[ind_i,ind_j] 
		sigma = self.M_sigma[ind_i,ind_j] 
		unif_scal = self.M_unif_scal[ind_i,ind_j] 
		unif_loca = self.M_unif_loca[ind_i,ind_j] 
		tri_a = self.M_tri_a[ind_i,ind_j] 
		tri_b = self.M_tri_b[ind_i,ind_j] 
		tri_c = self.M_tri_c[ind_i,ind_j] 
		gamma_shap = self.M_gamma_shap[ind_i,ind_j] 
		gamma_scal = self.M_gamma_scal[ind_i,ind_j] 
		tnormal_a = self.M_tnormal_a[ind_i,ind_j] 
		tnormal_b = self.M_tnormal_b[ind_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[ind_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)*(-(rv-mu)/(sigma**2))
		if ind_distr == 2:
			rt = ((-1)/(mu**2))*np.exp(rv/(-mu))
		if ind_distr == 3:
			if rv >= tri_a and rv <= tri_c:
				rt = 2/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv <= tri_b:
				rt = (-2)/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4 or ind_distr == 7:
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
		rv = self.M_rv[ind_i,ind_j]
		mu = self.M_mu[ind_i,ind_j] 
		sigma = self.M_sigma[ind_i,ind_j] 
		unif_scal = self.M_unif_scal[ind_i,ind_j] 
		unif_loca = self.M_unif_loca[ind_i,ind_j] 
		tri_a = self.M_tri_a[ind_i,ind_j] 
		tri_b = self.M_tri_b[ind_i,ind_j] 
		tri_c = self.M_tri_c[ind_i,ind_j] 
		gamma_shap = self.M_gamma_shap[ind_i,ind_j] 
		gamma_scal = self.M_gamma_scal[ind_i,ind_j] 
		tnormal_a = self.M_tnormal_a[ind_i,ind_j] 
		tnormal_b = self.M_tnormal_b[ind_i,ind_j] 
		theta = self.M_theta[ind_i,ind_j]
		ind_distr = self.Mdst[ind_i,ind_j] 

		if ind_distr == 1:
			rt = (1/(sigma*math.sqrt(2*math.pi)))*np.exp((-1/2)*((rv-mu)/sigma)**2)
		if ind_distr == 2:
			rt = (1/mu)*np.exp((-1/mu)*rv)
		if ind_distr == 3:
			if rv < tri_a or rv > tri_b:
				rt = 0
			elif rv >= tri_a and rv <= tri_c:
				rt = (2*(rv-tri_a))/((tri_b-tri_a)*(tri_c-tri_a))
			elif rv > tri_c and rv <= tri_b:
				rt = (2*(tri_b-rv))/((tri_b-tri_a)*(tri_b-tri_c))
		if ind_distr == 4 or ind_distr == 7:
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

	def tnormal_rvs(self,mu,sigma,a,b):
		u = uniform.rvs(loc = 0,scale = 1)
		rt = norm.ppf(norm.cdf(a) + u*(norm.cdf(b) - norm.cdf(a)))*sigma + mu
		return rt

class Simulate:
	def __init__(self,Arc,Node,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,nonzero_list):

		## Mdst is the matrix of distributions an n by n matrix with inputs integers ranges from 1 to 7 ##
		self.Arc = Arc
		self.Node = Node
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
		self.Mdst = Mdst
		self.nonzero_list = nonzero_list

	def SimuArcs(self):
		n = self.Node
		m = self.Arc
		simulate = np.zeros([n,n])
		distri_matrix = self.Mdst

		for k in range(m):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if distri_matrix[i,j] == 1:
				simulate[i,j] = np.random.normal(self.M_mu[i,j],self.M_sigma[i,j])
			if distri_matrix[i,j] == 2:
				simulate[i,j] = np.random.exponential(self.M_mu[i,j])
			if distri_matrix[i,j] == 3:
				simulate[i,j] = np.random.triangular(self.M_tri_a[i,j],self.M_tri_c[i,j],self.M_tri_b[i,j])
			if distri_matrix[i,j] == 4:
				a = 0
				b = float('inf')
				simulate[i,j] = self.tnormal_rvs(self.M_mu[i,j],self.M_sigma[i,j],a,b)
			if distri_matrix[i,j] == 5:
				simulate[i,j] = uniform.rvs(loc = self.M_unif_loca[i,j],scale = self.M_unif_scal[i,j])
			if distri_matrix[i,j] == 6:
				simulate[i,j] = gamma.rvs(self.M_gamma_shap[i,j],loc=0,scale = self.M_gamma_scal[i,j])
			if distri_matrix[i,j] == 7:
				simulate[i,j] = self.tnormal_rvs(self.M_mu[i,j],self.M_sigma[i,j],self.M_tnormal_a[i,j],self.M_tnormal_b[i,j])
			if distri_matrix[i,j] == 8:
				simulate[i,j] = np.random.triangular(0,self.M_theta[i,j],2*self.M_theta[i,j])
			if distri_matrix[i,j] == 9:
				pt = PERT(0,self.M_theta[i,j],2*self.M_theta[i,j])
				simulate[i,j] = pt.rvs(1)

		return simulate

if __name__ == "__main__":
	gnwk = RmGraphGen(10,20)
	Arc = 20
	Node = 10
	mu_b = 1
	mu_c = 14
	sigma_b = 0.25
	sigma_c = 0.33 - sigma_b
	unif_loca_b = 1
	unif_loca_c = 4
	unif_scal_b = 1
	unif_scal_c = 9
	tri_left_b = 0.25
	tri_left_c = 0.33 - tri_left_b
	tri_right_b = 0.25
	tri_right_c = 0.33 - tri_right_b
	gamma_shap_b = 1
	gamma_shap_c = 9
	gamma_scal_b = 0.5
	gamma_scal_c = 2
	tnormal_left_b = 0.25
	tnormal_left_c = 0.33 - tnormal_left_b
	tnormal_right_b = 0.25
	tnormal_right_c = 0.33 - tnormal_right_b
	G = gnwk.generate_graph_am()
	nonzero_list = np.nonzero(G)
	par = ParaGen(Arc,Node,mu_b,mu_c,sigma_b,sigma_c,unif_loca_b,unif_loca_c,unif_scal_b,unif_scal_c,tri_left_b,tri_left_c,tri_right_b,tri_right_c,gamma_shap_b,gamma_shap_c,gamma_scal_b,gamma_scal_c,tnormal_left_b,tnormal_left_c,tnormal_right_b,tnormal_right_c,nonzero_list)
	sublist = [3]
	print(par.dstr_gen(sublist))
	print(G)


	
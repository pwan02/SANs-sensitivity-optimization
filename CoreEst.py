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

from SimulateArcs import ParaGen
from SimulateArcs import Simulate
from SimulateArcs import DstrSimu

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

class GradientEst:

	def __init__(self,Arc,Node,simu,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,eps):
		self.Arc = Arc
		self.Node = Node
		self.simu = simu
		self.G = G
		self.dictionary = dictionary
		self.inform = inform
		self.net_struct = net_struct
		self.nonzero_list = nonzero_list
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
		self.eps = eps
		self.B33 = sc.beta(3,3)
	
	def plen_thd(self):
		simulate = self.simu
		dit = self.dictionary
		length = np.zeros(len(dit))
		for i in range(len(dit)):
			lt = 0
			for j in range(len(dit[i]) - 1):
				lt = lt + simulate[dit[i][j],dit[i][j+1]]
			length[i] = lt
		return length

	def retur_1stnonzero(self,vec):
		k = 0
		while 1:
			if vec[k] != 0:
				rt = k
				break
			else:
				k += 1
		return rt        

	def cal_thd(self):
		simulate = self.simu
		dit = self.dictionary
		inform = self.inform 

		threshold = np.zeros([self.Node,self.Node])
		length = self.plen_thd()
		sort_info = inform[np.argsort(-length),:,:]
		sort_length = length[np.argsort(-length)]
		
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			target = sort_info[:,i,j]
			rest = np.ones(len(target)) - target
			target_value = sort_length[self.retur_1stnonzero(target)]
			rest_value = sort_length[self.retur_1stnonzero(rest)]
			threshold[i,j] = max((rest_value - (target_value - simulate[i,j])),0)
		return threshold

	def cal_thd_Amatrix(self):
		simulate = self.simu
		dit = self.dictionary
		inform = self.inform 

		threshold = np.zeros([self.Node,self.Node])
		A = np.zeros([self.Node,self.Node])
		length = self.plen_thd()
		sort_info = inform[np.argsort(-length),:,:]
		sort_length = length[np.argsort(-length)]
		
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			target = sort_info[:,i,j]
			rest = np.ones(len(target)) - target
			target_value = sort_length[self.retur_1stnonzero(target)]
			rest_value = sort_length[self.retur_1stnonzero(rest)]
			A[i,j] = target_value - simulate[i,j]
			threshold[i,j] = max(rest_value - A[i,j],0)
		return threshold, A

	def ca_tac(self):
		ca_tac_est = np.zeros([self.Node,self.Node])
		distri = DstrSimu(self.Arc,self.Node,self.cal_thd(),self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]  
			ca_tac_est[i,j] = 1 - distri.cdf(i,j)
		return ca_tac_est

	def dca_tac(self):
		dca_tac_est = np.zeros([self.Node,self.Node])
		distri = DstrSimu(self.Arc,self.Node,self.cal_thd(),self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]  
			dca_tac_est[i,j] = - distri.cdf_dpara(i,j)
		return dca_tac_est 	
	## Calculating the node release time ##

	def ptime(self):
		n = self.Node
		simulate = self.simu
		mt = self.G
		t = np.zeros(n)
		node_time = np.zeros([n,n])
		for j in range(n):
			for i in range(n):
				if mt[i,j] != 0:
					node_time[j,i] = t[i]
			## make some changes here at 2020/10/24 19:39pm ##
			if np.count_nonzero(mt[:,j]) > 0:
				nonzerolist = np.nonzero(mt[:,j])[0]
				t[j] = max(simulate[nonzerolist,j] + node_time[j,nonzerolist])
			else:
				t[j] = 0
			## make some changes here at 2020/10/24 19:39pm ##
		return t

	## Calculating difference of node release time ##
	def release_diff_gen(self):
		n = self.Node
		t = self.ptime()
		diff = np.zeros([n,n])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			diff[i,j] = t[j] - t[i]
		return diff

	## To be used in release_indicator_gen ##
	def release_diff_gen2(self,t):
		n = self.Node
		diff = np.zeros([n,n])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			diff[i,j] = t[j] - t[i]
		return diff

	def ptime2(self,simulate):
		n = self.Node
		mt = self.G
		t = np.zeros(n)
		node_time = np.zeros([n,n])
		for j in range(n):
			for i in range(n):
				if mt[i,j] != 0:
					node_time[j,i] = t[i]
			## make some changes here at 2020/10/24 19:39pm ##
			if np.count_nonzero(mt[:,j]) > 0:
				nonzerolist = np.nonzero(mt[:,j])[0]
				t[j] = max(simulate[nonzerolist,j] + node_time[j,nonzerolist])
			else:
				t[j] = 0
			## make some changes here at 2020/10/24 19:39pm ##
		return t

	## Calculating pdf and cdf ##
	def distribute_list(self):
		n = self.Node
		pdf = np.zeros([n,n])
		cdf = np.zeros([n,n])
		distri = DstrSimu(self.Arc,self.Node,self.release_diff_gen(),self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			pdf[i,j] = distri.pdf(i,j)
			cdf[i,j] = distri.cdf(i,j)
		return pdf, cdf
	
	## Calculating the left part for CAC ##
	def critical_left(self):
		## n is the dimension of matrix, mt is the input matrix, mu is the mean, and col_num is the column number ##
		n = self.Node
		mu = self.M_mu
		pdf,cdf = self.distribute_list()

		arc_left = np.zeros([n,n]) 
		for col_index in range(n):
			d = np.count_nonzero(mu[:,col_index])
			output = np.zeros(d)
			vec_pdf = np.zeros(d)
			vec_cdf = np.zeros(d)
			left = np.zeros(d)
			j = 0
			for i in range(n):
				if mu[i,col_index] != 0:
					vec_pdf[j] = pdf[i,col_index]
					vec_cdf[j] = cdf[i,col_index]
					j = j + 1
			for i in range(d):
				left[i] = vec_pdf[i]*np.prod(np.concatenate((vec_cdf[:i],vec_cdf[i+1:])))
			left_sum = sum(left)
			k = 0
			for i in range(n):
				if mu[i,col_index] != 0:
					arc_left[i,col_index] = vec_pdf[k]*np.prod(np.concatenate((vec_cdf[:k],vec_cdf[k+1:])))/left_sum
					k = k + 1
		return arc_left   


	## Calculating the arc criticalities ##
	def ca_struct(self):
		dic = self.net_struct
		arc_left = self.critical_left()
		mu = self.M_mu

		n = len(mu[0,:])
		c = np.zeros([n,n])
		cn = np.zeros(n)
		cn[n-1] = 1
		length = len(dic[:,0])
		
		for i in range(length):
			if dic[i,0] == 1:
				m1 = dic[i,1]
				m2 = dic[i,2]
				c[m1,m2] = arc_left[m1,m2]*cn[m2]
			else:
				cn[dic[i,1]] = sum(c[dic[i,1],:])
		return c, cn  

	### Preparation for Calculating the IPA of CAC ###

	def check_diff_gen(self):
		n = self.Node
		diff = self.release_diff_gen()

		a = np.count_nonzero(diff)
		vec = np.zeros(a)
		k = 0
		for i in range(n):
			for j in range(n):
				if diff[i,j] > 0:
					vec[k] = diff[i,j]
					k = k + 1
		return min(vec)

	def release_indicator_gen(self,i,j):
		n = self.Node 
	  
		times = 0.01
		indicator = np.zeros([n,n])
		test = copy.deepcopy(self.simu)
		diff0 = self.release_diff_gen()

		sdiff = self.check_diff_gen()
		if sdiff > 0:
			perturb = times*sdiff
		else:
			perturb = 0.0001
		test[i,j] = test[i,j] + perturb

		t1 = self.ptime2(test)
		diff1 = self.release_diff_gen(t1)
		check = diff1 - diff0
		for k in range(self.Arc):
			i_h = self.nonzero_list[0][k]
			j_h = self.nonzero_list[1][k]
			if abs(check[i_h,j_h] - perturb) < times*perturb:
				indicator[i_h,j_h] = 1
			elif abs(check[i_h,j_h] + perturb)  < times*perturb:
				indicator[i_h,j_h] = -1
		return indicator

	def dflist_gen(self,m1,m2):
		a = self.Arc
		n = self.Node
		indicator = self.release_indicator_gen(m1,m2)
		distri = DstrSimu(self.Arc,self.Node,self.release_diff_gen(),self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)

		dpdf = np.zeros([n,n])
		dcdf = np.zeros([n,n])
		if self.Mdst[m1,m2] == 2:
			ipa = self.simu[m1,m2]/self.M_mu[m1,m2]
		else:
			ipa = 1
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if i == m1 and j == m2:
				dpdf[i,j] = distri.pdf_dx(i,j)*indicator[i,j]*ipa + distri.pdf_dpara(i,j)
				dcdf[i,j] = distri.pdf(i,j)*indicator[i,j]*ipa + distri.cdf_dpara(i,j)
			else:
				dpdf[i,j] = distri.pdf_dx(i,j)*indicator[i,j]*ipa 
				dcdf[i,j] = distri.pdf(i,j)*indicator[i,j]*ipa 
		return dpdf, dcdf

	def ipa_left_gen(self,m1,m2):
		n = self.Node
		ipa_left = np.zeros([n,n])
		pdf,cdf = self.distribute_list()
		dpdf,dcdf = self.dflist_gen(m1,m2)
		for col_index in range(m2,n):
			d = np.count_nonzero(self.G[:,col_index])
			output = np.zeros(d)
			vec_pdf = np.zeros(d)
			vec_cdf = np.zeros(d)
			vec_dpdf = np.zeros(d)
			vec_dcdf = np.zeros(d)
			j = 0
			for i in range(n):
				if mu[i,col_index] != 0:
					vec_pdf[j] = pdf[i,col_index]
					vec_dpdf[j] = dpdf[i,col_index]
					vec_cdf[j] = cdf[i,col_index]
					vec_dcdf[j] = dcdf[i,col_index]
					j = j + 1
					
			if d > 1:
				left_t = np.zeros(d)
				s = np.zeros(d)
				multi = np.zeros([d,d])
				for k in range(d):
					s[k] = vec_pdf[k]*np.prod(np.concatenate((vec_cdf[:k],vec_cdf[k+1:])))
					for j in range(d):
						multi[k,j] = vec_cdf[j]
						multi[k,k] = vec_pdf[k]
							 
				sum_s = sum(s)
				dv = np.zeros(d)
				for k in range(d):
					multi_dev = np.zeros([d,d])
					sum_dev = np.zeros(d)
					for j in range(d):
						multi_dev[j,:] = multi[k,:]
						if j != k:
							multi_dev[j,j] = vec_dcdf[j]
						else:
							multi_dev[j,j] = vec_dpdf[j]
						sum_dev[j] = np.prod(multi_dev[j,:])
					dv[k] = sum(sum_dev)
					
				sum_all_dev = sum(dv)
				for i in range(d):
					######### make changes here 2021/07/26 22:30 ################
					#left_t[i] = dv[i]/sum_s - (s[i]*sum_all_dev)/(sum_s**2)
					left_t[i] = (dv[i] - (s[i]*sum_all_dev)/(sum_s))*(1/sum_s)
					
					######### make changes here 2021/07/26 22:30 ################
				k = 0
				for i in range(n):
					if mu[i,col_index] != 0:
						ipa_left[i,col_index] = left_t[k]
						k = k + 1
		return ipa_left
					   
	def dca_struct(self,j,k):
		n = self.Node
		dic = self.net_struct
		arc_left = self.critical_left()
		ipa_left = self.ipa_left_gen(j,k)
		c,cn = self.ca_struct()

		dc = np.zeros([n,n])
		dcn = np.zeros(n)
		dcn[n-1] = 0.0
		length = len(dic[:,0])
		
		for i in range(length):
			if dic[i,0] == 1:
				m1 = dic[i,1]
				m2 = dic[i,2]
				dc[m1,m2] = ipa_left[m1,m2]*cn[m2] + arc_left[m1,m2]*dcn[m2]
			else:
				dcn[dic[i,1]] = sum(dc[dic[i,1],:])
		return dc    

	def dca_cac(self):
		dc_est = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			dc_est[i,j] = self.struct(i,j)[i,j]
		return dc_est

	############## Measure Based Methods ###############

	def ca_iac(self,simulate,i,j):
		t = self.ptime2(simulate)
		diff = self.release_diff_gen2(t)
		indic = diff[i,j]
		return indic

	def dca_lr(self):
		dc_est = np.zeros([self.Node,self.Node])
		distri = DstrSimu(self.Arc,self.Node,self.simu,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			dc_est[i,j] = self.ca_iac(self.simu[i,j],i,j)*distri.lr_part(i,j)
		return dc_est


	######## Weak Derivative Methods ##########

	def wd_dstr_left(self,i,j):
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			u = uniform.rvs(loc = 0, scale = 1)
			if self.Mdst[i,j] == 7:
				eps2 = self.M_tnormal_b[i,j] - self.M_mu[i,j]
				A = 1 - math.exp(-(eps2/(math.sqrt(2)*self.M_sigma[i,j]))**2)
				rt = math.sqrt(2*self.M_sigma[i,j])*math.sqrt(-np.log(1 - A*u))
			if self.Mdst[i,j] == 3:
				eps2 = self.M_tri_b[i,j] - self.M_tri_c[i,j]
				rt = eps2*u + self.M_tri_c[i,j]
			if self.Mdst[i,j] == 8:
				rt = np.random.triangular(self.M_theta[i,j],2*self.M_theta[i,j],2*self.M_theta[i,j])
			if self.Mdst[i,j] == 9:
				pt = PERT(0,(3/2)*self.M_theta[i,j],2*self.M_theta[i,j])
				rt = pt.rvs(1)
		return rt

	def wd_dstr_right(self,i,j):
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			u = uniform.rvs(loc = 0, scale = 1)
			if self.Mdst[i,j] == 7:
				eps1 = self.M_mu[i,j] - self.M_tnormal_a[i,j] 
				A = 1 - math.exp(-(eps1/(math.sqrt(2)*self.M_sigma[i,j]))**2)
				rt = math.sqrt(2*self.M_sigma[i,j])*math.sqrt(-np.log(1 - A*u))
			if self.Mdst[i,j] == 3:
				eps1 = self.M_tri_c[i,j] - self.M_tri_a[i,j]
				eps2 = self.M_tri_b[i,j] - self.M_tri_c[i,j]
				if u >=0 and u <= (2/(eps1 + eps2)):
					rt = ((eps1 + eps2)*eps1*u + 2*self.M_tri_a[i,j])/2
				else:
					sm12 = 2/(eps1 + eps2)
					rt = self.M_tri_c[i,j] + (eps2*(u - sm12))/(1 - sm12)
			if self.Mdst[i,j] == 8:
				rt = np.random.triangular(0,self.M_theta[i,j],self.M_theta[i,j])
			if self.Mdst[i,j] == 9:
				pt = PERT(0,self.M_theta[i,j],2*self.M_theta[i,j])
				rt = pt.rvs(1)
		return rt 

	def dca_wd_cmrv(self):
		dc_est = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 1:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				weib = np.random.weibull(2)
				simu0[i,j] = self.M_mu[i,j] + math.sqrt(2*self.M_sigma[i,j]**2)*weib
				simu1[i,j] = self.M_mu[i,j] - math.sqrt(2*self.M_sigma[i,j]**2)*weib
				dc_est[i,j] = (1/(math.sqrt(2*math.pi)*self.M_sigma[i,j]))*(self.ca_iac(simu0,i,j) - self.ca_iac(simu1,i,j))
			if self.Mdst[i,j] == 2:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = np.random.gamma(2,self.mu[i,j])
				dc_est[i,j] = (1/self.M_mu[i,j])*(self.ca_iac(simu0,i,j) - self.ca_iac(simu1,i,j))
		return dc_est

	def dEY_wd_cmrv(self):
		dc_est = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 1:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				weib = np.random.weibull(2)
				simu0[i,j] = self.M_mu[i,j] + math.sqrt(2*self.M_sigma[i,j]**2)*weib
				simu1[i,j] = self.M_mu[i,j] - math.sqrt(2*self.M_sigma[i,j]**2)*weib
				dc_est[i,j] = (1/(math.sqrt(2*math.pi)*self.M_sigma[i,j]))*(self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 2:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = np.random.gamma(2,self.mu[i,j])
				dc_est[i,j] = (1/self.M_mu[i,j])*(self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 6:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = gamma.rvs(self.M_gamma_shap[i,j]+1,loc=0,scale = self.M_gamma_scal[i,j])
				simu1[i,j] = gamma.rvs(self.M_gamma_shap[i,j],loc=0,scale = self.M_gamma_scal[i,j])
				dc_est[i,j] = (self.M_gamma_shap[i,j]/self.M_gamma_scal[i,j])*(self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 7:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = self.M_mu[i,j] + self.wd_dstr_left(i,j)
				simu1[i,j] = self.M_mu[i,j] - self.wd_dstr_right(i,j)
				dc_est[i,j] = (norm.cdf((self.M_tnormal_b[i,j] - self.M_mu[i,j])/self.M_sigma[i,j]) - norm.cdf((self.M_tnormal_a[i,j] - self.M_mu[i,j])/self.M_sigma[i,j]))*(1/(math.sqrt(2*math.pi)*self.M_sigma[i,j]))*(self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 3:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = self.wd_dstr_left(i,j)
				simu1[i,j] = self.wd_dstr_right(i,j)
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 8:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = self.wd_dstr_left(i,j)
				simu1[i,j] = self.wd_dstr_right(i,j)
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(self.M_theta[i,j])
			if self.Mdst[i,j] == 9:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = self.wd_dstr_left(i,j)
				simu1[i,j] = self.wd_dstr_right(i,j)
				dc_est[i,j] = 5*(self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(self.M_theta[i,j])
		return dc_est
	
	def dEY_finite_diff_cmrv(self):
		dc_est = np.zeros([self.Node,self.Node])
		eps = self.eps
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 1:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = np.random.normal((1 + eps/2)*self.M_mu[i,j],self.M_sigma[i,j],1)
				simu1[i,j] = np.random.normal((1 - eps/2)*self.M_mu[i,j],self.M_sigma[i,j],1)
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(eps*self.M_mu[i,j])
			if self.Mdst[i,j] == 2:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = np.random.exponential((1 + eps/2)*self.M_mu[i,j])
				simu1[i,j] = np.random.exponential((1 - eps/2)*self.M_mu[i,j])
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(eps*self.M_mu[i,j])
			if self.Mdst[i,j] == 6:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = gamma.rvs(self.M_gamma_shap[i,j],loc=0,scale = (1 + eps/2)*self.M_gamma_scal[i,j])
				simu1[i,j] = gamma.rvs(self.M_gamma_shap[i,j],loc=0,scale = (1 - eps/2)*self.M_gamma_scal[i,j])
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(eps*self.M_gamma_scal[i,j])
			if self.Mdst[i,j] == 8:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu0[i,j] = np.random.triangular(0,(1 + eps/2)*self.M_theta[i,j],2*(1 + eps/2)*self.M_theta[i,j])
				simu1[i,j] = np.random.triangular(0,(1 - eps/2)*self.M_theta[i,j],2*(1 - eps/2)*self.M_theta[i,j])
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(eps*self.M_theta[i,j])
			if self.Mdst[i,j] == 9:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				pt0 = PERT(0,(1 + eps/2)*self.M_theta[i,j],2*(1 + eps/2)*self.M_theta[i,j])
				simu0[i,j] = pt0.rvs(1)
				pt1 = PERT(0,(1 - eps/2)*self.M_theta[i,j],2*(1 - eps/2)*self.M_theta[i,j])
				simu1[i,j] = pt1.rvs(1)
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])/(eps*self.M_theta[i,j])
		return dc_est

	def EY_finite_diff_cmrv(self):
		dc_est = np.zeros([self.Node,self.Node])
		eps = self.eps
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 1:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu1[i,j] = np.random.normal((1 - eps)*self.M_mu[i,j],self.M_sigma[i,j],1)
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 2:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu1[i,j] = np.random.exponential((1 - eps)*self.M_mu[i,j])
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 6:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu1[i,j] = gamma.rvs(self.M_gamma_shap[i,j],loc=0,scale = (1 - eps)*self.M_gamma_scal[i,j])
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 8:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				simu1[i,j] = np.random.triangular(0,(1 - eps)*self.M_theta[i,j],2*(1 - eps)*self.M_theta[i,j])
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
			if self.Mdst[i,j] == 9:
				simu0 = copy.deepcopy(self.simu)
				simu1 = copy.deepcopy(self.simu)
				pt1 = PERT(0,(1 - eps)*self.M_theta[i,j],2*(1 - eps)*self.M_theta[i,j])
				simu1[i,j] = pt1.rvs(1)
				dc_est[i,j] = (self.ptime2(simu0)[-1] - self.ptime2(simu1)[-1])
		return dc_est

	def dEY_finite_diff_nocmrv(self):
		dc_est = np.zeros([self.Node,self.Node])
		eps = self.eps
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				theta0 = copy.deepcopy(self.M_theta)
				theta1 = copy.deepcopy(self.M_theta)
				theta0[i,j] = (1 + eps/2)*self.M_theta[i,j]
				theta1[i,j] = (1 - eps/2)*self.M_theta[i,j]
				simu0 = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,theta0,self.Mdst,self.nonzero_list)
				simu1 = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,theta1,self.Mdst,self.nonzero_list)
				dc_est[i,j] = (self.ptime2(simu0.SimuArcs())[-1] - self.ptime2(simu1.SimuArcs())[-1])/(eps*self.M_theta[i,j])
		return dc_est

	def lr_est(self,i,j):
		if self.Mdst[i,j] == 1 or self.Mdst[i,j] == 7:
			lr_part = (self.simu[i,j] - self.M_mu[i,j])/(self.M_sigma[i,j]**2)
		if self.Mdst[i,j] == 2:
			lr_part = (self.simu[i,j]/self.M_mu[i,j] - 1)/self.M_mu[i,j]
		if self.Mdst[i,j] == 6:
			lr_part = (self.simu[i,j]/self.M_gamma_scal[i,j] - self.M_gamma_shap[i,j])/self.M_gamma_scal[i,j]
		#### here triangle distribution takes the parameter c as a location parameter, i.e., both a, b, c changes the same amount ####
		if self.Mdst[i,j] == 3:
			if self.simu[i,j] >= self.M_tri_a[i,j] and self.simu[i,j] <= self.M_tri_c[i,j]:
				lr_part = (-1)/(self.simu[i,j] - self.M_tri_a[i,j])
			elif self.simu[i,j] > self.M_tri_c[i,j] and self.simu[i,j] <= self.M_tri_b[i,j]:
				lr_part = 1/(self.M_tri_b[i,j] - self.simu[i,j])
		if self.Mdst[i,j] == 8:
			if self.simu[i,j] > 2*self.M_theta[i,j] or self.simu[i,j] < 0:
				lr_part = 0
			elif self.simu[i,j] >= 0 and self.simu[i,j] <= self.M_theta[i,j]:
				lr_part = -2*self.M_theta[i,j]**(-1)
			elif self.simu[i,j] > self.M_theta[i,j] and self.simu[i,j] <= 2*self.M_theta[i,j]:
				lr_part = 2*(self.simu[i,j] - self.M_theta[i,j])/((2*self.M_theta[i,j] - self.simu[i,j])*self.M_theta[i,j])
		if self.Mdst[i,j] == 9:
			if self.simu[i,j] >= 0 and self.simu[i,j] <= 2*self.M_theta[i,j]:
				lr_part = 4*(self.M_theta[i,j] - self.simu[i,j])/(self.simu[i,j]*(2*self.M_theta[i,j] - self.simu[i,j]))
			else:
				lr_part = 0
		return lr_part 

	def dEY_lr(self):
		dc_est = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			dc_est[i,j] = self.ptime()[-1]*self.lr_est(i,j)
		return dc_est

	def dEY_tac(self):
		thd = self.cal_thd()
		est = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			mi = thd[i,j]
			theta = self.M_theta[i,j]
			if self.Mdst[i,j] == 8:
				if mi > 2*theta or mi < 0:
					est[i,j] = 0
				elif mi >= 0 and mi <= theta:
					est[i,j] = ((1 - (mi/theta)**3)+2)/3  
				elif mi > theta and mi <= 2*theta:
					est[i,j] = (4 - 3*(mi/theta)**2 + (mi/theta)**3)/3
			if self.Mdst[i,j] == 9:
				if mi > 2*theta or mi < 0:
					est[i,j] = 0
				else:
					c6 = theta**6
					cpert = 1/(32*self.B33*c6)
					est[i,j] = cpert*((16*c6/15)-mi**4*(mi**2/6 - 4*theta*mi/5 + theta**2))
		return est

	def EY_taylor_coeff(self,order):
		est0 = np.zeros([self.Node,self.Node])
		est1 = np.zeros([self.Node,self.Node])
		est2 = np.zeros([self.Node,self.Node])
		est3 = np.zeros([self.Node,self.Node])
		thd = self.cal_thd()
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				theta = self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				
			mi = thd[i,j]
			if self.Mdst[i,j] == 8:
				if mi > 2*theta or mi < 0:
					est0[i,j] = 0
					est1[i,j] = 0
					est2[i,j] = 0
					est3[i,j] = 0
				elif mi >= 0 and mi <= theta:
					if order == 0:
						est0[i,j] = ((1 - (mi/theta)**3)+2)/3  
					elif order == 1:
						cm_pt = (mi/theta)**3
						est0[i,j] = ((1 - cm_pt)+2)/3
						est1[i,j] = cm_pt/theta
					elif order == 2:
						cm_pt = (mi/theta)**3
						est0[i,j] = ((1 - cm_pt)+2)/3
						est1[i,j] = cm_pt/theta
						est2[i,j] = -4*cm_pt*theta**(-2)
				elif mi > theta and mi <= 2*theta:
					cm_pt = (mi/theta)**2
					if order == 0:				
						est0[i,j] = (4 - 3*cm_pt + (mi/theta)*cm_pt)/3
					if order == 1:
						est0[i,j] = (4 - 3*cm_pt + (mi/theta)*cm_pt)/3
						est1[i,j] = (cm_pt/theta)*(2 - mi*theta**(-1))
					if order == 2:
						est0[i,j] = (4 - 3*cm_pt + (mi/theta)*cm_pt)/3
						est1[i,j] = (cm_pt/theta)*(2 - mi*theta**(-1))
						est2[i,j] = (cm_pt*theta**(-2))*(-6 + 4*mi*theta**(-1))       
			if self.Mdst[i,j] == 9:
				if mi > 2*theta or mi < 0:
					est0[i,j] = 0
					est1[i,j] = 0
					est2[i,j] = 0
					est3[i,j] = 0
				else:
					cpt = 1/(32*self.B33)
					cm_pt = mi**4*theta**(-4)
					cpert = mi**4*cpt
					y2 = mi**2*theta**(-2)
					y1 = theta**(-1)*mi
					if order == 0:
						est0[i,j] = cpt*((16/15)-cm_pt*(y2/6 - 4*y1/5 + 1))
					elif order == 1:
						est0[i,j] = cpt*((16/15)-cm_pt*(y2/6 - 4*y1/5 + 1))
						est1[i,j] = cpert*theta**(-5)*(y2 - 4*y1 + 4)
					elif order == 2:
						est0[i,j] = cpt*((16/15)-cm_pt*(y2/6 - 4*y1/5 + 1))
						est1[i,j] = cpert*theta**(-5)*(y2 - 4*y1 + 4)
						est2[i,j] = cpert*theta**(-6)*(-7*y2 + 24*y1 - 20)
					elif order == 3:
						est0[i,j] = cpt*((16/15)-cm_pt*(y2/6 - 4*y1/5 + 1))
						est1[i,j] = cpert*theta**(-5)*(y2 - 4*y1 + 4)
						est2[i,j] = cpert*theta**(-6)*(-7*y2 + 24*y1 - 20)
						est3[i,j] = cpert*theta**(-7)*8*(7*y2 - 21*y1 + 15)
			if self.Mdst[i,j] == 2:
				cm_pt = math.exp(-mi*(1/theta))
			#if self.Mdst[i,j] == 6:
				#if order == 0:
		return est0,est1,est2,est3

	def EX_taylor_coeff(self,order):
		est0 = np.zeros([self.Node,self.Node])
		est1 = np.zeros([self.Node,self.Node])
		est2 = np.zeros([self.Node,self.Node])
		est3 = np.zeros([self.Node,self.Node])
		thd,A = self.cal_thd_Amatrix()
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			theta = self.M_theta[i,j]
			mi = thd[i,j]
			if self.Mdst[i,j] == 8:
				if mi > 2*theta or mi < 0:
					est0[i,j] = 0
					est1[i,j] = 0
					est2[i,j] = 0
					est3[i,j] = 0
				elif mi >= 0 and mi <= theta:
					cm_pt = (mi**4)/(theta**2)
					if order == 0:
						est0[i,j] = 7*theta**2/6 - cm_pt/4
					elif order == 1:
						est0[i,j] = 7*theta**2/6 - cm_pt/4
						est1[i,j] = 7*theta/3 + cm_pt/(2*theta)
					elif order == 2:
						est0[i,j] = 7*theta**2/6 - cm_pt/4
						est1[i,j] = 7*theta/3 + cm_pt/(2*theta)
						est2[i,j] = 7/3 - 3*cm_pt/(2*theta**2)
				elif mi > theta and mi <= 2*theta:
					cm_pt = (mi**3)/theta
					if order == 0:				
						est0[i,j] = 4*theta**2/3 - cm_pt*(2/3 - mi/(4*theta))
					if order == 1:
						est0[i,j] = 4*theta**2/3 - cm_pt*(2/3 - mi/(4*theta))
						est1[i,j] = 8*theta/3 + cm_pt*(2/(3*theta) - mi/(2*theta**2))
					if order == 2:
						est0[i,j] = 4*theta**2/3 - cm_pt*(2/3 - mi/(4*theta))
						est1[i,j] = 8*theta/3 + cm_pt*(2/(3*theta) - cm_pt*mi/(2*theta**2))
						est2[i,j] = 8/3 - 4*cm_pt/(3*theta**2) + 3*cm_pt*mi/(2*theta**3)  
			if self.Mdst[i,j] == 9:
				if mi > 2*theta or mi < 0:
					est0[i,j] = 0
					est1[i,j] = 0
					est2[i,j] = 0
					est3[i,j] = 0
				else:
					cpt = 1/(32*self.B33)
					cm_pt = mi**5*theta**(-4)
					y2 = mi**2*theta**(-2)
					y1 = theta**(-1)*mi
					if order == 0:
						est0[i,j] = cpt*((theta/105) - cm_pt*(y2/7 - 2*y1/3 + 4/5))
					elif order == 1:
						est0[i,j] = cpt*((theta/105) - cm_pt*(y2/7 - 2*y1/3 + 4/5))
						est1[i,j] = cpt*(1/105 + (2*cm_pt/theta)*(3*y2/7 - 5*y1/3 + 8/5))
					elif order == 2:
						est0[i,j] = cpt*((theta/105) - cm_pt*(y2/7 - 2*y1/3 + 4/5))
						est1[i,j] = cpt*(1/105 + (2*cm_pt/theta)*(3*y2/7 - 5*y1/3 + 8/5))
						est2[i,j] = cpt*2*cm_pt*theta**(-2)*(-3*y2 + 10*y1 - 8)
					elif order == 3:
						est0[i,j] = cpt*((theta/105) - cm_pt*(y2/7 - 2*y1/3 + 4/5))
						est1[i,j] = cpt*(1/105 + (2*cm_pt/theta)*(3*y2/7 - 5*y1/3 + 8/5))
						est2[i,j] = cpt*2*cm_pt*theta**(-2)*(-3*y2 + 10*y1 - 8)
						est3[i,j] = cpt*4*cm_pt*theta**(-3)*(12*y2 - 35*y1 + 24)		
		return est0,est1,est2,est3,A
	

	def EY_taylor_est(self,order,percent):
		est0, est1, est2, est3 = self.EY_taylor_coeff(order)
		estimation = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				delta = percent*self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				delta = percent*self.M_mu[i,j]
			elif self.Mdst[i,j] == 6:
				delta = percent*self.M_gamma_scal[i,j]
			if order == 0:
				estimation[i,j] = est0[i,j]*delta
			elif order == 1:
				estimation[i,j] = est0[i,j]*delta - 0.5*est1[i,j]*delta**2
			elif order == 2:
				estimation[i,j] = est0[i,j]*delta - 0.5*est1[i,j]*delta**2 + est2[i,j]*delta**3/6
			elif order == 3:
				estimation[i,j] = est0[i,j]*delta - 0.5*est1[i,j]*delta**2 + est2[i,j]*delta**3/6 - est3[i,j]*delta**4/24
		return estimation

	def EY2_taylor_coeff(self,order):
		ey0, ey1, ey2, ey3 = self.EY_taylor_coeff(order)
		ex0, ex1, ex2, ex3, A = self.EX_taylor_coeff(order)
		est0 = np.zeros([self.Node,self.Node])
		est1 = np.zeros([self.Node,self.Node])
		est2 = np.zeros([self.Node,self.Node])
		est3 = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]

			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				theta = self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				theta = self.M_mu[i,j]
			elif self.Mdst[i,j] == 6:
				theta = self.M_gamma_scal[i,j]

			if order == 0:
				est0[i,j] = 2*(ex0[i,j] + A[i,j]*ey0[i,j])/theta
			elif order == 1:
				est0[i,j] = 2*(ex0[i,j] + A[i,j]*ey0[i,j])/theta
				est1[i,j] = 2*(ex1[i,j] + A[i,j]*ey1[i,j])/theta
			elif order == 2:
				est0[i,j] = 2*(ex0[i,j] + A[i,j]*ey0[i,j])/theta
				est1[i,j] = 2*(ex1[i,j] + A[i,j]*ey1[i,j])/theta
				est2[i,j] = 2*(ex2[i,j] + A[i,j]*ey2[i,j])/theta
			elif order == 3:
				est0[i,j] = 2*(ex0[i,j] + A[i,j]*ey0[i,j])/theta
				est1[i,j] = 2*(ex1[i,j] + A[i,j]*ey1[i,j])/theta
				est2[i,j] = 2*(ex2[i,j] + A[i,j]*ey2[i,j])/theta
				est3[i,j] = 2*(ex3[i,j] + A[i,j]*ey3[i,j])/theta
		return ey0, ey1, ey2, ey3, est0, est1, est2, est3

	def EY_taylor_ratio(self,order,percent):
		est0, est1, est2, est3 = self.EY_taylor_coeff(order)
		estimation = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				delta = percent*self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				delta = percent*self.M_mu[i,j]
			elif self.Mdst[i,j] == 6:
				delta = percent*self.M_gamma_scal[i,j]
			if order == 0:
				estimation[i,j] = est0[i,j]
			elif oder == 1:
				estimation[i,j] = est0[i,j] - 0.5*est1[i,j]*delta
			elif order == 2:
				estimation[i,j] = est0[i,j] - 0.5*est1[i,j]*delta + est2[i,j]*delta**2/6
			elif order == 3:
				estimation[i,j] = est0[i,j] - 0.5*est1[i,j]*delta + est2[i,j]*delta**2/6 - est3[i,j]*delta**3/24
		return estimation

	def dEY_cac(self):
		c, cn = self.ca_struct()
		est = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.simu[i,j] <= 2*self.M_theta[i,j] and self.simu[i,j] > 0:
				est[i,j] = c[i,j]*self.simu[i,j]/self.M_theta[i,j]
		return est













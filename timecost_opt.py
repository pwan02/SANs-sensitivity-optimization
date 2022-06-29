## Author: Peng Wan
## Date: 10/05/2021

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

from CoreEst import GradientEst

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

class EYconf:
	def __init__(self,N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent):
		self.N_repl = N_repl
		self.Arc = Arc
		self.Node = Node
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
		self.percent = percent

	def EY_cfit(self):
		Ne = 2000
		EY_est_vec = np.zeros(Ne)
		for k in range(Ne):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)	
			EY_est_vec[k] = EYGR.ptime()[-1]
		mean = np.mean(EY_est_vec)
		variance = np.var(EY_est_vec,ddof = 1)
		return mean,variance
	
	def EY_linear_coeff(self):
		Ne = 20000
		EY_est_vec = np.zeros(Ne)
		for k in range(Ne):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)	
			EY_est_vec[k] = EYGR.ptime()[-1]
		mean = np.mean(EY_est_vec)
		variance = np.var(EY_est_vec,ddof = 1)
		sm = mean + variance
		coeff1 = 1/(1+(variance/mean))
		coeff0 = 1 - coeff1
		return coeff0, coeff1

class Optimization:
	
	def __init__(self,N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,cost,budget,percent,order,lbd_mt,coeff_var0,coeff_var1):
		self.N_repl = N_repl
		self.Arc = Arc
		self.Node = Node
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
		self.cost = cost
		self.budget = budget
		self.percent = percent
		self.order = order
		self.lbd_mt = lbd_mt

		self.coeff_var0 = coeff_var0
		self.coeff_var1 = coeff_var1
		
	def argmax_cord(self,mt):
		mv = np.max(mt)
		n = len(mt[0])
		for i in range(n):
			for j in range(n):
				if abs(mt[i,j] - mv) < 0.0001:
					max_i = i
					max_j = j
		return max_i,max_j

	def diff_ratio(self,M):
		M_diff_ratio = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			M_diff_ratio[i,j] = M[i,j]/self.cost[i,j]
		return M_diff_ratio

	def Mean_Matrix(self,M):
		M_mean = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			M_mean[i,j] = np.mean(M[:,i,j])
		return M_mean

	def compare_matrix(self,M1,M2):
		count = 0
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			count += abs(M1[i,j] - M2[i,j])
		avg_count = count/self.Arc
		return avg_count

	def RSE_average(self,M):
		rse = 0
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			vec = np.zeros(self.N_repl)
			for n in range(self.N_repl):
				vec[n] = M[n,i,j]
			rse_k = self.RSE_measure(vec)
			rse += rse_k
		average = rse/self.Arc
		return average

	def simulate_dictionary(self):
		dist_dic = {}
		for k in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			dist_dic[k] = simu_sing.SimuArcs()
		return dist_dic

	def EY_diff_mean(self):
		N_large = 10000 
		M_measure_fd = np.zeros([N_large,self.Node,self.Node])
		for i in range(N_large):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)
			M_measure_fd[i,:,:] = EYGR.EY_finite_diff_cmrv()
			M_mean_fd = self.Mean_Matrix(M_measure_fd)
		return M_mean_fd

	## Calculate the difference of EY ##

	def EY_est(self,parameter):
		N_large = self.N_repl
		dEY_0 = np.zeros([N_large,self.Node,self.Node])
		dEY_1 = np.zeros([N_large,self.Node,self.Node])
		dEY_2 = np.zeros([N_large,self.Node,self.Node])
		dEY_3 = np.zeros([N_large,self.Node,self.Node])

		dEY2_0 = np.zeros([N_large,self.Node,self.Node])
		dEY2_1 = np.zeros([N_large,self.Node,self.Node])
		dEY2_2 = np.zeros([N_large,self.Node,self.Node])
		dEY2_3 = np.zeros([N_large,self.Node,self.Node])
		
		M_mu = self.M_mu
		M_sigma = self.M_sigma
		M_unif_loca = self.M_unif_loca
		M_unif_scal = self.M_unif_scal
		M_tri_a = self.M_tri_a
		M_tri_b = self.M_tri_b
		M_tri_c = self.M_tri_c
		M_gamma_shap = self.M_gamma_shap
		M_gamma_scal = self.M_gamma_scal
		M_tnormal_a = self.M_tnormal_a
		M_tnormal_b = self.M_tnormal_b
		M_theta = self.M_theta

		if self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 8 or self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 9:
			M_theta = copy.deepcopy(parameter)
		elif self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 1 or self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 2:
			M_mu = copy.deepcopy(parameter)

		for k in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,self.Mdst,self.percent)
			dEY_0[k,:,:],dEY_1[k,:,:],dEY_2[k,:,:],dEY_3[k,:,:] = EYGR.EY_taylor_coeff(self.order)

		dEY_est0 = self.Mean_Matrix(dEY_0)
		dEY_est1 = self.Mean_Matrix(dEY_1)
		dEY_est2 = self.Mean_Matrix(dEY_2)
		dEY_est3 = self.Mean_Matrix(dEY_3)

		return dEY_est0,dEY_est1,dEY_est2,dEY_est3

	def EY2_est(self,parameter):
		N_large = self.N_repl
		dEY_0 = np.zeros([N_large,self.Node,self.Node])
		dEY_1 = np.zeros([N_large,self.Node,self.Node])
		dEY_2 = np.zeros([N_large,self.Node,self.Node])
		dEY_3 = np.zeros([N_large,self.Node,self.Node])

		dEY2_0 = np.zeros([N_large,self.Node,self.Node])
		dEY2_1 = np.zeros([N_large,self.Node,self.Node])
		dEY2_2 = np.zeros([N_large,self.Node,self.Node])
		dEY2_3 = np.zeros([N_large,self.Node,self.Node])
		
		M_mu = self.M_mu
		M_sigma = self.M_sigma
		M_unif_loca = self.M_unif_loca
		M_unif_scal = self.M_unif_scal
		M_tri_a = self.M_tri_a
		M_tri_b = self.M_tri_b
		M_tri_c = self.M_tri_c
		M_gamma_shap = self.M_gamma_shap
		M_gamma_scal = self.M_gamma_scal
		M_tnormal_a = self.M_tnormal_a
		M_tnormal_b = self.M_tnormal_b
		M_theta = self.M_theta

		if self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 8 or self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 9:
			M_theta = copy.deepcopy(parameter)
		elif self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 1 or self.Mdst[self.nonzero_list[0][0],self.nonzero_list[1][0]] == 2:
			M_mu = copy.deepcopy(parameter)

		for k in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,self.Mdst,self.percent)
			dEY_0[k,:,:],dEY_1[k,:,:],dEY_2[k,:,:],dEY_3[k,:,:],dEY2_0[k,:,:],dEY2_1[k,:,:],dEY2_2[k,:,:],dEY2_3[k,:,:] = EYGR.EY2_taylor_coeff(self.order)

		dEY_est0 = self.Mean_Matrix(dEY_0)
		dEY_est1 = self.Mean_Matrix(dEY_1)
		dEY_est2 = self.Mean_Matrix(dEY_2)
		dEY_est3 = self.Mean_Matrix(dEY_3)
		dEY2_est0 = self.Mean_Matrix(dEY2_0)
		dEY2_est1 = self.Mean_Matrix(dEY2_1)
		dEY2_est2 = self.Mean_Matrix(dEY2_2)
		dEY2_est3 = self.Mean_Matrix(dEY2_3)

		return dEY_est0,dEY_est1,dEY_est2,dEY_est3,dEY2_est0,dEY2_est1,dEY2_est2,dEY2_est3

	def VarY_est(self,EY,parameter):
		dEY_est0,dEY_est1,dEY_est2,dEY_est3,dEY2_est0,dEY2_est1,dEY2_est2,dEY2_est3 = self.EY2_est(parameter)

		dVar_est0 = np.zeros([self.Node,self.Node])
		dVar_est1 = np.zeros([self.Node,self.Node])
		dVar_est2 = np.zeros([self.Node,self.Node])
		dVar_est3 = np.zeros([self.Node,self.Node])

		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			dVar_est0[i,j] = dEY2_est0[i,j] - 2*EY*dEY_est0[i,j]
			dVar_est1[i,j] = dEY2_est1[i,j] - 2*(dEY_est0[i,j])**2 - 2*EY*dEY_est1[i,j]
			dVar_est2[i,j] = dEY2_est2[i,j] - 6*dEY_est0[i,j]*dEY_est1[i,j] - 2*EY*dEY_est2[i,j]
			dVar_est3[i,j] = dEY2_est3[i,j] - 6*(dEY_est1[i,j])**2 - 8*dEY_est0[i,j]*dEY_est2[i,j] - 2*EY*dEY_est3[i,j]

		return dVar_est0,dVar_est1,dVar_est2,dVar_est3,dEY_est0,dEY_est1,dEY_est2,dEY_est3


	def generate_lowbd_matrix(self):
		M_lbd = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			M_lbd[i,j] = 1 - round(uniform.rvs(loc = 0.5, scale = 0.3),1)
		return M_lbd

	def EY2_diff(self,EY,parameter):
		EY_diff = np.zeros([self.Node,self.Node])
		VarY_diff = np.zeros([self.Node,self.Node])
		Vary_0,Vary_1,Vary_2,Vary_3,Ey_0,Ey_1,Ey_2,Ey_3 = self.VarY_est(EY,parameter)
		lbd_mt = self.lbd_mt
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				delta = self.percent*lbd_mt[i,j]*self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				delta = self.percent*lbd_mt[i,j]*self.M_mu[i,j] 
			elif self.Mdst[i,j] == 6:
				delta = self.percent*lbd_mt[i,j]*self.M_gamma_scal[i,j] 
			if order == 0:
				EY_diff[i,j] = Ey_0[i,j]*delta
				VarY_diff[i,j] = Vary_0[i,j]*delta
			elif order == 1:
				EY_diff[i,j] = Ey_0[i,j]*delta - 0.5*Ey_1[i,j]*delta**2
				VarY_diff[i,j] = Vary_0[i,j]*delta - 0.5*Vary_1[i,j]*delta**2
			elif order == 2:
				EY_diff[i,j] = Ey_0[i,j]*delta - 0.5*Ey_1[i,j]*delta**2 + Ey_2[i,j]*delta**3/6
				VarY_diff[i,j] = Vary_0[i,j]*delta - 0.5*Vary_1[i,j]*delta**2 + Vary_2[i,j]*delta**3/6
			elif order == 3:
				EY_diff[i,j] = Ey_0[i,j]*delta - 0.5*Ey_1[i,j]*delta**2 + Ey_2[i,j]*delta**3/6 - Ey_3[i,j]*delta**4/24
				VarY_diff[i,j] = Vary_0[i,j]*delta - 0.5*Vary_1[i,j]*delta**2 + Vary_2[i,j]*delta**3/6 - Vary_3[i,j]*delta**4/24
		return EY_diff, VarY_diff

	def EY_diff_ratio(self,parameter):
		EY_diff_estimate = np.zeros([self.Node,self.Node])
		EY_diff_ratio = np.zeros([self.Node,self.Node])
		VarY_diff_ratio = np.zeros([self.Node,self.Node])
		Ey_0,Ey_1,Ey_2,Ey_3 = self.EY_est(parameter)
		lbd_mt = self.lbd_mt
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				delta = self.percent*lbd_mt[i,j]*self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				delta = self.percent*lbd_mt[i,j]*self.M_mu[i,j] 
			elif self.Mdst[i,j] == 6:
				delta = self.percent*lbd_mt[i,j]*self.M_gamma_scal[i,j] 
			if order == 0:
				EY_diff_ratio[i,j] = Ey_0[i,j]/self.cost[i,j]
			elif order == 1:
				EY_diff_ratio[i,j] = (Ey_0[i,j] - 0.5*Ey_1[i,j]*delta)/self.cost[i,j]
			elif order == 2:
				EY_diff_ratio[i,j] = (Ey_0[i,j] - 0.5*Ey_1[i,j]*delta + Ey_2[i,j]*delta**2/6)/self.cost[i,j]
			elif order == 3:
				EY_diff_ratio[i,j] = (Ey_0[i,j] - 0.5*Ey_1[i,j]*delta + Ey_2[i,j]*delta**2/6 - Ey_3[i,j]*delta**3/24)/self.cost[i,j]
		return EY_diff_ratio

	def VarY_diff_ratio(self,EY,parameter):
		EY_diff_estimate = np.zeros([self.Node,self.Node])
		EY_diff_ratio = np.zeros([self.Node,self.Node])
		VarY_diff_ratio = np.zeros([self.Node,self.Node])
		Vary_0,Vary_1,Vary_2,Vary_3,Ey_0,Ey_1,Ey_2,Ey_3 = self.VarY_est(EY,parameter)
		lbd_mt = self.lbd_mt
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				delta = self.percent*lbd_mt[i,j]*self.M_theta[i,j]
			elif self.Mdst[i,j] == 2:
				delta = self.percent*lbd_mt[i,j]*self.M_mu[i,j] 
			elif self.Mdst[i,j] == 6:
				delta = self.percent*lbd_mt[i,j]*self.M_gamma_scal[i,j] 
			if order == 0:
				EY_diff_ratio[i,j] = Ey_0[i,j]/self.cost[i,j]
				EY_diff_estimate[i,j] = Ey_0[i,j]*delta
				VarY_diff_ratio[i,j] = Vary_0[i,j]/self.cost[i,j]
			elif order == 1:
				EY_diff_ratio[i,j] = (Ey_0[i,j] - 0.5*Ey_1[i,j]*delta)/self.cost[i,j]
				EY_diff_estimate[i,j] = Ey_0[i,j]*delta - 0.5*Ey_1[i,j]*delta**2
				VarY_diff_ratio[i,j] = (Vary_0[i,j] - 0.5*Vary_1[i,j]*delta)/self.cost[i,j]
			elif order == 2:
				EY_diff_ratio[i,j] = (Ey_0[i,j] - 0.5*Ey_1[i,j]*delta + Ey_2[i,j]*delta**2/6)/self.cost[i,j]
				EY_diff_estimate[i,j] = Ey_0[i,j]*delta - 0.5*Ey_1[i,j]*delta**2 + Ey_2[i,j]*delta**3/6
				VarY_diff_ratio[i,j] = (Vary_0[i,j] - 0.5*Vary_1[i,j]*delta + Vary_2[i,j]*delta**2/6)/self.cost[i,j]
			elif order == 3:
				EY_diff_ratio[i,j] = (Ey_0[i,j] - 0.5*Ey_1[i,j]*delta + Ey_2[i,j]*delta**2/6 - Ey_3[i,j]*delta**3/24)/self.cost[i,j]
				EY_diff_estimate[i,j] = Ey_0[i,j]*delta - 0.5*Ey_1[i,j]*delta**2 + Ey_2[i,j]*delta**3/6 - Ey_3[i,j]*delta**4/24
				VarY_diff_ratio[i,j] = (Vary_0[i,j] - 0.5*Vary_1[i,j]*delta + Vary_2[i,j]*delta**2/6 - Vary_3[i,j]*delta**3/24)/self.cost[i,j]
		## Combination of EY and VarY as the objective function ##
		Combine_diff_ratio = self.coeff_var0*EY_diff_ratio + self.coeff_var1*VarY_diff_ratio
		return EY_diff_ratio, Combine_diff_ratio, EY_diff_estimate


	def paralwbd_check(self,parameter,parameter0):
		check_matrix = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			# if abs(parameter[i,j] - (1-self.lbd_mt[i,j])*parameter0[i,j]) < 0.01:
			# 	check_matrix[i,j] = 0
			# else:
			# 	check_matrix[i,j] = 1
			if parameter[i,j] >= (1-self.lbd_mt[i,j])*parameter0[i,j] + min(1-self.lbd_mt[i,j],self.lbd_mt[i,j])*parameter0[i,j]*self.percent:
				check_matrix[i,j] = 1
		return check_matrix

	def argmax_cord(self,mt):
		mv = np.max(mt)
		n = len(mt[0])
		for i in range(n):
			for j in range(n):
				if abs(mt[i,j] - mv) < 0.0001:
					max_i = i
					max_j = j
		return max_i,max_j

	def optimization_VarY(self):
		parameter0 = np.zeros([self.Node,self.Node])
		parameter_E = np.zeros([self.Node,self.Node])
		parameter_V = np.zeros([self.Node,self.Node])
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if self.Mdst[i,j] == 2:
				parameter0[i,j] = copy.deepcopy(self.mu[i,j])
				parameter_E[i,j] = copy.deepcopy(self.mu[i,j])
				parameter_V[i,j] = copy.deepcopy(self.mu[i,j])
			elif self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
				parameter0[i,j] = copy.deepcopy(self.M_theta[i,j])
				parameter_E[i,j] = copy.deepcopy(self.M_theta[i,j])
				parameter_V[i,j] = copy.deepcopy(self.M_theta[i,j])
			elif self.Mdst[i,j] == 6:
				parameter0[i,j] = copy.deepcopy(self.M_gamma_scal[i,j])
				parameter_E[i,j] = copy.deepcopy(self.M_gamma_scal[i,j])
				parameter_V[i,j] = copy.deepcopy(self.M_gamma_scal[i,j])
		
		Ne = 5000
		EY_est_vec = np.zeros(Ne)
		for k in range(Ne):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)	
			EY_est_vec[k] = EYGR.ptime()[-1]

		EY = np.mean(EY_est_vec)
		current_budget_E = copy.deepcopy(self.budget)
		current_budget_V = copy.deepcopy(self.budget)

		while -1:
			EY_diff_ratio_V, Combine_diff_ratio_V, EY_diff_estimate_V = self.VarY_diff_ratio(EY,parameter_E)
			check_mt_E = np.multiply(EY_diff_ratio,self.paralwbd_check(parameter_E,parameter0))
			check_mt_V = np.multiply(Combine_diff_ratio,self.paralwbd_check(parameter_V,parameter0))
			indE_i,indE_j = self.argmax_cord(check_mt_E)
			indV_i,indV_j = self.argmax_cord(check_mt_V)
			ot_cost_E = self.cost[indE_i,indE_j]*self.percent*self.lbd_mt[indE_i,indE_j]*parameter0[indE_i,indE_j]
			ot_cost_V = self.cost[indV_i,indV_j]*self.percent*self.lbd_mt[indV_i,indV_j]*parameter0[indE_i,indE_j]

			if ot_cost_E <= current_budget_E:
				parameter_E[indE_i,indE_j] = parameter_E[indE_i,indE_j] - lbd_mt[indE_i,indE_j]*parameter0[indE_i,indE_j]*percent
				current_budget_E = current_budget_E - ot_cost_E

			elif ot_cost_E > current_budget_E and current_budget_E != 0:
				parameter_E[indE_i,indE_j] = parameter_E[indE_i,indE_j] - (current_budget_E/self.cost[indE_i,indE_j])
				current_budget_E = 0

			if ot_cost_V <= current_budget_V:
				parameter_V[indV_i,indV_j] = parameter_V[indV_i,indV_j] - lbd_mt[indV_i,indV_j]*parameter0[indV_i,indV_j]*percent
				current_budget_V = current_budget_V - ot_cost_V
				EY = EY - EY_diff_estimate[indV_i,indV_j]

			elif ot_cost_V > current_budget_V and current_budget_V != 0:
				parameter_V[indV_i,indV_j] = parameter_V[indV_i,indV_j] - (current_budget_V/self.cost[indV_i,indV_j])
				current_budget_V = 0
			
			if current_budget_E + current_budget_V == 0:
				break
		return parameter_E, parameter_V

	def optimization_EY(self):
		parameter0 = np.zeros([self.Node,self.Node])
		parameter_E = np.zeros([self.Node,self.Node])
		i = self.nonzero_list[0][0]
		j = self.nonzero_list[1][0]
		if self.Mdst[i,j] == 2:
			parameter0 = copy.deepcopy(self.mu)
			parameter_E = copy.deepcopy(self.mu)
				
		elif self.Mdst[i,j] == 8 or self.Mdst[i,j] == 9:
			parameter0 = copy.deepcopy(self.M_theta)
			parameter_E = copy.deepcopy(self.M_theta)
				
		elif self.Mdst[i,j] == 6:
			parameter0 = copy.deepcopy(self.M_gamma_scal)
			parameter_E = copy.deepcopy(self.M_gamma_scal)				
		
		current_budget_E = copy.deepcopy(self.budget)

		while -1:
			EY_diff_r = self.EY_diff_ratio(parameter_E)
			check_mt_E = np.multiply(EY_diff_r,self.paralwbd_check(parameter_E,parameter0))
			indE_i,indE_j = self.argmax_cord(check_mt_E)
			ot_cost_E = self.cost[indE_i,indE_j]*self.percent*self.lbd_mt[indE_i,indE_j]*parameter0[indE_i,indE_j]

			if ot_cost_E <= current_budget_E:
				parameter_E[indE_i,indE_j] = parameter_E[indE_i,indE_j] - lbd_mt[indE_i,indE_j]*parameter0[indE_i,indE_j]*percent
				current_budget_E = current_budget_E - ot_cost_E

			elif ot_cost_E > current_budget_E:
				parameter_E[indE_i,indE_j] = parameter_E[indE_i,indE_j] - (current_budget_E/self.cost[indE_i,indE_j])
				current_budget_E = 0
				break

		return parameter_E



import argparse

parser = argparse.ArgumentParser(description='Node and Arc number')
parser.add_argument('-N','--Node',type = int,required = True, help = 'Node Numbers')
parser.add_argument('-A','--Arc',type = int,required = True, help = 'Arc Numbers')
args = parser.parse_args()

if __name__ == "__main__":
	Node = args.Node
	Arc = args.Arc
	N_repl = 200

	gnwk = RmGraphGen(Node,Arc)
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
	theta_b = 5
	theta_c = 15
	cost_b = 1
	cost_c = 9
	percent = 0.25
	order = 2
	low_bd = 0.05

	G = gnwk.generate_graph_am()
	nonzero_list = np.nonzero(G)
	graph_information = GraphInform(Node,Arc,G)
	net_struct = graph_information.network_struct()
	dictionary = graph_information.countpath()
	inform = graph_information.informt()

	par = ParaGen(Arc,Node,mu_b,mu_c,sigma_b,sigma_c,unif_loca_b,unif_loca_c,unif_scal_b,unif_scal_c,tri_left_b,tri_left_c,tri_right_b,tri_right_c,gamma_shap_b,gamma_shap_c,gamma_scal_b,gamma_scal_c,tnormal_left_b,tnormal_left_c,tnormal_right_b,tnormal_right_c,theta_b,theta_c,cost_b,cost_c,nonzero_list)
	M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta = par.paragenerate()
	lbd_mt = par.generate_lowbd_matrix()
	cost = par.generate_cost()
	budget = np.sum(np.multiply(np.multiply(cost,M_theta),lbd_mt))/3

	# sublist = [8]
	# Mdst = par.dstr_gen(sublist)

	# sublist = [9]
	# Mdst = par.dstr_gen(sublist)

	# EYCFINT_tri = EYconf(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent)
	# EY,variance = EYCFINT_tri.EY_cfit()
	# coeff_var0_tri, coeff_var1_tri = EYCFINT_tri.EY_linear_coeff()
	# OPTIM_tri = Optimization(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,cost,budget,percent,order,lbd_mt,coeff_var0_tri,coeff_var1_tri)
	# EY_diff_ratio, Combine_diff_ratio, EY_diff_estimate = OPTIM_tri.VarY_diff_ratio(EY)
	# dEY_est0,dEY_est1,dEY_est2,dEY_est3,dEY2_est0,dEY2_est1,dEY2_est2,dEY2_est3 = OPTIM_tri.VarY_est(EY)

	# print(EY_diff_ratio)
	# print(Combine_diff_ratio)
	# print(EY_diff_estimate)
	# print(cost)
	# print(coeff_var0_tri)
	# print(coeff_var1_tri)
	# print(EY)
	# print(variance)
	# print(dEY_est0)
	# print(dEY_est1)
	# print(dEY_est2)
	# print(dEY_est3)
	# print(dEY2_est0)
	# print(dEY2_est1)
	# print(dEY2_est2)
	# print(dEY2_est3)

	# coeff_var0_tri, coeff_var1_tri = EYCFINT_tri.EY_linear_coeff()
	# OPTIM_tri = Optimization(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,cost,budget,percent,order,lbd_mt,coeff_var0_tri,coeff_var1_tri)
	# parE_tri, parV_tri = OPTIM_tri.optimization_VarY()
	# EYCFINT_E_tri = EYconf(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,parE_tri,Mdst,percent)
	# EYCFINT_V_tri = EYconf(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,parV_tri,Mdst,percent)
	# sm_E_tri,svar_E_tri = EYCFINT_E_tri.EY_cfit()
	# sm_V_tri,svar_V_tri = EYCFINT_V_tri.EY_cfit()

	sublist = [9]
	Mdst = par.dstr_gen(sublist)

	# EYCFINT_pert = EYconf(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent)
	# coeff_var0_pert, coeff_var1_pert = EYCFINT_pert.EY_linear_coeff()
	coeff_var0_pert = 1
	coeff_var1_pert = 0
	OPTIM_pert = Optimization(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,cost,budget,percent,order,lbd_mt,coeff_var0_pert,coeff_var1_pert)
	parE_pert = OPTIM_pert.optimization_EY()

	print('###########################################################')
	print(parE_pert)
	print(M_theta)
	print(budget)
	print(np.sum(np.multiply(M_theta-parE_pert,cost)))

	# EYCFINT_E_pert = EYconf(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,parE_tri,Mdst,percent)
	# EYCFINT_V_pert = EYconf(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,parV_tri,Mdst,percent)
	# sm_E_pert,svar_E_pert = EYCFINT_E_pert.EY_cfit()
	# sm_V_pert,svar_V_pert = EYCFINT_V_pert.EY_cfit()

	# import xlwt

	# book = xlwt.Workbook(encoding="utf-8")

	# sheet1 = book.add_sheet("Triangular Dstr")
	# sheet1.write(0, 0, "Node Number")
	# sheet1.write(0, 1, Node)
	# sheet1.write(1, 0, "Arc Number")
	# sheet1.write(1, 1, Arc)

	# sheet1.write(3, 0, "Sample Mean of First order")
	# sheet1.write(3, 1, sm_E_tri)
	# sheet1.write(4, 0, "Sample Variance of First order")
	# sheet1.write(4, 1, svar_E_tri)
	# sheet1.write(5, 0, "Sample Mean of Second order")
	# sheet1.write(5, 1, sm_V_tri)
	# sheet1.write(6, 0, "Sample Variance of Second order")
	# sheet1.write(6, 1, svar_V_tri)


	# sheet2 = book.add_sheet("PERT Dstr")

	# sheet2.write(0, 0, "Node Number")
	# sheet2.write(0, 1, Node)
	# sheet2.write(1, 0, "Arc Number")
	# sheet2.write(1, 1, Arc)

	# sheet2.write(3, 0, "Sample Mean of First order")
	# sheet2.write(3, 1, sm_E_pert)
	# sheet2.write(4, 0, "Sample Variance of First order")
	# sheet2.write(4, 1, svar_E_pert)
	# sheet2.write(5, 0, "Sample Mean of Second order")
	# sheet2.write(5, 1, sm_V_pert)
	# sheet2.write(6, 0, "Sample Variance of Second order")
	# sheet2.write(6, 1, svar_V_pert)

	# filename = "TimecostTradeoff" + str(Node) + "n" + str(Arc) + "a.xls"
	# book.save(filename)







	






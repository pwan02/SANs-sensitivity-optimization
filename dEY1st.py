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

class EY1stgradient:

	def __init__(self,N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,eps):
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
		self.eps = eps
	
	def RSE_measure(self,vec):
		#N = len(vec)
		#mean = np.mean(vec)
		std = math.sqrt(np.var(vec,ddof=1))
		#if mean != 0:
		rse = std/(math.sqrt(self.N_repl))#*mean)
		# else:
		# 	rse = 0
		return rse

	def RSE_measure_validation(self,vec,mean):
		N = len(vec)
		std = math.sqrt(np.var(vec,ddof=1))
		if mean != 0:
			rse = std/(math.sqrt(N)*mean)
		else:
			rse = 0
		return rse

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

	def RSE_average_validation(self,M,M2):
		rse = 0
		#count = 0
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			vec = np.zeros(self.N_repl)
			for n in range(self.N_repl):
				vec[n] = M[n,i,j]
			rse_k = self.RSE_measure_validation(vec,M2[i,j])
			rse += rse_k
		# 	if rse_k > 0:
		# 		count += 1
		# if count == 0:
		# 	average = 0
		# else:
		average = rse/self.Arc
		return average

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
			# if M2[i,j] != 0:
			# 	count += abs(M1[i,j] - M2[i,j])/M2[i,j]
			# else:
			# 	count += abs(M1[i,j])
		avg_count = count/self.Arc
		return avg_count

	def argmax_cord(self,mt):
		mv = np.max(mt)
		n = len(mt[0])
		for i in range(n):
			for j in range(n):
				if abs(mt[i,j] - mv) < 0.0001:
					max_i = i
					max_j = j
		return max_i,max_j

	def top_list(self,N,matrix):
		mt = copy.deepcopy(matrix)
		index = np.zeros([2,N])
		count = 0
		while -1:
			i,j = self.argmax_cord(mt)
			mt[i,j] = 0
			index[0][count], index[1][count] = i,j
			count += 1
			if count >= N:
				break
		index = index.astype(int)
		return index

	def top_cac(self,percent):
		M_ca_mean = np.zeros([N_repl,self.Node,self.Node])
		N = int(math.ceil(percent*self.Arc))
		for i in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			M_ca_mean[i,:,:],cn = EYGR.ca_struct()
		ca_mean = self.Mean_Matrix(M_ca_mean)
		tpl = self.top_list(N,ca_mean)
		return tpl

	def dEY_measure(self):
		M_measure_tac = np.zeros([N_repl,self.Node,self.Node])
		M_measure_cac = np.zeros([N_repl,self.Node,self.Node])
		M_measure_lr = np.zeros([N_repl,self.Node,self.Node])
		M_measure_wd = np.zeros([N_repl,self.Node,self.Node])
		tac_time = 0
		cac_time = 0
		lr_time = 0
		wd_time = 0
		for i in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			t0 = time.time()
			M_measure_tac[i,:,:] = EYGR.dEY_tac()
			t1 = time.time()
			M_measure_cac[i,:,:] = EYGR.dEY_cac()
			t2 = time.time()
			M_measure_lr[i,:,:] = EYGR.dEY_lr()
			t3 = time.time()
			M_measure_wd[i,:,:] = EYGR.dEY_wd_cmrv()
			t4 = time.time()

			tac_time += t1 - t0
			cac_time += t2 - t1
			lr_time += t3 - t2
			wd_time += t4 - t3

		tac_rse = self.RSE_average(M_measure_tac)
		cac_rse = self.RSE_average(M_measure_cac)
		lr_rse = self.RSE_average(M_measure_lr)
		wd_rse = self.RSE_average(M_measure_wd)

		tac_time = tac_time/N_repl
		cac_time = cac_time/N_repl
		lr_time = lr_time/N_repl
		wd_time = wd_time/N_repl

		return tac_rse, cac_rse, lr_rse, wd_rse #tac_time, cac_time, lr_time, wd_time 

	def dEY_cac_mean(self):
		N_large = 50*self.N_repl
		M_measure_fd = np.zeros([N_large,self.Node,self.Node])
		for i in range(N_large):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			M_measure_fd[i,:,:] = EYGR.dEY_cac()
		M_mean_fd = self.Mean_Matrix(M_measure_fd)
		return M_mean_fd

	def dEY_tac_mean(self):
		N_large = 5000 
		M_measure_fd = np.zeros([N_large,self.Node,self.Node])
		for i in range(N_large):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			M_measure_fd[i,:,:] = EYGR.dEY_tac()
		M_mean_fd = self.Mean_Matrix(M_measure_fd)
		return M_mean_fd

	def dEY_finite_diff_mean_nocmrv(self):
		N_large = 50*self.N_repl
		M_measure_fd = np.zeros([N_large,self.Node,self.Node])
		simu = np.zeros([self.Node,self.Node])
		for i in range(N_large):
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			M_measure_fd[i,:,:] = EYGR.dEY_finite_diff_nocmrv()
		M_mean_fd = self.Mean_Matrix(M_measure_fd)
		return M_mean_fd

	def dEY_finite_diff_mean_cmrv(self):
		N_large = 50*self.N_repl
		M_measure_fd = np.zeros([N_large,self.Node,self.Node])
		for i in range(N_large):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			M_measure_fd[i,:,:] = EYGR.dEY_finite_diff_cmrv()
		M_mean_fd = self.Mean_Matrix(M_measure_fd)
		return M_mean_fd

	def dEY_mean_compare(self):
		M_measure_tac = np.zeros([N_repl,self.Node,self.Node])
		M_measure_cac = np.zeros([N_repl,self.Node,self.Node])
		M_measure_lr = np.zeros([N_repl,self.Node,self.Node])
		M_measure_wd = np.zeros([N_repl,self.Node,self.Node])
		tac_time = 0
		cac_time = 0
		lr_time = 0
		wd_time = 0

		M = self.dEY_tac_mean()

		for i in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			t0 = time.time()
			M_measure_tac[i,:,:] = EYGR.dEY_tac()
			t1 = time.time()
			M_measure_cac[i,:,:] = EYGR.dEY_cac()
			t2 = time.time()
			M_measure_lr[i,:,:] = EYGR.dEY_lr()
			t3 = time.time()
			M_measure_wd[i,:,:] = EYGR.dEY_wd_cmrv()
			t4 = time.time()

			tac_time += t1 - t0
			cac_time += t2 - t1
			lr_time += t3 - t2
			wd_time += t4 - t3

		tac_mean = self.compare_matrix(self.Mean_Matrix(M_measure_tac),M)
		cac_mean = self.compare_matrix(self.Mean_Matrix(M_measure_cac),M)
		lr_mean = self.compare_matrix(self.Mean_Matrix(M_measure_lr),M)
		wd_mean = self.compare_matrix(self.Mean_Matrix(M_measure_wd),M)

		tac_rse = self.RSE_average(M_measure_tac)
		cac_rse = self.RSE_average(M_measure_cac)
		lr_rse = self.RSE_average(M_measure_lr)
		wd_rse = self.RSE_average(M_measure_wd)

		tac_time = tac_time/self.N_repl
		cac_time = cac_time/self.N_repl
		lr_time = lr_time/self.N_repl
		wd_time = wd_time/self.N_repl

		return tac_mean, cac_mean, lr_mean, wd_mean, tac_rse, cac_rse, lr_rse, wd_rse, tac_time, cac_time, lr_time, wd_time 

	def dEY_mean_seperate(self):
		M_measure_tac = np.zeros([N_repl,self.Node,self.Node])
		M_measure_cac = np.zeros([N_repl,self.Node,self.Node])
		M_measure_lr = np.zeros([N_repl,self.Node,self.Node])
		M_measure_wd = np.zeros([N_repl,self.Node,self.Node])
		M_measure_fd = np.zeros([N_repl,self.Node,self.Node])

		M = self.dEY_cac_mean()

		for i in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.eps)
			t0 = time.time()
			M_measure_tac[i,:,:] = EYGR.dEY_tac()
			t1 = time.time()
			M_measure_cac[i,:,:] = EYGR.dEY_cac()
			t2 = time.time()
			M_measure_lr[i,:,:] = EYGR.dEY_lr()
			t3 = time.time()
			M_measure_wd[i,:,:] = EYGR.dEY_wd_cmrv()
			t4 = time.time()

		tac_mean = self.Mean_Matrix(M_measure_tac)
		cac_mean = self.Mean_Matrix(M_measure_cac)
		lr_mean = self.Mean_Matrix(M_measure_lr)
		wd_mean = self.Mean_Matrix(M_measure_wd)

		return tac_mean, cac_mean, lr_mean, wd_mean, M

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
	theta_b = 1
	theta_c = 4
	eps = 0.05

	G = gnwk.generate_graph_am()
	nonzero_list = np.nonzero(G)
	graph_information = GraphInform(Node,Arc,G)
	net_struct = graph_information.network_struct()
	dictionary = graph_information.countpath()
	inform = graph_information.informt()

	par = ParaGen(Arc,Node,mu_b,mu_c,sigma_b,sigma_c,unif_loca_b,unif_loca_c,unif_scal_b,unif_scal_c,tri_left_b,tri_left_c,tri_right_b,tri_right_c,gamma_shap_b,gamma_shap_c,gamma_scal_b,gamma_scal_c,tnormal_left_b,tnormal_left_c,tnormal_right_b,tnormal_right_c,theta_b,theta_c,nonzero_list)
	M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta = par.paragenerate()
	sublist = [8]
	Mdst = par.dstr_gen(sublist)
	EY_tri = EY1stgradient(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,eps)
	tac_mean_t, cac_mean_t, lr_mean_t, wd_mean_t, tac_rse_t, cac_rse_t, lr_rse_t, wd_rse_t, tac_time_t, cac_time_t, lr_time_t, wd_time_t = EY_tri.dEY_mean_compare()
	
	sublist = [9]
	Mdst = par.dstr_gen(sublist)
	EY_pert = EY1stgradient(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,eps)
	tac_mean_p, cac_mean_p, lr_mean_p, wd_mean_p, tac_rse_p, cac_rse_p, lr_rse_p, wd_rse_p, tac_time_p, cac_time_p, lr_time_p, wd_time_p = EY_pert.dEY_mean_compare()
	#pert_tac,pert_cac,pert_lr,pert_wd,pert_fd = EY_pert.dEY_mean_seperate()

	import xlwt

	book = xlwt.Workbook(encoding="utf-8")

	# sheet1 = book.add_sheet("Triangular Dstr tac")
	# sheet2 = book.add_sheet("Triangular Dstr cac")
	# sheet3 = book.add_sheet("Triangular Dstr lr")
	# sheet4 = book.add_sheet("Triangular Dstr wd")
	# sheet5 = book.add_sheet("Triangular Dstr fd")

	# sheet6 = book.add_sheet("PERT Dstr tac")
	# sheet7 = book.add_sheet("PERT Dstr cac")
	# sheet8 = book.add_sheet("PERT Dstr lr")
	# sheet9 = book.add_sheet("PERT Dstr wd")
	# sheet10 = book.add_sheet("PERT Dstr fd")
	
	# for k in range(Arc):
	# 	i = int(nonzero_list[0][k])
	# 	j = int(nonzero_list[1][k])
	# 	sheet1.write(i,j,tri_tac[i,j])
	# 	sheet2.write(i,j,tri_cac[i,j])
	# 	sheet3.write(i,j,tri_lr[i,j])
	# 	sheet4.write(i,j,tri_wd[i,j])
	# 	sheet5.write(i,j,tri_fd[i,j])
	# 	sheet6.write(i,j,pert_tac[i,j])
	# 	sheet7.write(i,j,pert_cac[i,j])
	# 	sheet8.write(i,j,pert_lr[i,j])
	# 	sheet9.write(i,j,pert_wd[i,j])
	# 	sheet10.write(i,j,pert_fd[i,j])
	
	# book.save("dEY1st_mean_seperate2.xls")

	sheet1 = book.add_sheet("Triangular Dstr")
	sheet1.write(0, 0, "Node Number")
	sheet1.write(0, 1, Node)
	sheet1.write(1, 0, "Arc Number")
	sheet1.write(1, 1, Arc)

	sheet1.write(3, 0, "tac bias")
	sheet1.write(3, 1, tac_mean_t)
	sheet1.write(4, 0, "cac bias")
	sheet1.write(4, 1, cac_mean_t)
	sheet1.write(5, 0, "lr bias")
	sheet1.write(5, 1, lr_mean_t)
	sheet1.write(6, 0, "wd bias")
	sheet1.write(6, 1, wd_mean_t)

	sheet1.write(3,2,tac_rse_t)
	sheet1.write(4,2,cac_rse_t)
	sheet1.write(5,2,lr_rse_t)
	sheet1.write(6,2,wd_rse_t)

	sheet1.write(8, 0, "tac time")
	sheet1.write(8, 1, tac_time_t)
	sheet1.write(9, 0, "cac time")
	sheet1.write(9, 1, cac_time_t)
	sheet1.write(10, 0, "lr time")
	sheet1.write(10, 1, lr_time_t)
	sheet1.write(11, 0, "wd time")
	sheet1.write(11, 1, wd_time_t)


	sheet2 = book.add_sheet("PERT Dstr")

	sheet2.write(0, 0, "Node Number")
	sheet2.write(0, 1, Node)
	sheet2.write(1, 0, "Arc Number")
	sheet2.write(1, 1, Arc)

	sheet2.write(3, 0, "tac bias")
	sheet2.write(3, 1, tac_mean_p)
	sheet2.write(4, 0, "cac bias")
	sheet2.write(4, 1, cac_mean_p)
	sheet2.write(5, 0, "lr bias")
	sheet2.write(5, 1, lr_mean_p)
	sheet2.write(6, 0, "wd bias")
	sheet2.write(6, 1, wd_mean_p)

	sheet2.write(3,2,tac_rse_p)
	sheet2.write(4,2,cac_rse_p)
	sheet2.write(5,2,lr_rse_p)
	sheet2.write(6,2,wd_rse_p)

	sheet2.write(8, 0, "tac time")
	sheet2.write(8, 1, tac_time_p)
	sheet2.write(9, 0, "cac time")
	sheet2.write(9, 1, cac_time_p)
	sheet2.write(10, 0, "lr time")
	sheet2.write(10, 1, lr_time_p)
	sheet2.write(11, 0, "wd time")
	sheet2.write(11, 1, wd_time_p)

	filename = "dEY1st_mean_" + str(Node) + "n" + str(Arc) + "a_new.xls"
	book.save(filename)









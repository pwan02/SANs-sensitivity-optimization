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

class EYdifference:
	
	def __init__(self,N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent,order,low_bd):
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
		self.order = order
		self.low_bd = low_bd

	def subset_ca(self):
		N_large = 2000 
		M_ca = np.zeros([N_large,self.Node,self.Node])
		for i in range(N_large):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)
			M_ca[i,:,:],cn = EYGR.ca_struct()
			ca = self.Mean_Matrix(M_ca)
		sublist_long = np.zeros([self.Arc,self.Arc])
		ct = 0
		for k in range(self.Arc):
			i = int(self.nonzero_list[0][k])
			j = int(self.nonzero_list[1][k])
			if ca[i,j] >= self.low_bd:
				sublist_long[i,j] = 1
		sublist_ca = np.nonzero(sublist_long) 
		return sublist_ca

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
		for k in range(len(self.nonzero_list[0])):
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
		num = 0
		for k in range(self.Arc):
			i = self.nonzero_list[0][k]
			j = self.nonzero_list[1][k]
			if M2[i,j] > 0:
				count += abs(M1[i,j] - M2[i,j])
				num += 1
			else:
				count += 0
		avg_count = count/num
		return avg_count

	def RSE_measure(self,vec):
		std = math.sqrt(np.var(vec,ddof=1))
		rse = std/(math.sqrt(self.N_repl))
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

	def simulate_dictionary(self):
		dist_dic = {}
		for k in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			dist_dic[k] = simu_sing.SimuArcs()
		return dist_dic

	def EY_diff_mean(self):
		N_large = 5000 
		M_measure_fd = np.zeros([N_large,self.Node,self.Node])
		for i in range(N_large):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)
			M_measure_fd[i,:,:] = EYGR.EY_finite_diff_cmrv()
			M_mean_fd = self.Mean_Matrix(M_measure_fd)
		return M_mean_fd
	
	## test for different order of taylor expansions ##
	def EY_diff_estimate_seperate_compare(self):
		M_EY_diff0 = np.zeros([self.N_repl,self.Node,self.Node])
		M_EY_diff1 = np.zeros([self.N_repl,self.Node,self.Node])
		M_EY_diff2 = np.zeros([self.N_repl,self.Node,self.Node])
		M_EY_diff3 = np.zeros([self.N_repl,self.Node,self.Node])
		t0 = 0
		t1 = 0
		t2 = 0
		t3 = 0 

		time_fd0 = time.time()
		M = self.EY_diff_mean()
		time_fd1 = time.time()
		
		t_fd = time_fd1 - time_fd0

		for k in range(self.N_repl):
			time_s = time.time()
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)
			time0 = time.time()
			M_EY_diff0[k,:,:] = EYGR.EY_taylor_est(0,self.percent)
			time1 = time.time()
			M_EY_diff1[k,:,:] = EYGR.EY_taylor_est(1,self.percent)
			time2 = time.time()
			M_EY_diff2[k,:,:] = EYGR.EY_taylor_est(2,self.percent)
			time3 = time.time()
			M_EY_diff3[k,:,:] = EYGR.EY_taylor_est(3,self.percent)
			time4 = time.time()

			t0 += time1 - time0 + time0 - time_s
			t1 += time2 - time1 + time0 - time_s
			t2 += time3 - time2 + time0 - time_s
			t3 += time4 - time3 + time0 - time_s

		diff_mean0 = self.compare_matrix(self.Mean_Matrix(M_EY_diff0),M)
		diff_mean1 = self.compare_matrix(self.Mean_Matrix(M_EY_diff1),M)
		diff_mean2 = self.compare_matrix(self.Mean_Matrix(M_EY_diff2),M)
		diff_mean3 = self.compare_matrix(self.Mean_Matrix(M_EY_diff3),M)

		diff_rse0 = self.RSE_average(M_EY_diff0)
		diff_rse1 = self.RSE_average(M_EY_diff1)
		diff_rse2 = self.RSE_average(M_EY_diff2)
		diff_rse3 = self.RSE_average(M_EY_diff3)
		
		return diff_mean0, diff_mean1, diff_mean2, diff_mean3, diff_rse0, diff_rse1, diff_rse2, diff_rse3, t0, t1, t2, t3, t_fd

	## Calculate the difference of EY ##

	def EY_diff_ratio(self):
		M_EY_diff = np.zeros([self.N_repl,self.Node,self.Node])
		for k in range(self.N_repl):
			simu_sing = Simulate(self.Arc,self.Node,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.nonzero_list)
			simu_arclt = simu_sing.SimuArcs()
			EYGR = GradientEst(self.Arc,self.Node,simu_arclt,self.G,self.dictionary,self.inform,self.net_struct,self.nonzero_list,self.M_mu,self.M_sigma,self.M_unif_loca,self.M_unif_scal,self.M_tri_a,self.M_tri_b,self.M_tri_c,self.M_gamma_shap,self.M_gamma_scal,self.M_tnormal_a,self.M_tnormal_b,self.M_theta,self.Mdst,self.percent)
			M_EY_diff[k,:,:] = EYGR.EY_taylor_ratio(self.order,self.percent)
		EY_diff_est = self.Mean_Matrix(M_EY_diff)
		return EY_diff_est

	def tac_diff_ratio(self):
		M_diff = self.EY_diff_ratio()
		cost_ratio = self.diff_ratio(M_diff)
		return cost_ratio

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
	theta_b = 0.5
	theta_c = 29.5
	cost_b = 1
	cost_c = 9
	percent = 0.15
	order = 3
	low_bd = 0.05

	G = gnwk.generate_graph_am()
	nonzero_list = np.nonzero(G)
	graph_information = GraphInform(Node,Arc,G)
	net_struct = graph_information.network_struct()
	dictionary = graph_information.countpath()
	inform = graph_information.informt()

	par = ParaGen(Arc,Node,mu_b,mu_c,sigma_b,sigma_c,unif_loca_b,unif_loca_c,unif_scal_b,unif_scal_c,tri_left_b,tri_left_c,tri_right_b,tri_right_c,gamma_shap_b,gamma_shap_c,gamma_scal_b,gamma_scal_c,tnormal_left_b,tnormal_left_c,tnormal_right_b,tnormal_right_c,theta_b,theta_c,cost_b,cost_c,nonzero_list)
	M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta = par.paragenerate()
	sublist = [8]
	Mdst = par.dstr_gen(sublist)
	EY_tri_test = EYdifference(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent,order,low_bd)
	subset_tri = EY_tri_test.subset_ca()
	subarc_tri = len(subset_tri[0])
	EY_tri = EYdifference(N_repl,subarc_tri,Node,G,dictionary,inform,net_struct,subset_tri,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent,order,low_bd)
	diff_mean0_t, diff_mean1_t, diff_mean2_t, diff_mean3_t, diff_rse0_t, diff_rse1_t, diff_rse2_t, diff_rse3_t, t0_t, t1_t, t2_t, t3_t, t_fd_t = EY_tri.EY_diff_estimate_seperate_compare()
	
	sublist = [9]
	Mdst = par.dstr_gen(sublist)
	EY_pert_test = EYdifference(N_repl,Arc,Node,G,dictionary,inform,net_struct,nonzero_list,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent,order,low_bd)
	subset_pert = EY_pert_test.subset_ca()
	subarc_pert = len(subset_pert[0])
	EY_pert = EYdifference(N_repl,subarc_pert,Node,G,dictionary,inform,net_struct,subset_pert,M_mu,M_sigma,M_unif_loca,M_unif_scal,M_tri_a,M_tri_b,M_tri_c,M_gamma_shap,M_gamma_scal,M_tnormal_a,M_tnormal_b,M_theta,Mdst,percent,order,low_bd)
	diff_mean0_p, diff_mean1_p, diff_mean2_p, diff_mean3_p, diff_rse0_p, diff_rse1_p, diff_rse2_p, diff_rse3_p, t0_p, t1_p, t2_p, t3_p, t_fd_p = EY_pert.EY_diff_estimate_seperate_compare()

	import xlwt

	book = xlwt.Workbook(encoding="utf-8")

	sheet1 = book.add_sheet("Triangular Dstr")
	sheet1.write(0, 0, "Node Number")
	sheet1.write(0, 1, Node)
	sheet1.write(1, 0, "Arc Number")
	sheet1.write(1, 1, Arc)

	sheet1.write(3, 0, "1st bias")
	sheet1.write(3, 1, diff_mean0_t)
	sheet1.write(4, 0, "2nd bias")
	sheet1.write(4, 1, diff_mean1_t)
	sheet1.write(5, 0, "3rd order bias")
	sheet1.write(5, 1, diff_mean2_t)
	sheet1.write(6, 0, "4th bias")
	sheet1.write(6, 1, diff_mean3_t)

	sheet1.write(3,2,diff_rse0_t)
	sheet1.write(4,2,diff_rse1_t)
	sheet1.write(5,2,diff_rse2_t)
	sheet1.write(6,2,diff_rse3_t)

	sheet1.write(8, 0, "1st time")
	sheet1.write(8, 1, t0_t)
	sheet1.write(9, 0, "2nd time")
	sheet1.write(9, 1, t1_t)
	sheet1.write(10, 0, "3rd time")
	sheet1.write(10, 1, t2_t)
	sheet1.write(11, 0, "4th time")
	sheet1.write(11, 1, t3_t)
	sheet1.write(12, 0, "fd time")
	sheet1.write(12, 1, t_fd_t)


	sheet2 = book.add_sheet("PERT Dstr")

	sheet2.write(0, 0, "Node Number")
	sheet2.write(0, 1, Node)
	sheet2.write(1, 0, "Arc Number")
	sheet2.write(1, 1, Arc)

	sheet2.write(3, 0, "1st bias")
	sheet2.write(3, 1, diff_mean0_p)
	sheet2.write(4, 0, "2nd bias")
	sheet2.write(4, 1, diff_mean1_p)
	sheet2.write(5, 0, "3rd bias")
	sheet2.write(5, 1, diff_mean2_p)
	sheet2.write(6, 0, "4th bias")
	sheet2.write(6, 1, diff_mean3_p)

	sheet2.write(3,2,diff_rse0_p)
	sheet2.write(4,2,diff_rse1_p)
	sheet2.write(5,2,diff_rse2_p)
	sheet2.write(6,2,diff_rse3_p)

	sheet2.write(8, 0, "1st time")
	sheet2.write(8, 1, t0_p)
	sheet2.write(9, 0, "2nd time")
	sheet2.write(9, 1, t1_p)
	sheet2.write(10, 0, "3rd time")
	sheet2.write(10, 1, t2_p)
	sheet2.write(11, 0, "4th time")
	sheet2.write(11, 1, t3_p)
	sheet2.write(12, 0, "fd time")
	sheet2.write(12, 1, t_fd_p)

	filename = "EY_diff_sublist" + str(Node) + "n" + str(Arc) + "a.xls"
	book.save(filename)




	






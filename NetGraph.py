## Author: Peng Wan
## Date: 08/23/2021


import networkx as nx
import numpy as np
import math
import random
import time
import statistics
from collections import deque 

#### Generate a random structure SAN ####
class RmGraphGen:
	
	def __init__(self,Node,Arc):
		self.Node = Node
		self.Arc = Arc

	def upper_m(self):
		n = self.Node
		mt = np.zeros([n,n])
		for i in range(n):
			for j in range(n):
				if j>i:
					mt[i,j] = 1
		return mt

	def generate_j(self):
		n = self.Node
		y = np.random.uniform(low=0,high=1)
		j = math.floor(n + 0.5 - math.sqrt(n*(n-1)*y + 0.25))
		return j

	def generate_k(self,j):
		n = self.Node
		y = np.random.uniform(low=0,high=1)
		k = math.floor(j + 1 + y*(n-j))
		return k

	def check_condition(self,mt,j):
		n = self.Node
		index0 = 0
		index1 = 1
		if np.count_nonzero(mt[j-1,:]) == 1:
			index0 = 1
		for k in range(n):
			if k > j - 1 and np.count_nonzero(mt[:,k])!=1:
				index1 = 0
		if index0 + index1 > 0:
			rt = 1
		else:
			rt = 0
		return rt
		   
	def step2(self,initial):
		n = self.Node
		test_j = 0
		while test_j > -1:
			test_j = self.generate_j()
			check = self.check_condition(initial,test_j)
			if check == 0:
				j = test_j
				break
		it = 0
		while it > -1:
			test_k = self.generate_k(j)
			it += 1
			if np.count_nonzero(initial[:,test_k-1]) != 1:
				k = test_k
				break
		return j,k

	def generate_graph_dm(self):
		initial = self.upper_m()
		N = self.Node
		A = self.Arc
		L = 0
		K = N*(N-1)*0.5 - A
		loop2 = 0
		while loop2 > -1:
			j,k = self.step2(initial)
			loop2 += 1
			if initial[j-1,k-1] != 0:
				initial[j-1,k-1] = 0
				L += 1
				if L >= K:
					 break
		return initial


	### Addition Method of generating random networks ###
	def generate_graph_am(self):
		N = self.Node
		A = self.Arc		
		initial = np.zeros([N,N])
		initial[0,1] = 1
		initial[N-2,N-1] = 1
		L = 2
		m = N - 3
		n = N - 3
		F = A - L - m - n
		while 1:
			if F > 0:
				while 1:
					j = self.generate_j()
					k = self.generate_k(j)
					if initial[j-1,k-1] != 1:
						if sum(initial[j-1,:]) == 0:
							n = n - 1
						if sum(initial[:,k-1]) == 0:
							m = m - 1
						initial[j-1,k-1] = 1
						L += 1
						F = A - L - m - n
						break
			else:
				if m > 0:
					for i in range(N):
						if sum(initial[:,i]) == 0 and i > 0:
							k = i + 1
							break
					y = np.random.uniform(low=0,high=1)
					j = math.floor(1 + (k - 1)*y)
					if sum(initial[j-1,:]) == 0:
							n = n - 1
					if sum(initial[:,k-1]) == 0:
							m = m - 1
					initial[j-1,k-1] = 1
					L += 1
					F = A - L - m - n 
				else:
					if n > 0:
						for i in range(N):
							if sum(initial[i,:]) == 0 and i < N-1:
								j = i + 1
								break
						y = np.random.uniform(low=0,high=1)
						k = math.floor(j + 1 + (N - j)*y)
						if sum(initial[j-1,:]) == 0:
							n = n - 1
						if sum(initial[:,k-1]) == 0:
							m = m - 1
						initial[j-1,k-1] = 1
						L += 1
						F = A - L - m - n 
					else:
						while L > A:
							DL = 0
							K = L - A
							while DL < K:
								j,k = self.step2(initial)
								if initial[j-1,k-1] != 0:
									initial[j-1,k-1] = 0
									DL += 1
						break
		return initial


#### Extract All Network Informations ####
class GraphInform:

	def __init__(self,Node,Arc,G):
		self.Node = Node
		self.Arc = Arc
		self.G = G

	## Calculate the structure for Bowman ##
	def network_struct(self):
		test_arc_left = self.G
		mt = self.G
		n = self.Node
		a = np.count_nonzero(test_arc_left)
		c = np.zeros([n,n])
		cn = np.zeros(n)
		dic= np.zeros([n+a-2,3])
		cn[n-1] = 1
		
		k = 0
		for j in range(n):
			for i in range(n):
				if mt[n - 1 - i,n - 1 - j] != 0 and cn[n - 1 -j] != 0:
					c[n - 1 - i,n - 1 - j] = test_arc_left[n - 1 - i,n - 1 - j]*cn[n - 1 - j]
					dic[k,0] = 1
					## operation is to calculate c when 1 ##
					dic[k,1] = n - 1 - i
					dic[k,2] = n - 1 - j
					k += 1
				if np.count_nonzero(c[n - 1 - i,:]) == np.count_nonzero(mt[n - 1 - i,:]) and i!=0 and cn[n - 1 - i] == 0 and n - 1 - i != 0:
					cn[n - 1 - i] = sum(c[n - 1 - i,:])
					dic[k,0] = 0
					## operation is to calculate c when 0 ##
					dic[k,1] = n - 1 - i
					k += 1 
		output = dic.astype(int)
		return output
	 
	### Calculate All paths in the network ###
	def countpath(self):
		mt = self.G
		dictionary = {}
		target = len(mt[0,:])
		initial = np.array([0])

		q = deque()
		q.append(initial)
		k = 0
		while 1:
			start = q[0][-1]
			if start == target - 1:
				dictionary[k] = q[0]
				k = k + 1
				q.popleft()
			else:
				lst = np.nonzero(mt[start,:])[0]
				for j in range(len(lst)):
					q.append(np.append(q[0],lst[j]))
				q.popleft()
			if len(q) == 0:
				break
		return dictionary      

	def informt(self):
		n = self.Node
		dit = self.countpath()
		inform = np.zeros([len(dit),n,n])
		for i in range(len(dit)):
			for j in range(len(dit[i]) - 1):
				inform[i,dit[i][j],dit[i][j+1]] = 1
		return inform






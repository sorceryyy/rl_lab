import numpy as np
import random

from abc import abstractmethod
from collections import defaultdict

class QAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


class my_QAgent(QAgent):
	def __init__(self):
		super().__init__()
		self.alpha = 0.1  #learning rate
		self.gamma = 0.01 #衰减系数
		self.q_val = defaultdict(lambda:[0.0 , 0.0 , 0.0 , 0.0])

	def select_action(self,obs):
		'''select an action without exploration'''
		q_action = self.q_val[str(obs)]	#TODO:check whether correct
		max_val = -1
		ans = 0
		for i in range(4):
			if q_action[i] > max_val:
				max_val = q_action[i]
				ans = i
		return ans

	def update(self,obs_before,obs_next,action:int,reward):
		'''update the Q function'''
		qs_before = self.q_val[str(obs_before)]
		qs_next = self.q_val[str(obs_next)]
		image_act = self.select_action(obs_next)
		qs_before[action] += self.alpha*(reward+self.gamma*qs_next[image_act]-qs_before[action])
				
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

import scipy.io as io
from LearningAlgorithm import *

class Node:
	def __init__(self,LayerNumber,NodePos):
		self.LayerNumber = LayerNumber
		self.NodePosition = NodePos 

	def initLearningAlgorithm(self,AlgorithmChoice,AlgParams,InitNodeBelief,InitNodeLearnedFeatures):
		self.AlgorithmChoice = AlgorithmChoice
		if AlgorithmChoice == 'LogRegression':
			self.AlgorithmChoice = AlgorithmChoice # Name of the Algorithm
			self.Belief = InitNodeBelief
			self.LearnedFeatures = InitNodeLearnedFeatures
			#Here do some LogRegression Specific Variable Assignments
			#Attrbutes For the learning Algorithm Class
			AlgParams['D'] = [self.Input,np.random.randint(size=AlgParams['N'], low=0, high=2)]
			self.LearningAlgorithm = LearningAlgorithm(AlgParams)
			self.LearningAlgorithm.D = [self.Input,np.random.randint(size=AlgParams['N'], low=0, high=2)]
			self.LearningAlgorithm.N = AlgParams['N']
			self.LearningAlgorithm.training_steps = AlgParams['training_steps']
			self.LearningAlgorithm.feats = AlgParams['feats']
			self.LearningAlgorithm.w = AlgParams['w']
	def loadInput(self,In):
		self.Input = In
	def doLearning(self, Mode):
		if self.AlgorithmChoice == 'LogRegression':
			self.LearningAlgorithm.runLearningAlgorithm(Mode)
			self.Belief = self.LearningAlgorithm.w * self.LearningAlgorithm.D[0]
			self.LearnedFeatures = self.LearningAlgorithm.w
		else:
			print("only LogRegression Exists")

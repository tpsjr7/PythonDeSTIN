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
			self.LearningAlgorithm = LearningAlgorithm(AlgParams)
			self.LearningAlgorithm.D = AlgParams['D']
			self.LearningAlgorithm.N = AlgParams['N']
			self.LearningAlgorithm.training_steps = AlgParams['training_steps']
			self.LearningAlgorithm.feats = AlgParams['feats']
			self.LearningAlgorithm.w = AlgParams['w']
	def loadInput(self,Input):
		self.Input = Input
	def doLearning(self, Mode):
		if self.AlgorithmChoice == 'LogRegression':
			self.LearningAlgorithm.runLearningAlgorithm(Mode)
			self.Belief = self.LearningAlgorithm.w * self.LearningAlgorithm.D[0]
			self.LearnedFeatures = self.LearningAlgorithm.w
		else:
			


'''
Steps to train a node
	1. Declare The Node Object with LayerNum and LayerPosition
	2. Initialize LearningAlgorithm
	3. Load the node with Input
	3. doLearning by specifying the Mode
	 
'''

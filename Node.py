# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

import numpy as np
import scipy.io as io
from random import randrange
import theano
from theano import function
import theano.tensor as T
from LearningAlgorithm import *
rng = np.random

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
			#Attrbutes For the learning Algorithm Class
			self.LearningAlgorithm = LearningAlgorithm(AlgParams)
			self.LearningAlgorithm.D = AlgParams['D']
			self.LearningAlgorithm.N = AlgParams['N']
			self.LearningAlgorithm.training_steps = AlgParams['training_steps']
			self.LearningAlgorithm.feats = AlgParams['feats']
			self.LearningAlgorithm.w = AlgParams['w']
		else:
			print('make sure that you are choosing an available learning algorithm')
			print('python is exitting')
			exit(0)

	def loadInput(self,Input):
		self.Input = Input

	def doLearning(self, Mode):
		self.LearningAlgorithm.runLearningAlgorithm(Mode)
		self.Belief = self.LearningAlgorithm.w * self.LearningAlgorithm.D[0]
		self.LearnedFeatures = self.LearningAlgorithm.w
'''
Steps to train a node
	1. Declare The Node Object with LayerNum and LayerPosition
	2. Initialize LearningAlgorithm
	3. Load the node with Input
	3. doLearning by specifying the Mode
	 
'''

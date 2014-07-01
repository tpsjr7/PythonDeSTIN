# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 2014
@author: teddy
"""

import numpy as np
import scipy.io as io
from random import randrange
import theano.tensor as TT
from theano import function
class Node:
	def __init__(self,LayerNumber,NodePos):
		self.LayerNumber = LayerNumber
		self.NodePosition = NodePos 

	def initLearningAlgorithm(self,AlgorithmChoice,AlgParams,InitNodeBelief,InitNodeLearnedFeatures):
		self.AlgorithmChoice = AlgorithmChoice
		if AlgorithmChoice == 'SparseAutoEncoder':
			InitialWeightMatrix = InitNodeLearnedFeatures #
			self.LearningAlgorithm = SparseAutoEncoder(AlgParams,InitNodeLearnedFeatures)
		elif AlgorithmChoice == 'KMeansClustering':
			#InitialCentroids = InitNodeLearnedFeatures #
			#self.LearningAlgorithm = KMClustering(AlgParams,InitialCentroids)
		elif AlgorithmChoice == 'EvolutionaryLearning':
			# God Knows what to do here :D
		elif AlgorithmChoice == 'MiscLearning':
			# We'll write new stuff here, like combining two learning algorithms
		else:
			print('make sure that you are choosing an available learning algorithm')
			print('python is exitting')
			exit(0)

	def loadInput(self,Input):
		self.Input = Input

	def doLearning(self, Mode):
		self.LearningAlgorithm.runLearningAlgorithm(Mode) 
		# Mode Differentiates Training from Testing 
		# Mode == 1 Training: there will be update of Model/ or parameters
		# Mode == 0 Testing: No model-updating only encoding
'''
Steps to train a node
	1. Declare The Node Object with LayerNum and LayerPosition
	2. Initialize LearningAlgorithm
	3. Load the node with Input
	3. doLearning by specifying the Mode
	 
'''

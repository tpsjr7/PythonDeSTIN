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
rng = np.random
from LearningAlgorithm import *
from Node import *
def main():
	# Declare Node 
	myNode = Node(1,[2,2]) # LayerNum=1, LayerPos=[2,2]
	# Prepare AlgParams,InitNodeBelief,InitNodeLearnedFeatures
	N = 400
	feats = 784
	AlgParams = {}
	D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
	AlgParams['D'] = D
	AlgParams['N'] = 400
	AlgParams['feats'] = feats
	AlgParams['training_steps'] = 10000
	AlgParams['w'] = theano.shared(rng.randn(feats), name="w")
	myLearningAlgorithm = LearningAlgorithm(AlgParams)
	InitNodeBelief = AlgParams['w']*D[0]
	InitNodeLearnedFeatures = AlgParams['w']
	myNode.initLearningAlgorithm('LogRegression',AlgParams,InitNodeBelief,InitNodeLearnedFeatures)
	myNode.loadInput(D)
	myNode.doLearning(1)
main()

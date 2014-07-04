# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 2014
@author: teddy
"""
from loadData import *
import numpy as np
from Node import *

class Layer:

	def __init__(self,LayerNum,NumberOfNodes,PatchMode=None,ImageType=None):
		self.PatchMode = PatchMode
		self.ImageType = ImageType
		self.LayerNumber = LayerNum
		self.NumberOfNodes = NumberOfNodes
		Row = NumberOfNodes[0]
		Col = NumberOfNodes[1]
		Nodes = [[Node(LayerNum,[i,j]) for j in range(Row)] for i in range(Col)]
		self.Nodes = Nodes

	def loadInput(self,Input,Ratio):
		# Ratio tells equals to the number of lower layer units getting combined and being fed to the upper layer
		# 1. LayerNumber==0 getting Layer is getting input from raw image
		# 2. LayerNumber>=1 getting Layer is getting input from beliefs of lower layer Nodes i.e the Nodes matrix passed up
		if self.LayerNumber == 0:
			Nx = 0 # X coordinate of the current node
			for I in range(0,len(Input[0]),Ratio):
				Ny = 0 # Y coordinate of the current node
				for J in range(0,len(Input[1]),Ratio):
					self.Nodes[Nx][Ny].loadInput(returnNodeInput(Input,[I,J],Ratio,self.PatchMode,self.ImageType)	) # returns inputs to the node located at Position [I,J]
		else:
			Nx = 0 # X coordinate of the current node
			Ny = 0 # Y coordinate of the curret node
			for I in range(0,len(Input[0]),Ratio):
				Ny = 0
				for J in range(0,len(Input[1]),Ratio):
					InutTemp = np.array([])
					for K in range(I,I+Ratio):
						for L in range(J,J+Ratio):
							InutTemp = np.concatenate((InutTemp,Input[K][L].Belief))# Combine the Beliefs of Nodes
					self.Nodes[Nx][Ny].loadInput(InputTemp)
					Ny += 1
				Nx += 1
	def intializeLayerLearning(self,AlgorithmChoice,AlgParams,InitNodeBelief,InitNodeLearnedFeatures):
		for I in range(len(self.Nodes[0])):
			for J in range(len(self.Nodes[1])):
				self.Nodes[I][J].initLearningAlgorithm(AlgorithmChoice,AlgParams,InitNodeBelief,InitNodeLearnedFeatures)
	def doLayerLearning(self,Mode):
		for I in range(len(self.Nodes[0])):
			for J in range(len(self.Nodes[1])):
				self.Nodes[I][J].doLearning(Mode)	

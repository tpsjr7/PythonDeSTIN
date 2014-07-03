# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 2014
@author: teddy
"""

# Layer Class isn't comleted yet coz I first have to decide how to handle Images
# I can use pylearn2 (with theano) or cv2 (opencv python)
# On Thursday and Friday I have to complete Layer Class
class Layer:
	def __init__(self,LayerNum,NumberOfNodes):
		self.LayerNumber = LayerNum
		self.NumberOfNodes = NumberOfNodes
		Row = NumberOfNodes[0]
		Col = NumberOfNodes[1]
		Nodes = [[Node(LayerNum,[i,j]) for j in range(Row)] for i in range(Col)]
		self.Nodes = Nodes
	def loadInput(self,Input,Ratio):
		#ratio tells us about the number of lower layer units getting combined
		# and being fed to the upper layer
		# Here there will be two general cases which should treat differently
		# 1. LayerNumber==0 getting Layer is getting input from raw image
		# 2. LayerNumber>=1 getting Layer is getting input from beliefs of lower layer Nodes i.e the Nodes matrix passed up
		if self.LayerNumber == 0:
			#do the following two things for every Node of the layer
			#rearrange its corresponding image patch (of size Ratio x Ratio x 3) into a theano tensor vector of len Ratio*Ratio*3
			#then run loadInput(InputVector) method of the node 
		else:
			for I in range(0,len(Input[0]),Ratio):
				for J in range(0,len(Input[1]),Ratio):
					# Combine the Beliefs of Nodes[I:I+Ratio][J:J+Ratio]
					# into a new vector
					# Pass the new vector as input to the the current layer

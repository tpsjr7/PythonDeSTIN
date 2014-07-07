# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

from LearningAlgorithm import *


class Node:
    def __init__(self, LayerNumber, NodePos):
        self.LayerNumber = LayerNumber
        self.NodePosition = NodePos
        self.Belief = []
        self.LearnedFeatures = []
        self.LearningAlgorithm = []
        self.AlgorithmChoice = []
        self.Input = []

    def initNodeLearningParams(self, AlgorithmChoice, AlgParams):
        self.AlgorithmChoice = AlgorithmChoice
        if AlgorithmChoice == 'LogRegression':
            # Here do some LogRegression Specific Variable Assignments
            #Attrbutes For the learning Algorithm Class
            #AlgParams['D'] = [self.Input, np.random.randint(size=AlgParams['N'], low=0, high=2)]
            self.Belief = AlgParams['w'] * AlgParams['D'][0]
            self.LearnedFeatures = AlgParams['w']
            #Nodes = [[Node(LayerNum,[i,j]) for j in range(Row)] for i in range(Col)]
            self.LearningAlgorithm = LearningAlgorithm(AlgParams)

    def loadInput(self, In):
        self.Input = In

    def doNodeLearning(self, Mode):
        if self.AlgorithmChoice == 'LogRegression':
            self.LearningAlgorithm.runLearningAlgorithm(Mode)
            self.Belief = self.LearningAlgorithm.w * self.LearningAlgorithm.D[0]
            self.LearnedFeatures = self.LearningAlgorithm.w
        else:
            print("only LogRegression Exists")
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 2014
@author: teddy
"""

from LearningAlgorithm import *
from Clustering import *

class Node:
    def __init__(self, LayerNumber, NodePos):
        self.LayerNumber = LayerNumber
        self.NodePosition = NodePos
        self.Belief = []

    def initNodeLearningParams(self, AlgorithmChoice, AlgParams):
        self.AlgorithmChoice = AlgorithmChoice
        if AlgorithmChoice == 'Clustering':
            InputWidths = AlgParams['NumCentsPerLayer']
            # InputWidth = InputWidths[LayerNum]
            if self.LayerNumber == 0:
                InputWidth = 48
            else:
                InputWidth = InputWidths[self.LayerNumber] * 4
            self.LearningAlgorithm = Clustering(AlgParams['mr'], AlgParams['vr'], AlgParams['sr'], InputWidth,
                                                AlgParams['NumCentsPerLayer'][self.LayerNumber], self.NodePosition)
        else:
            print('Only Incremental Clustering Exists')

    def loadInput(self, In):
        self.Input = In

    def doNodeLearning(self, Mode):
        if self.AlgorithmChoice == 'Clustering':
            self.LearningAlgorithm.update_node(self.Input, Mode)
            self.Belief = self.LearningAlgorithm.belief
        else:
            print("only Incremental Clustering Algorithm Exists")
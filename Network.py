# -*- coding: utf-8 -*-
__author__ = 'teddy'
from Layer import *
import scipy.io as io
#io.savemat(FileName,Dict,True)
# TODO: get ridoff the sequential requirements like first feed the layer an input the you can initialize it

class Network():

    def __init__(self, numLayers, AlgChoice, AlgParams, NumNodesPerLayer, PatchMode='Adjacent', ImageType='Color'):
        self.NetworkBelief = {}
        self.NetworkBelief['data'] = [] # this is going to store beliefs for every image DeSTIN sees
        self.saveBeliefOption = 'True'
        self.BeliefFileName = 'Beliefs.mat'
        self.NumberOfLayers = numLayers
        self.AlgorithmChoice = AlgChoice
        self.AlgorithmParams = AlgParams
        self.NumberOfNodesPerLayer = NumNodesPerLayer
        self.PatchMode = PatchMode
        self.ImageType = ImageType
        self.Layers = [[Layer(j, NumNodesPerLayer[j], self.PatchMode, self.ImageType) for j in range(numLayers)]]

    def setMode(self, Mode):
        self.OperatingMode = Mode

    def initLayer(self, LayerNum):  # TODO make sure lower layer is initialized (or trained at least once)
        self.Layers[0][LayerNum].initLayerLearningParams(self.AlgorithmChoice, self.AlgorithmParams)

    def trainLayer(self, LayerNum):
        self.Layers[0][LayerNum].doLayerLearning(self.OperatingMode)

    def updateBeliefExporter(self):
        for i in range(len(self.NumberOfLayers)):
            for j in range(len(self.Layers[0][i].NumberOfNodes[0])):
                for k in range(len(self.Layers[0][i].NumberOfNodes[1])):
                    self.NetworkBelief['data'].append(self.Layers[0][i].Nodes[j][k].Belief)
                    #self.Belief['labels'].append(self.Layers[0][i].Nodes[j][k].Label)
                    #label is dropped from dumping coz it doesn't exist when testing

    def dumpBelief(self):
        io.save_as_module(self.BeliefFileName,self.NetworkBelief)
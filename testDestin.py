__author__ = 'teddy'
from Network import *
import numpy.random as rng
#*****Define Parameters for the Network and Nodes
#Network Params
numLayers = 4
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
NumCentsPerLayer = [25,25,25,25]
PatchMode = 'Adjacent' #
ImageType = 'Color'
#For a Node
AlgorithmChoice = 'Clustering'
AlgParams = {'mr': 0.01, 'vr': 0.01, 'sr': 0.001, 'DIMS': [],'CENTS': [], 'node_id': [],'NumCentsPerLayer':NumCentsPerLayer}
DESTIN = Network(numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, PatchMode, ImageType)
print("Starting Training")
#
img = rng.rand(32, 32,3)
DESTIN.Layers[0][0].loadInput(img,4)
DESTIN.initLayer(0)
print DESTIN.Layers[0][0].Nodes[0][0].Belief
DESTIN.Layers[0][0].doLayerLearning(True)
print DESTIN.Layers[0][0].Nodes[0][0].Belief

DESTIN.Layers[0][1].loadInput(DESTIN.Layers[0][0].Nodes,2)

exit(0)
DESTIN.initLayer(0)
print DESTIN.Layers[0][0].Nodes[0][0].Belief
DESTIN.Layers[0][0].doLayerLearning(True)
print DESTIN.Layers[0][0].Nodes[0][0].Belief

#print DESTIN.Layers[0][0].Nodes[1][1].Input
#print type(DESTIN.Layers[0][1].Nodes[0][0])
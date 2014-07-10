__author__ = 'teddy'
from Network import *
import numpy.random as rng
from loadData import *
# *****Define Parameters for the Network and Nodes

#Network Params
numLayers = 4
NumNodesPerLayer = [[8, 8], [4, 4], [2, 2], [1, 1]]
NumCentsPerLayer = [25, 25, 25, 25]
PatchMode = 'Adjacent'  #
ImageType = 'Color'
NetworkMode = True # training is set true
#For a Node: specify Your Algorithm Choice and Coresponding parameters
AlgorithmChoice = 'Clustering'
AlgParams = {'mr': 0.01, 'vr': 0.01, 'sr': 0.001, 'DIMS': [], 'CENTS': [], 'node_id': [],
             'NumCentsPerLayer': NumCentsPerLayer}
#Declare a Network Object

DESTIN = Network(numLayers, AlgorithmChoice, AlgParams, NumNodesPerLayer, PatchMode, ImageType)
DESTIN.setMode(NetworkMode)
DESTIN.setLowestLayer(2)
#Load Data
[data, labels] = loadCifar(1) # loads cifar_data_batch_1

for L in range(DESTIN.NumberOfLayers):
    DESTIN.initLayer(L)
#data.shape[0]
for I in range(2):# For Every image in the data set
    if I%1 == 0:
        print("Iteration Number %d" % I)
    for L in range(DESTIN.NumberOfLayers):
        if L == 0:
            img = data[I][:].reshape(32,32,3)
            DESTIN.Layers[0][L].loadInput(img,4)# loadInput to Layer[0]
        else:
            DESTIN.Layers[0][L].loadInput(DESTIN.Layers[0][L-1].Nodes,2)
        DESTIN.Layers[0][L].doLayerLearning(NetworkMode)
    DESTIN.updateBeliefExporter()
DESTIN.dumpBelief()
#print np.asarray(DESTIN.NetworkBelief['Belief'][339]).shape
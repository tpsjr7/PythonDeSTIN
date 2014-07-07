from Layer import *
# TODO: get ridoff the sequential requirements like first feed the layer an input the you can initialize it

class Network():
    NumberOfLayers = []
    AlgorithmChoice = []
    AlgorithmParams = {}
    NumberOfNodesPerLayer = []
    PatchMode = 'Adjacent'
    Layers = []
    ImageType = 'Color'
    OperatingMode = 'Training'

    def __init__(self, numLayers, AlgChoice, AlgParams, NumNodesPerLayer, PatchMode='Adjacent', ImageType='Color'):
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

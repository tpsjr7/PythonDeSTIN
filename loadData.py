# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import scipy.io as io
# Contains loading cifar batches and
# feeding input to lower layer nodes
def loadCifar(batchNum):
	# For training_batches specify numbers 1 to 5
	# for the test set pass 6
	if batchNum <= 5:
		FileName = 'Cifar/data_batch_' + str(batchNum)
		FID = open(FileName, 'rb')
		dict = cPickle.load(FID)
		FID.close()
		return dict['data'],dict['labels']
	elif batchNum==6:
		FileName = 'Cifar/test_batch'
		FID = open(FileName, 'rb')
		dict = cPickle.load(FID)
		FID.close()
		return dict['data'],dict['labels']
	else:# here we will get the whole 50,000x3072 dataset
		I = 0
		FileName = 'Cifar/data_batch_' + str(I+1)
		FID = open(FileName, 'rb')
		dict = cPickle.load(FID)
		FID.close()
		data = dict['data']
		labels = dict['labels']
		for I in range(1,5):
			FileName = 'Cifar/data_batch_' + str(I+1)
			FID = open(FileName, 'rb')
			dict = cPickle.load(FID)
			FID.close()
			data = np.concatenate((data,dict['data']), axis=0)
			labels = np.concatenate((labels,dict['labels']), axis=0)
		return data,labels
def returnNodeInput(Input,Position,Ratio,Mode,ImageType):
	if Mode == 'Adjacent': #Non overlapping or Adjacent Patches
		PatchWidth = Ratio
		PatchHeight = Ratio
		if ImageType == 'Color':
			PatchDepth = 3
		else:
			PatchDepth = 1
		Patch = Input[Position[0]:Position[0]+PatchWidth,Position[1]:Position[1]+PatchHeight].reshape(1,(Ratio**2)*PatchDepth)
	else:# TODO Overlapping Patchecould be fed to a node
		print('Overlapping Patches Are Not Implemented Yet')
		patch = np.array([])
	return Patch

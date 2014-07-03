# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import scipy.io as io

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
	else:
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

__author__ = 'teddy'
from theano import function
import theano.tensor as T
import numpy as np
# Implements basic operatioons needed by Learning Algorithms
# Matrix-Matrix Addition, Division, Subtraction, Multiplication
def theanoMatMatAdd(In1,In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    AddMat = function([var1,var2],[var1+var2])
    return AddMat(In1,In2)
def theanoMatMatDiv(In1,In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    DivVec = function([var1,var2],[var1/var2])
    return DivVec(In1,In2)
def theanoMatMatSub(In1,In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    SubMat = function([var1,var2],[var1-var2])
    return SubMat(In1,In2)
def theanoMatMatMul(In1,In2,option):
    if option == 'E':
        var1 = T.dmatrix('var1')
        var2 = T.dmatrix('var2')
        MulMat = function([var1,var2],[var1*var2])
    else:
        var1 = T.dmatrix('var1')
        var2 = T.dmatrix('var2')
        MulMat = function([var1,var2], [np.dot(var1,var2)])
    return MulMat(In1,In2)
def theanoMatSum(In1,axs):
    var1 = T.dmatrix('var1')
    MatSum = function([var1],[np.sum(var1,axis=axs)])
    return MatSum(In1)
def theanoVecVecAdd(In1,In2):
    var1 = T.dvector('var1')
    var2 = T.dvector('var2')
    AddVec = function([var1,var2],[var1+var2])
    return AddVec(In1,In2)
def theanoVecVecDiv(In1,In2):
    var1 = T.dvector('var1')
    var2 = T.dvector('var2')
    DivVec = function([var1,var2],[var1/var2])
    return DivVec(In1,In2)
def theanoMatScaDiv(In1,In2):
    var1 = T.dmatrix('var1')
    var2 = T.dscalar('var2')
    Div = function([var1,var2],[var1/var2])
    return Div(In1,In2)
def theanoMatVecDiv(In1,In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    AddVec = function([var1,var2],[var1+var2])
    return AddVec(In1,In2)

'''

def theanoMatVecDiv(In1,In2):
    var1 = T.dmatrix('var1')
    var2 = T.dmatrix('var2')
    AddVec = function([var1,var2],[var1+var2])
    return AddVec(In1,In2)

def theanoVecSum(In1):
    var1 = T.dmatrix('var1')
    MatSum = function([var1],np.sum(var1))
    return MatSum(In1)
def theanoVecScaDiv(In1,In2):
    var1 = T.vector('var1')
    var2 = T.scalar('var2')
    VecScaDiv = function([var1,var2],[var1/var2])
    return VecScaDiv(In1,In2)
def theanoMatScaDiv(In1,In2):
    var1 = T.matrix('var1')
    var2 = T.scalar('var2')
    MatScaDiv = function([var1,var2],[var1/var2])
    return MatScaDiv(In1,In2)
'''
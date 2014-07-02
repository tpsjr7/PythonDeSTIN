This is a repo for the development of Python DeSTIN (PyDeSTIN).
As a starting point: 
	-> here I have developed a Node and Learning Algorithm Classes
		- Just to get a sense og Learning Algorithm inside a node.
		  I put a LogisticRegression implemented in theano 
	-> LearningAlgorithm Class is placed as attribute of the Node class.
	-> LearningAlgorithm Object does the actual learning for the node.
The whole PyDeSTIN will have Four Classes: LearningAlgorithm, Node, Layer and Network
The Classes will be placed in a nested fashion as follows:
-> Network
     Layer
	Node
	  LearningAlgorithm


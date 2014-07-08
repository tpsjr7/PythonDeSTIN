
PythonDeSTIN is a repo for the development of Python DeSTIN (PyDeSTIN).
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


TODO List:
- Learning Algorithms in Theano:
	- Online Non-Negative Sparse AutoEncoder
	- Stable Incremental Clustering 
- Data Preprocessing i.e ZCA Whitening and Normalization
- A Network Class

Taking into consideration points listed here: http://wiki.opencog.org/w/New_DeSTIN_Redesign_Proposal	
We will have explicit branches for A to D.

A) pure DeSTIN Framework: flexible enough to support different learning algorithms
		(Done)
B) Implemeting Online-NonNegative Sparse AutoEncoder in theano
		(Started)
C) Implemeting Stable Incremental K Means Clustering in theano

D) a LeNet style CNN built using the general-purpose CNN layer 
	(The theory may require revision)
	(How to make sense of the Complex and Simple cell like filters simulated in 	the CNN?)
	(Pooling is also an issue.....)

D) hybrid DeSTIN-CNN without feedback 


D) hybrid DeSTIN-CNN with feedback



Reading List For DeSTIN:

* http://web.eecs.utk.edu/~itamar/Papers/BICA2009.pdf
* http://www.ece.utk.edu/~itamar/Papers/BICA2011T.pdf
* http://web.eecs.utk.edu/~itamar/Papers/AI_MAG_2011.pdf
* http://research.microsoft.com/en-us/um/people/dongyu/nips2009/papers/Arel-DeSTIN_NIPS%20tpk2.pdf
* http://goertzel.org/Goertzel_AAAI11.pdf
* http://goertzel.org/DeSTIN_OpenCog_paper_v1.pdf
* http://goertzel.org/papers/DeSTIN_OpenCog_paper_v2.pdf
* http://www.springerlink.com/content/264p486742666751/fulltext.pdf
* http://goertzel.org/VisualAttention_AGI_11.pdf
* http://goertzel.org/Uniform_DeSTIN_paper.pdf
* http://goertzel.org/papers/Uniform_DeSTIN_paper_v2.pdf
* http://goertzel.org/papers/CogPrime_Overview_Paper.pdf

* Visit Dr. Itamar Arel's Machine Intelligence Lab at University of Tennessee at Knoxville. http://mil.engr.utk.edu/

import numpy as np


def sigmoid(x):
 	return 1/(1+np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x)*(1-sigmoid(x))
	

	
class Layer:
	# consctuctor
	def __init__(self, nPrev,nNode):
		#self.node = np.zeros((batchSize,nNode)) # value of each node
		self.w = np.random.randn(nNode,nPrev)# init = random
		#self.w =  np.ones((nNode,nPrev))# init = 0
		#self.w = self.w*0.5
		#self.bias = np.random.uniform(-1,1) # A single value

	def printf (self):
		print ("layer information: ")
		print ("this layer has : " + str(self.nNode) + " nodes. Prev layer has: " + str (self.nPrev) +" nodes")
		print (self.w.shape)

	def getNextLayer(self,_input,activate):
		self.node =  _input 
		returnValue = self.w.dot(self.node)
		return sigmoid(returnValue)

	def setWeight(self,_w):
		self.w = _w

	def backPropagation (self,error,learningRate,deactivate):
        # dError/dw = 2* errorFromNextLayer * sigmoid (NodeNextLayer) * currentNode
		error = 2*sigmoidPrime(self.w.dot(self.node)) * error
		self.delta = (error.dot(self.node.T))/self.node.shape[1]
		self.w -= learningRate * self.delta
		error = self.w.T.dot(error)

		return error

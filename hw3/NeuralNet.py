from Layer import Layer
import numpy as np

def softmax(x):
	returnValue = np.exp(x - np.max(x, axis=0))
	return returnValue/np.sum(returnValue, axis=0)

class NeuralNet:
	# consctuctor

	def __init__(self, structure,batchSize,learningRate):
		self.layer =[]
		self.nLayer = len(structure)-1
		self.structure = structure
		self.inputSize = self.structure[0]
		
		for i in range (1,self.nLayer+1,1 ):
			newLayer = Layer(structure[i-1],structure[i])
			self.layer.append(newLayer)

		self.learningRate = learningRate
		self.batchSize = int (batchSize)

	def printf (self):
		print ("network structure : " + str (self.structure))
		print ("network information: ")
		for i in range (0,self.nLayer):
			print ("Layer " + str(i+1) + ": ")
			self.layer[i].printf()
		print ("Input " + str (self.layer[0].node))
		print ("Output " + str (self.layer[self.nLayer-1].node))

	def printWeight (self):
		for l in self.layer:
			print (l.w)
			print ("")

	def feedFoward (self,_input):
		sizeData  = _input.shape[0]
		_input = _input.T
		for i in range ( self.nLayer ):
			_input  = self.layer[i].getNextLayer(_input,i-self.nLayer+1)

		return _input
	
	def training (self,error):
		for i in range( self.nLayer-1,-1,-1):
			error = self.layer[i].backPropagation(error,self.learningRate,self.nLayer-1-i)

	#NEED TO MERGE TO BE ONE FUNCTION LATER
	def feedFowardClassification (self,_input):
		sizeData  = _input.shape[0]
		_input = _input.T
		for i in range ( self.nLayer-1 ):
			_input  = self.layer[i].getNextLayer(_input,i-self.nLayer+1)
		
		self.layer[self.nLayer-1].node = _input
		_input = self.layer[self.nLayer-1].w.dot(self.layer[self.nLayer-1].node)
		_input = softmax(_input)
		return _input
	
	def trainingClassification (self,error):
		
		self.layer[self.nLayer-1].delta = (error.dot(self.layer[self.nLayer-1].node.T))/self.layer[self.nLayer-1].node.shape[1]
		error = self.layer[self.nLayer-1].w.T.dot(error)
		self.layer[self.nLayer-1].w -= self.learningRate * self.layer[self.nLayer-1].delta
		
		
		for i in range( self.nLayer-2,-1,-1):
			error = self.layer[i].backPropagation(error,self.learningRate,self.nLayer-1-i)
			
			
			
			
			


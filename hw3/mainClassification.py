import math
from Layer import Layer
from NeuralNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys 
from mpl_toolkits.mplot3d import Axes3D 

def cost_function(val, target):
	return math.sqrt(np.mean((np.square(target - val))))

def convertToOneHotVector(x) :
	loockup = np.array(([0,0,0,0,0],[0,0,0,0,1], [0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]), dtype=float) 
	return loockup[int(x)]

def convertBinary(x) :
	loockup = np.array(([0,1],[1,0]), dtype=int) 
	return loockup[int(x)]

def getDiff(A,B) :
	tempResult = A - B
	tempResult = abs(tempResult)
	mergeCol = tempResult[:,0]+tempResult[:,1]
	return list(mergeCol).count(2)*1.0/ A.shape[0]

#load data 
data = [] 
nSize =0 
nFeature = 34

with open('ionosphere_data.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	temp = 0
	for row in csv_reader:
		for j in range (nFeature) :
			data.append(float(row[j]))
		if (row[nFeature]=='g') :
			data.append(1)
		else :
			data.append(0)
		nSize +=1


CollectedData = np.reshape(data,(nSize,nFeature+1))
#RANDOM
np.random.shuffle(CollectedData)

nTraingSize = int(0.75 *CollectedData.shape[0])
nTestingSize = int(0.25 *CollectedData.shape[0])
trainData =   CollectedData[:nTraingSize]
#CONFIGURATION
learningRate = 0.0001
batchSize =  16
NN = NeuralNet([nFeature,128,64,32,2],batchSize,learningRate)
nTime = nTraingSize/batchSize
epoch = 10000
print (nTime)
ECE = []
CEE = []
validationError = []
meanH1Good =[]
meanH1Bad =[]
meanH1Good999 =[]
meanH1Bad999 =[]
for i in range (epoch) :
	currentEpochError = 0
	cossEntropyError = 0
	np.random.shuffle(trainData)
	for t in range (nTime) :
		_input =  trainData[t*batchSize:(t+1)*(batchSize),0:nFeature]
		_rawOutput =  trainData[t*batchSize:(t+1)*(batchSize),nFeature]
		_output =[]
		for item in _rawOutput:
			_output.append(convertBinary(item))
			
		_output = np.array(_output)
		_output= _output.T
		
		predictValue = NN.feedFowardClassification(_input)

		predictValue = np.array(predictValue)
		_output = np.array(_output)

		error = ( predictValue - _output)
		NN.trainingClassification(error)

		currentEpochError += getDiff(_output.T, predictValue.T.round())
		#currentEpochError += zero_one_loss(_output.T, predictValue.T.round())
		_output = _output.T
		predictValue =predictValue.T
		for k in range (batchSize) :
			index =  np.argmax(_output[k],axis=0)
			cossEntropyError-=math.log (predictValue[k][index])

		
	# PLOTTING
	
	if (i==5) :
		_input =  CollectedData[0:nTraingSize,0:nFeature]
		_rawOutput = CollectedData[0:nTraingSize,nFeature]
		predictValue = NN.feedFowardClassification(_input)
		for k in range(nTraingSize) :
			if (_rawOutput[k]==1) :
				meanH1Good.append (  NN.layer[3].node[:,k]  )
			else :
				meanH1Bad.append ( NN.layer[3].node[:,k] )
	if (i==390) :
		_input =  CollectedData[0:nTraingSize,0:nFeature]
		_rawOutput = CollectedData[0:nTraingSize,nFeature]
		predictValue = NN.feedFowardClassification(_input)
		for k in range(nTraingSize) :
			if (_rawOutput[k]==1) :
				meanH1Good999.append (NN.layer[3].node[:,k]  )
			else :
				meanH1Bad999.append ( NN.layer[3].node[:,k]  )
		
	ECE.append (round((currentEpochError/nTime)*100,2))
	CEE.append (cossEntropyError/nTime)
	
	if (i%100==0):
		print ( str(round((currentEpochError/nTime)*100,2)) +"%")
		#prepair data
		_input =  CollectedData[nTraingSize:nSize,0:nFeature]
		_rawOutput =  CollectedData[nTraingSize:nSize,nFeature]
		_output =[]
		for item in _rawOutput:
			_output.append(convertBinary(item))
		_output = np.array(_output)
		_output= _output.T		  
		#predict data
		predictData = NN.feedFowardClassification(_input)
		predictData = (np.array(predictData))
		predictData = predictData.round()

		predictData =predictData.T
		_output =_output.T

		loss = getDiff(_output, predictData)
		print (loss*100)

	validationError.append (loss*100)
	
plt.plot (validationError,"b-",label="Validation")
plt.plot (ECE,"g-",label="Training")
plt.title("Error Rate Validation")
plt.legend(loc='best')
plt.xlabel("#epoch case")
plt.ylabel("Error Rate")

meanH1Good999 = np.array(meanH1Good999)
meanH1Bad999 = np.array(meanH1Bad999)
meanH1Good = np.array(meanH1Good)
meanH1Bad = np.array(meanH1Bad)

plt.figure(figsize=(4,4)) 
plt.subplot(121)
plt.plot (meanH1Good[:,0],meanH1Good[:,1],"go",label="Good")
plt.plot (meanH1Bad[:,0],meanH1Bad[:,1],"ro",label="Bad")
plt.title("2D feature 5th epoch")
plt.subplot(122)
plt.plot (meanH1Good999[:,0],meanH1Good999[:,1],"go",label="Good")
plt.plot (meanH1Bad999[:,0],meanH1Bad999[:,1],"ro",label="Bad")
plt.title("2D feature 999th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
 
ax = plt.subplot(121, projection='3d')
ax.scatter(meanH1Good[:,0],meanH1Good[:,1],meanH1Good[:,2] ,marker='o', color="green", label="Good",alpha=1.0)
ax.scatter(meanH1Bad[:,0],meanH1Bad[:,1],meanH1Bad[:,2], marker='o', color="red",label="Bad",alpha=1.0)
plt.title("3D feature 5th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax = plt.subplot(122, projection='3d')
ax.scatter(meanH1Good999[:,0],meanH1Good999[:,1],meanH1Good999[:,2] ,marker='o', color="green",alpha=1.0, label="Good")
ax.scatter(meanH1Bad999[:,0],meanH1Bad999[:,1],meanH1Bad999[:,2], marker='o', color="red",alpha=1.0,label="Bad")
plt.title("3D feature 999th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

#prepair data
_input =  CollectedData[nTraingSize:nSize,0:nFeature]
_rawOutput =  CollectedData[nTraingSize:nSize,nFeature]
_output =[]
for item in _rawOutput:
	_output.append(convertBinary(item))
_output = np.array(_output)
_output= _output.T		  
#predict data
predictData = NN.feedFowardClassification(_input)
predictData = (np.array(predictData))
predictData = predictData.round()


predictData =predictData.T
_output =_output.T


print (predictData.shape)
print (_output.shape)
loss = getDiff(_output, predictData)
print ("Test Prediction")
print (str(loss*100) + "%")

# for i in range (nTestingSize) :
# 	target = 'b'
# 	predictVal = 'b'
# 	if (_output[i][0] ==1) :
# 		target = 'g'
# 	if (predictData[i][0] ==1) :
# 		predictVal = 'g'   
# 	print (str(predictVal) + " - " + str(target))

plt.figure(figsize=(4,4)) 
plt.subplot(121)
plt.plot (ECE,"b-")
plt.title("Error Rate Training Phase")
plt.xlabel("#epoch case")
plt.ylabel("Error Rate")

plt.subplot(122)
plt.plot (CEE,"b-")
plt.title("Cross Entropy Training Phase")
plt.xlabel("#epoch case")
plt.ylabel("Cross Entropy Error")
plt.show()

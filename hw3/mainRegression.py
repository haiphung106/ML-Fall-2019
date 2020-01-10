import math
from Layer import Layer
from NeuralNet import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
def cost_function(val, target):
	return math.sqrt(np.mean((np.square(target - val))))

def convertToOneHotVector(x) :
	loockup = np.array(([0,0,0,0,0],[0,0,0,0,1], [0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]), dtype=float) 
	return loockup[int(x)]

#load data 
data = [] 
nSize =0 
nFeature = 10

with open('energy_efficiency_data.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	temp = 0
	next(csv_file,None)
	for row in csv_reader:
		for j in range (nFeature) :
			data.append(float(row[j]))
		nSize=nSize+1
 
data = np.reshape(data,(nSize,nFeature))
np.random.shuffle(data)

# 0: RelativeCompactness	1: SurfaceArea	2: WallArea	3: RoofArea	4: OverallHeight	5: Orientation 6: GlazingArea	7: GlazingAreaDistribution	8: HeatingLoad	9: CoolingLoad
#configuration
nTraingSize = int(0.75 *data.shape[0])
nTestingSize = int(0.25 *data.shape[0])

rawOrientationGlazingArea = np.reshape (data[:,5],(nSize,1))
rawGlazingAreaDistribution = np.reshape (data[:,7],(nSize,1))

orientationGlazingArea = np.zeros((nSize,5))
glazingAreaDistribution = np.zeros((nSize,5))

for i in range (nSize) :
	orientationGlazingArea[i] = convertToOneHotVector (rawOrientationGlazingArea[i])
	glazingAreaDistribution[i] = convertToOneHotVector (rawGlazingAreaDistribution[i])


dataPart1 = np.reshape(data[:,0:5],(nSize,5))
dataPart2 = np.reshape(data[:,6],(nSize,1))
dataPart3 = np.reshape(data[:,8:10],(nSize,2))
 
#stactkData
CollectedData = np.column_stack((dataPart1,dataPart2,orientationGlazingArea,glazingAreaDistribution,dataPart3))

#NORMALIZE
CollectedData[:,0:6] = CollectedData[:,0:6]/CollectedData[:,0:6].max(axis=0) # normolize
maxHeatingLoad= CollectedData[:,16].max(axis=0)
maxCoolingLoad= CollectedData[:,17].max(axis=0)
CollectedData[:,16] = CollectedData[:,16]/maxHeatingLoad 
CollectedData[:,17] = CollectedData[:,17]/maxCoolingLoad
HeatingLoadth = 16
CoolingLoadth = 17
nFeature = 18
selectedFeuture = range(nFeature)
# CALCULATE VARIACE
variance = []
for i in range (6) :
	std = np.std(CollectedData[:,i])
	variance.append(std)



#SELECTION FEATURES
#remove feature
nRemoveFeature = 0
nFeature = len(selectedFeuture)-2 -nRemoveFeature
for i in range (nRemoveFeature) :
	removeFeature = variance.index(min(variance))
	variance.remove(min(variance))
	selectedFeuture.remove(removeFeature)
	print("remove Feature: " + str (removeFeature) +"th")

print("Total Used Feature(s): " + str(nFeature))
print (selectedFeuture)

HeatingLoadth = 16 - (18- nFeature-2)
CoolingLoadth = 17 - (18- nFeature-2)

CollectedData = CollectedData[:,selectedFeuture]

#CONFIGURATION
learningRate = 0.002
batchSize =  32
NN = NeuralNet([nFeature,10,10,1],batchSize,learningRate)

nTime = nTraingSize/batchSize
epoch = 10000
print (CollectedData.shape)
ERMS = []
for i in range (epoch) :
	currentEpochError = 0
	for t in range (nTime) :
		_input =  CollectedData[t*batchSize:(t+1)*(batchSize),	0:nFeature]
 		_output =  CollectedData[t*batchSize:(t+1)*(batchSize),HeatingLoadth]

 		_output = np.reshape (_output,(1,batchSize))
 		predictValue = NN.feedFoward(_input)

 		error = ( predictValue - _output)
		currentEpochError += np.sum((error) ** 2)

		NN.training(error)
 	ERMS.append ((math.sqrt(currentEpochError)))
 	if (i%1000 ==0) : 
 		print((math.sqrt(currentEpochError)))
 

_input =  CollectedData[0:nTraingSize,0:nFeature]
_output =  CollectedData[0:nTraingSize,HeatingLoadth]
predictData = NN.feedFoward(_input)

_output = np.reshape (_output,(nTraingSize,1))
predictData = np.reshape (predictData,(nTraingSize,1))

error = ( predictValue - _output)
traningError = np.sum((error) ** 2)
print ("Error Traning: " + str (math.sqrt((traningError/nTraingSize))))

plt.figure(figsize=(8,4)) 
plt.plot(_output,"b-", linewidth=1,label='target')
plt.plot(predictData,"r-" , linewidth=1,label='target')
plt.title("Predction for Training Data")
plt.xlabel("#th case")
plt.ylabel("HeatingLoad")
plt.show()

_input =  CollectedData[nTraingSize:nSize,0:nFeature]
_output =  CollectedData[nTraingSize:nSize,HeatingLoadth]
predictData = NN.feedFoward(_input)

_output = np.reshape (_output,(nTestingSize,1))
predictData = np.reshape (predictData,(nTestingSize,1))

error = ( predictValue - _output)
testingError = np.sum((error) ** 2)
print ("Error Testing: " + str (math.sqrt((testingError/nTestingSize))))


#for i in range (nTestingSize) :
	#print (str(_output[i]) + " - " + str(predictData[i]))

plt.figure(figsize=(8,4)) 
plt.plot(_output,"b-", linewidth=2,label='target')
plt.plot(predictData,"r-" , linewidth=2,label='target')
plt.title("Predction for Testing Data")
plt.xlabel("#th case")
plt.ylabel("HeatingLoad")
plt.show()

plt.plot(ERMS,"-")
plt.title("Training Curve")
plt.xlabel("# epoch")
plt.ylabel("ERMS")
plt.show()

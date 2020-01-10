import numpy as np
import matplotlib.pyplot as plt

bases = 3
lam = 0

with open("testfile.txt") as file:
	data = file.readlines()
	X = []
	Y = []
	for line in data:
		array = line.strip().split(',')
		for i in range(len(array)):
			array[i] = float(array[i])

		X.append(array[0])
		Y.append([array[1]])

X = np.array(X)
Y = np.array(Y)

design_matrix = np.zeros((len(X), bases)) # nxbases

for i in range(design_matrix.shape[0]):
	b = X[i]
	x = 1
	for j in range(design_matrix.shape[1]):
		design_matrix[i, j] = x
		x *= b

def transpose(A):
	row = A.shape[0]
	col = A.shape[1]
	B = np.zeros((col, row))
	for i in range(row):
		for j in range(col):
			B[j][i] = A[i][j]
	return B


def identitymatrix(n):
	I = np.zeros((int(n), int(n)))
	for i in range(n):
		I[i, i] = 1.0

	return I

def mul(A, B):
	result = np.zeros((A.shape[0], B.shape[1]))

	for i in range(A.shape[0]):
		for j in range(B.shape[1]):
			for k in range(A.shape[1]):
				result[i][j] += A[i][k] * B[k][j]

	return result

def inverse(X):
	print(X)
	n = len(X)
	print('n = {}'.format(n))
	L = [[0*0] * n for i in range(n)]
	U = [[0*0] * n for i in range(n)]
	for j in range(n):
		L[j][j] = 1
		for i in range(j+1):
			s1 = sum(U[k][j] * L[i][k] for k in range(i))
			U[i][j]= X[i][j] - s1

		for i in range(j, n):
			s2 = sum(U[k][j] * L[i][k] for k in range(j))
			L[i][j] = (X[i][j] - s2) / U[j][j]

	inverse_L = np.zeros((n, n))
	inverse_matrix = np.zeros((n, n))
	I = identitymatrix(n)

	for k in range(n):
		for i in range(n):
			s1 = sum(L[i][j] * inverse_L[j][k] for j in range(i))
			inverse_L[i][k] = (I[i][k] - s1) / L[i][i]

	for k in range(n):
		for i in reversed(range(n)):
			s2 = sum(U[i][j] * inverse_matrix[j][k] for j in reversed(range(i+1, n)))
			inverse_matrix[i][k] = (inverse_L[i][k] - s2) / U[i][i]

	return inverse_matrix


result = mul(mul(inverse(mul(transpose(design_matrix), design_matrix) + lam * identitymatrix(bases)), transpose(design_matrix)), Y)
# print(result)
print('LSE:')
print('Fitting line:', end='')
for i in reversed(range(len(result))):
	print('{}*X^{}'.format(result[i], i))

error = sum((mul(design_matrix, result) - Y)**2)
print('Total error:', end='')
print(error)

def polycoefficient(x, w):
	v = len(w)
	y = 0
	for i in range(v):
		y += w[i][0] * (x ** i)

	# print(y)
	return y


x = np.linspace(-6, 6,23)
plt.figure(figsize=(12, 4))
plt.scatter(X, Y, c='r')
plt.plot(x, polycoefficient(x, result), 'b-')
plt.title('Using LSE method')
plt.show()
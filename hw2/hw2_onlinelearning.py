# Created by haiphung106
import math

a = 10
b = 1

with open('testfile.txt') as file:
    data = file.read().split('\n')

# Prior
for rowth in range(len(data)):
    line = data[rowth]
    rowth += 1
    likelihood = 0
    total = 0
    x = 0
    Y = len(line)
    for i in range(Y):
        if line[i] == '1':
            x += 1
    likelihood = (math.factorial(Y) / (math.factorial(x) * math.factorial(Y - x)) * (
                ((x / Y) ** x) * (1 - (x/ Y)) ** (Y - x)))
    print('Case {}: {}'.format(rowth, line))
    print('Likelihood: {}'.format(likelihood))
    print('Beta prior: a = {} b = {}'.format(a, b))
    a += x
    b += (Y - x)
    print('Beta posterior: a = {} b = {}'.format(a, b))

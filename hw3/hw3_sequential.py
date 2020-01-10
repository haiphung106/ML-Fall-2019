# Created by haiphung106
from function import gaussian_data_generator

mean = 3.0
variance = 5.0
print('Data point source function: N({}, {})'.format(mean, variance))
n = 0
pre_mean = 0.0
pre_variance = 0.0
ml_mean = 0.0
ml_variance = 0.0

while(True):
# for n in range(1,100):
    n += 1
    data = float(gaussian_data_generator(mean, variance))
    print('Add data point: {}'.format(data))
    ml_mean = pre_mean + (data - pre_mean) / n
    ml_variance = ml_variance + (pre_mean ** 2) - (ml_mean ** 2) + (((data ** 2) - ml_variance - (pre_mean ** 2)) / n)
    print('Mean = {}'.format(ml_mean), ' Variance = {}'.format(ml_variance))
    if abs(pre_mean - ml_mean) < 0.00001 and abs(pre_variance - ml_variance) < 0.00001:
        break
    pre_mean = ml_mean
    pre_variance = pre_variance



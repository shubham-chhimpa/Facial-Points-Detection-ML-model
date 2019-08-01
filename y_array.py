import numpy as np

from numpy import genfromtxt
my_data = genfromtxt('y.txt', delimiter=',')

print(my_data[0][0])


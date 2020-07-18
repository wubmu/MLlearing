import numpy as np
import os
print(os.environ['PATH'])
data = np.genfromtxt('data.txt',delimiter=",",dtype=float)
print(type(data))
print(data)
# print(help())
vetor = np.array()
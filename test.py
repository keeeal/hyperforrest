
import time, random

import numpy as np
from panda3d.core import *

n = int(1e4)

def randPoint4():
    return Point4(*(2*random.random()-1 for i in range(4)))

def randVec4():
    return Vec4(*(2*random.random()-1 for i in range(4)))

points = [randPoint4() for i in range(n)]

x = randVec4()
y = randVec4()
z = randVec4()

start = time.time()
result = list([p.dot(x), p.dot(y), p.dot(z)] for p in points)
print(time.time() - start)

m = np.random.rand(4,3)

points = np.array(points)
start = time.time()
result = points.dot(m)
print(time.time() - start)

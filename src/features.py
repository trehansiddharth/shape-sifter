import util
import numpy as np
import math
import algorithms

def perimeter(contour):
    return np.sum(np.sqrt(util.norms(np.diff(contour, axis=0).T))) / (2*math.pi)

def curvature(contour, method, k=3):
    numPoints, numDimensions = contour.shape
    ks = np.zeros(numPoints)
    if method == 'circle':
        for j in range(0, numPoints):
            localPoints = util.sliceCircular(contour, j, j + k)
            x, r = algorithms.fitCircle('algebraic', 'exact', localPoints)
            ks[(j + (k - 1) / 2) % numPoints] = 1.0/r * np.sign(np.cross(\
                x.T - contour[(j + (k - 1) / 2) % numPoints],\
                contour[j] - contour[(j + k) % numPoints]))
    elif method == 'parabola':
        for j in range(0, numPoints):
            x = contour[j]
            p1 = contour[(j - 1) % numPoints]
            p2 = contour[(j + 1) % numPoints]
            z, u = algorithms.fitParabolaThreePoints(x, p1, p2)
            ks[j] = (1.0/z) * np.sign(np.cross(u, p1 - p2))
    return ks

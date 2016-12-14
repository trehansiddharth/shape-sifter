import numpy as np
import math
import itertools

def onehot(i, k):
    if i < k and i > 0:
        v = np.matrix(np.zeros([k, 1]))
        v[i, 0] = 1
        return v
    else:
        return None

def gradient_descent(gradient_function, initial):
    theta = initial
    return theta

def linear_regression(A, b):
    theta, residuals, rank, s = np.linalg.lstsq(A, b)
    return theta

def norms(points):
    (numDimensions, numPoints) = np.shape(points)
    return np.matrix([np.linalg.norm(points[:,i]) ** 2 for i in range(0, numPoints)])

def centerOfMass(points):
    (numDimensions, numPoints) = np.shape(points)
    return points * (1 / float(numPoints)) * np.matrix(np.ones((numPoints, 1)))

def spreadFrom(points, centerPoint):
    (numDimensions, numPoints) = np.shape(points)
    transformedPoints = points - centerPoint
    ns = norms(transformedPoints)
    return np.sum(ns) / float(numPoints)

def spread(points):
    return spreadFrom(points, centerOfMass(points))

def listOf(x):
    return x.flatten().tolist()[0]

def normalize(xs):
    s = sum(xs)
    return map(lambda x: float(x) / float(s), xs)

def rangeCircular(i, j, n):
    if (i % n) < (j % n):
        return range(i % n, j % n)
    else:
        return itertools.chain(range(i % n, n), range(0, j % n))

def sliceCircular(arr, i, j):
    n = len(arr)
    i %= n
    j = ((j - 1) % n) + 1
    if i < j:
        return arr[i:j]
    else:
        left = arr[:j]
        right = arr[i:]
        if np.empty(left):
            return right
        elif np.empty(right):
            return left
        else:
            return np.concatenate(right, left)

def rotate(xs, k):
    return xs[k%len(xs):] + xs[:k%len(xs)]

def contains(arr, x, y):
    (xs, ys) = arr.shape
    return x >= 0 and x < xs and y >= 0 and y < ys and arr[x, y]

def polar(arr, axis):
    return np.linalg.norm(arr), math.atan2(arr[1], arr[0])\
        - math.atan2(axis[1], axis[0])

def localMinima(arr):
    n = len(arr)
    compLeft = np.empty(n, np.bool)
    compRight = np.empty(n, np.bool)

    lastSeen = float("inf")
    for i in range(0, n):
        x = arr[i]
        compLeft[i] = x < lastSeen
        if x != arr[(i + 1) % n]:
            lastSeen = x
    compLeft[0] = arr[0] < lastSeen

    lastSeen = float("inf")
    for i in range(n - 1, -1, -1):
        x = arr[i]
        compRight[i] = x < lastSeen
        if x != arr[(i - 1) % n]:
            lastSeen = x
    compRight[n - 1] = arr[n - 1] < lastSeen

    return np.where(np.logical_and(compLeft, compRight))[0]

import numpy as np
import util
import algorithms
import uniontree
import features

class ContourTree:
    def __init__(self, contour, curvatures):
        self.__contour = contour
        self.__tree = uniontree.UnionTree(curvatures)
        self.__merges = []

    def contour(self):
        return self.__contour

    def curvatures(self):
        return features.curvature(np.array(self.__contour), 'parabola', k=3)

    def numPoints(self):
        return len(self.__contour)

    def roots(self):
        return self.__tree.roots()

    def children(self, node):
        return self.__tree.children(node)

    def stageMerge(self, curvature, indices):
        self.__merges.append((curvature, indices))

    def runStaged(self):
        top = self.__tree.top.copy()
        indicesToDelete = []
        for (curvature, indices) in self.__merges:
            self.__tree.merge(curvature, [top[index] for index in indices])
            indicesToDelete.extend(indices)
        for index in sorted(indicesToDelete, reverse=True):
            self.__contour.pop(index)


# Segments a contour into parts, based on criteria of the minima rule, shortcut
# rule, and convexity rule
# For now, it just uses a greedy algorithm
def segment(contourTree):
    # Get the current segmentation and curvatures
    contour = np.array(contourTree.contour())
    curvatures = contourTree.curvatures()
    n = contourTree.numPoints()

    # Find minima in curvature on the current contour
    curvatureMinima = util.localMinima(curvatures)

    # Find points at which curvature is negative
    curvatureNegatives = np.where(curvatures < 0)[0]

    # Merge the list of curvature minima and negative curvatures together
    segmentationPoints = np.union1d(curvatureMinima, curvatureNegatives)

    # Filter out consecutive points in the list of segmentation points
    segmentationPoints = segmentationPoints[np.where(np.logical_or(\
        (segmentationPoints - np.roll(segmentationPoints, 1) != 1),\
        (np.roll(segmentationPoints, -1) - segmentationPoints) != 1))]

    # Factor each point in the current segmentation into higher-order
    # segmentation
    for i in range(-1, len(segmentationPoints) - 1):
        # Get the indices of the endpoints in segment points
        ip1 = segmentationPoints[i]
        ip2 = segmentationPoints[i + 1]
        ipoints = util.rangeCircular(segmentationPoints[i] + 1,\
            segmentationPoints[i + 1], n)

        # Find the prabola that best fits the points between j and k
        p1 = np.array(contour[ip1])
        p2 = np.array(contour[ip2])
        points = np.array(contour[list(ipoints)])
        z, u = algorithms.fitParabola(p1, p2, points)

        # Stage a merge of all the points into a single point of curvature 1/z
        contourTree.stageMerge(1.0/z, list(ipoints))

    # Run all the merges
    contourTree.runStaged()

import numpy as np
import graph_tool
import cv2
import math
import util

# Converts a numpy array of a blob into a cloud of points, with graph edges
# indicating connectivity. The vertex properties "x" and "y" indicate the x and
# y coordinates of each point. dxy is a list of tuples indicate the relative
# positions of pixels that are considered neighbors to a point. Returns a graph.
def cloud(img, dxy):
    g = graph_tool.Graph(directed=False)
    xmap = g.new_vertex_property("double")
    ymap = g.new_vertex_property("double")
    for point in np.array(np.where(img > 1)).T:
        x = point[0]
        y = point[1]
        if ((x + y) % 500 == 0):
            print(x)
        v = g.add_vertex()
        xmap[v] = x
        ymap[v] = y
        neighbors = [(x + dx, y + dy)\
            for (dx, dy) in dxy\
            if dx > 0 or dy > 0\
            and util.contains(img, x + dx, y + dy)]
        for (xn, yn) in neighbors:
            vn = g.add_vertex()
            e = g.add_edge(v, vn)
    g.vertex_properties["x"] = xmap
    g.vertex_properties["y"] = ymap
    return g

# Takes a segmentation mask and returns an ordred list of points on the contour,
# in clockwise order around the shape
def findContour(img):
    # Ordered list of points on the contour
    contour = []

    # List of possible directions you can travel in along the contour, ordered
    # by priority
    dxy = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]

    # Set of states we've visited before so we don't loop forever
    visitedStates = set()

    # Choose the most upper left point as the starting point
    start = np.argmax(img)
    x0 = math.floor(start / img.shape[1])
    y0 = start - x0 * img.shape[1]

    # Iteratively find all the contour points
    x = x0
    y = y0
    while True:
        #print(dxy)
        # Add the current point to the ordered contour list
        contour.append(np.array([x, y]))

        # Get a list of possible neighbor points, based on if in the image and
        # if we've visited those points from the current point before
        neighbors = [(dx, dy) for (dx, dy) in dxy\
            if util.contains(img, x + dx, y + dy)\
            and (x, y, dx, dy) not in visitedStates]

        # If there is such a point, go to it, and add it to visitedStates
        # Otherwise, there are no more points on this contour
        if len(neighbors) > 0:
            dx, dy = neighbors[0]
            visitedStates.add((x, y, dx, dy))
            x += dx
            y += dy
            dxy = util.rotate(dxy, dxy.index((dx, dy)) - 3)
        else:
            break

    # Ignore the last point added since we've seen it before
    return np.array(contour[:-1])

# Erode an image. Modifies the image to erode it.
def erode(img, contour):
    for point in contour:
        x, y = point[0], point[1]
        if img[x, y] == 255:
            img[x, y] = 0
        elif img[x, y] == 0:
            img[x, y] = 128
    img[np.where(img == 128)] = 255

# Returns a map between an original contour pixel and the corresponding eroded
# contour pixel
def correspondences(originalContour, erodedContour):
    pass

# Skeletonize a cloud of points.
def skeletonize(img):
    pass

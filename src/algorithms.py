import util
import numpy as np
import math
import scipy.optimize

def align(features1, features2, metric, interpolation):
    pass

def fitParabolaThreePoints(x, p1, p2):
    # Calculate distance and angle of both points
    d1, theta1 = util.polar(p1 - x, np.array([0, 1]))
    d2, theta2 = util.polar(p2 - x, np.array([0, 1]))

    # Define the fitting function for Newton-Raphson
    zmax = lambda d, theta, phi: d * math.sin(phi - theta) * \
        math.tan(phi - theta)
    zmaxprime = lambda d, theta, phi: d * (math.sin(phi - theta) + \
        math.tan(phi - theta) / math.cos(phi - theta))
    f = lambda phi: zmax(d1, theta1, phi) - zmax(d2, theta2, phi)
    fprime = lambda phi: zmaxprime(d1, theta1, phi) - zmaxprime(d2, theta2, phi)

    # Calculate value of phi that fits all three points using Newton-Raphson
    phi = scipy.optimize.newton(f, (theta1 + theta2) / 2.0, fprime=fprime)

    # Calculate the focal length
    z = zmax(d1, theta1, phi)

    # Calculate the vector for the direction the parabola opens
    u = np.array([math.cos(phi), math.sin(phi)])

    # Always return a positive number for z
    u = np.sign(z) * u
    z = abs(z)

    # Return these parameters, understanding x is always the center point of
    # the parabola
    return z, u

def fitParabola(ep1, ep2, points):
    return 1.0, np.array([1.0, 1.0])

def fitCircle(algorithm, solve, points):
    # Record input dimensions
    points = points.T
    numDimensions, numPoints = points.shape
    if algorithm == 'algebraic':
        # Compute the 2-norm of each point in the input
        norms = util.norms(points)

        # Define the matrices A and b we're going to use for optimization
        A = np.hstack((norms.T, points.T))
        b = np.ones((numPoints, 1))

        if solve == 'exact':
            # Run linear regression
            theta = util.linear_regression(A, b)
        else:
            raise NotImplementedError

        # Determine the parameters of the algebraic equation of the circle
        a = np.float64(theta[0].item())
        b = theta[1:]
        c = -1

        # Determine the center and radius
        x = np.matrix(-b / (2.0 * a))
        r = math.sqrt(np.linalg.norm(x) ** 2 + 1 / a)
    else:
        # Transform the coordinates so that they are with respect to the center of mass
        (numDimensions, numPoints) = np.shape(points)
        center = util.centerOfMass(points)
        transformedPoints = points - center

        # Compute the norm of every point in the points matrix
        norms = util.norms(points)

        # Compute the matrices A and b to use in linear regression
        A = transformedPoints * transformedPoints.T
        b = 0.5 * transformedPoints * norms.T

        if solve == 'exact':
            # Run linear regression
            theta = util.linear_regression(A, b)
        else:
            raise NotImplementedError

        # Convert back to unshifted coordinate system and compute radius
        x = theta + center
        r = math.sqrt(np.linalg.norm(theta) ** 2 + np.sum(norms) / float(numPoints))

    points = points.T
    return x, r

import image
import model
import numpy as np
import cv2

def read(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def arrow():
    return read("../examples/arrow/images/train1.jpg")

def threshold(img):
    img[np.where(img < 128)] = 0
    img[np.where(img >= 128)] = 255
    return img

def contourImage(shape, contour, data, f):
    ximg = np.zeros((shape[0], shape[1], 3))
    dataMax = np.max(data)
    dataMin = np.min(data)
    for i in range(len(contour)):
        x = data[i]
        if x < 0:
            ximg[contour[i][0], contour[i][1], 0] = f(-x) * 255.0 / f(-dataMin)
        if x > 0:
            ximg[contour[i][0], contour[i][1], 2] = f(x) * 255.0 / f(dataMax)
    return ximg

def save(img, filename):
    cv2.imwrite(filename, img)

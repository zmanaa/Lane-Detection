import Utils.constants
import numpy as np
import cv2

def make_coordinates(image, line_paras):
    slope, intercept = line_paras
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/ slope)
    x2 = int((y2 - intercept)/ slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        paras = np.polyfit((x1,x2), (y1,y2), 1)
        slope = paras[0]
        intercept = paras[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_avg    = np.average(left_fit, axis=0)
    right_fit_avg   = np.average(right_fit, axis=0)

    left_line       = make_coordinates(image, left_fit_avg)
    right_line      = make_coordinates(image, right_fit_avg)
    return np.array([left_line, right_line])

def canny(image):
    gray        = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bluredImg   = cv2.GaussianBlur(gray, (5,5), 0)
    cannyImg    = cv2.Canny(bluredImg, 50, 150)
    return cannyImg


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0),
            Utils.constants.LINEWIDTH)
    return line_image

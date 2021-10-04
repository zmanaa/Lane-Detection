import Utils.usedFunctions
import Utils.constants
import numpy as np
import cv2

img         = cv2.imread("Assets/testImage.jpg")
lane_image  = np.copy(img)
canny_image = Utils.usedFunctions.canny(lane_image)
cropped_img = Utils.usedFunctions.region_of_interest(canny_image)
lines       = cv2.HoughLinesP(cropped_img, Utils.constants.PIXELS,
            Utils.constants.PERC, Utils.constants.THRESHOLD, np.array([]),
            minLineLength=Utils.constants.LINEPIX, maxLineGap=Utils.constants.LINEGAPS)

avged_lines = Utils.usedFunctions.average_slope_intercept(lane_image, lines)
# This will produce detected lines on black image
line_image  = Utils.usedFunctions.display_lines(lane_image, avged_lines)
comb_image  = cv2.addWeighted(lane_image, 0.7, line_image, 1, 1)
# Showing the image
cv2.imshow("Result", comb_image)
cv2.waitKey(0)

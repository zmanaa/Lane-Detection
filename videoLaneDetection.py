import Utils.usedFunctions
import Utils.constants
import numpy as np
import cv2


vid = cv2.VideoCapture("Assets/testVideo.mp4")
while(vid.isOpened()):
    _, frame = vid.read()
    canny_image = Utils.usedFunctions.canny(frame)
    cropped_img = Utils.usedFunctions.region_of_interest(canny_image)
    lines       = cv2.HoughLinesP(cropped_img, Utils.constants.PIXELS,
                Utils.constants.PERC, Utils.constants.THRESHOLD, np.array([]),
                minLineLength=Utils.constants.LINEPIX,
                maxLineGap=Utils.constants.LINEGAPS)

    avged_lines = Utils.usedFunctions.average_slope_intercept(frame, lines)
    # This will produce detected lines on black image
    line_image  = Utils.usedFunctions.display_lines(frame, avged_lines)
    comb_image  = cv2.addWeighted(frame, 0.7, line_image, 1, 1)
    # Showing the ane_imageimage
    cv2.imshow("Result", comb_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow()

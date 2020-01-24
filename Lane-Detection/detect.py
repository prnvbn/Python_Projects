import cv2
import numpy as np
LINE_COLOUR = (200,200,50)

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
# Optimizing the line displayed in combo_image
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # Adding more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def canny(image):
    # ED1.  Converting image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel =5
    # ED2. Reduce image noise i.e. smoothen it (optional)
    # A 5x5 kernel is imposed on the image to return a new image
    blur = cv2.GaussianBlur(gray, (kernel,kernel), 0)

    # ED3. Apply Canny method to identify edges (traces the edges)
    # It will perform a derivative on the image and measure the change in brightness i.e the gradient
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Using Hough Transformation for line detection
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),LINE_COLOUR,10)
    return line_image

# Creates a mask for the region of interest and implements it
def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


#######################################################################################################
# Reading the image and making a copy
image = cv2.imread('test_image.jpg') #saves image as a 2d array/matrix
lane_image = np.copy(image)
# EDGE DETECTION
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
# LINE DETECTION
lines = cv2.HoughLinesP(cropped_image, 2,np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# Displaying the image at different stages
cv2.imshow("canny", canny_image)
cv2.waitKey(60)
cv2.imshow("result", combo_image)
cv2.waitKey(0)
#######################################################################################################

# Implenting above Algorithm on every frame of the video
# cap = cv2.VideoCapture("test2.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#
#     #EDGE DETECTION
#     canny_image = canny(frame)
#     cropped_canny = region_of_interest(canny_image)
#
#     #LINE DETECTION
#     lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines)
#     line_image = display_lines(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#
#     cv2.imshow("result", combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
cv2.destroyAllWindows()

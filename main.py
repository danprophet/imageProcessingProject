import preprocessing
import registration
import verification
import cv2
import numpy as np

# 0. -------- Load Image --------
work_image = cv2.imread('image_resources/sample.png', 0) # open as gray picture

# 1. -------- PreProcessing --------
result_sharp_work_image = preprocessing.my_imfilter(work_image, preprocessing.shapreningFilter)
cv2.imwrite('image_resources/result_filter/sharpenResult.png', result_sharp_work_image)
result_smooth_work_image = preprocessing.my_imfilter(work_image, preprocessing.smoothFilter)
cv2.imwrite('image_resources/result_filter/smoothResult.png', result_smooth_work_image)

# 2. -------- Registration - Geometric operators --------


# 3. -------- Verification --------


# cv2.imshow('Original', work_image)
# cv2.imshow('Result Sharpen', result_sharp_work_image)
# cv2.waitKey()
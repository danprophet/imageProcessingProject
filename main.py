import preprocessing
import registration
import verification
import cv2
import numpy as np
import time

# 0. -------- Load Image --------
work_image = cv2.imread('image_resources/sample.png', 0) # open as gray picture

# 1. -------- PreProcessing --------
# result_sharp_work_image = preprocessing.my_imfilter(work_image, preprocessing.shapreningFilter)
# cv2.imwrite('image_resources/result_filter/sharpenResult.png', result_sharp_work_image)
# result_smooth_work_image = preprocessing.my_imfilter(work_image, preprocessing.smoothFilter)
# cv2.imwrite('image_resources/result_filter/smoothResult.png', result_smooth_work_image)

# 2. -------- Registration - Geometric operators --------
# result_scale_2x = registration.implement_geometric_transformation(work_image, registration.scale_filter_2x)
# cv2.imwrite('image_resources/result_filter/scaled_image_2x_{}.png'.format(time.time()), result_scale_2x)
# result_scale_halfx = registration.implement_geometric_transformation(work_image, registration.scale_filter_halfx)
# cv2.imwrite('image_resources/result_filter/scaled_image_halfx_{}.png'.format(time.time()), result_scale_halfx)
# result_scale_shear = registration.implement_geometric_transformation(work_image, registration.scale_filter_shear)
# cv2.imwrite('image_resources/result_filter/scaled_image_shear_{}.png'.format(time.time()), result_scale_shear)


# 3. -------- Verification --------
# result_threshold = verification.get_thershold_to_image(work_image)  # extract to image file
# cv2.imwrite('image_resources/result_filter/scaled_image_thresh_{}.png'.format(time.time()), result_threshold)
result_target = verification.morphological_operators(verification.test_image,
                                                     verification.test_se,
                                                     [1,1])
cv2.imwrite('image_resources/result_filter/found_{}.png'.format(time.time()), result_target)


# cv2.imshow('Original', work_image)
# cv2.imshow('Result Sharpen', result_sharp_work_image)
# cv2.waitKey()
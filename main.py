import preprocessing
import registration
import verification
import cv2
import numpy as np
import time

# 0. -------- Load Image --------
work_image = cv2.imread('image_resources/sample.png', 0) # open as gray picture
cv2.imwrite('image_output/original_bw.png', work_image)

# 1. -------- PreProcessing --------
# result_sharp_work_image = preprocessing.my_imfilter(work_image, preprocessing.shapreningFilter)
# cv2.imwrite('image_output/PreProcessing/sharpenResult_{}.png'.format(time.time()), result_sharp_work_image)
# result_smooth_work_image = preprocessing.my_imfilter(work_image, preprocessing.smoothFilter)
# cv2.imwrite('image_output/PreProcessing/smoothResult_{}.png'.format(time.time()), result_smooth_work_image)

# 2. -------- Registration - Geometric operators --------
# for transformation in registration.transformations.keys():
#     transformatiown_result = registration.implement_geometric_transformation(work_image, registration.transformations[transformation], transformation)
#     cv2.imwrite('image_output/geometric_operators/transformation_{}_{}.png'.format(transformation, time.time()), transformatiown_result)

# 3. -------- Verification --------
# result_threshold = verification.get_thershold_to_image(work_image)  # extract to image file
# cv2.imwrite('image_resources/result_filter/scaled_image_thresh_{}.png'.format(time.time()), result_threshold)
# result_target = verification.morphological_operators(verification.test_image,
#                                                      verification.test_se,
#                                                      [1,1])
# result_target = verification.binary_image_to_255(result_target)  # set all 1 to 255
# cv2.imwrite('image_resources/result_filter/found_{}.png'.format(time.time()), result_target)


# cv2.imshow('Original', work_image)
# cv2.imshow('Result Sharpen', result_sharp_work_image)
# cv2.waitKey()
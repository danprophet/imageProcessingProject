import preprocessing
import registration
import verification
import cv2
import numpy as np
import time

# 0. -------- Load Image --------
work_image = cv2.imread('image_resources/sample.png', 0) # open as gray picture
# cv2.imwrite('image_output/original_bw.png', work_image)

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
# # 3.1 - find 1:
# # slice
# work_image_sliced = work_image[450:600, 75:300]
# cv2.imwrite('image_output/morpological_operators/work_image_sliced_{}.png'.format(time.time()), work_image_sliced)
#
# work_image_threshold = verification.get_thershold(work_image_sliced, thrshold=0.689)  # extract to image file
# work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)  # for file writing
# cv2.imwrite('image_output/morpological_operators/work_image_threshold_to_255_{}.png'.format(time.time()), work_image_threshold_to_255)
#
# # reverted_threshold = verification.opposite_threshold(result_threshold)
# work_hit = verification.morphological_operators("hit_and_miss",
#                                                 work_image_threshold,
#                                                 verification.structure_element_number_1,
#                                                 [10 ,5])
# work_hit_to_255 = verification.binary_image_to_255(work_hit)
# cv2.imwrite('image_output/morpological_operators/work_hit_to_255_{}.png'.format(time.time()), work_hit_to_255)
#
# work_miss = verification.morphological_operators("hit_and_miss",
#                                                  verification.opposite_threshold(work_image_threshold),
#                                                 verification.structure_element_number_general_revert,
#                                                  [10,5])
# work_miss_to_255 = verification.binary_image_to_255(work_miss)
# cv2.imwrite('image_output/morpological_operators/work_miss_to_255_{}.png'.format(time.time()), work_miss_to_255)
#
# work_hit_and_miss = verification.hit_and_miss(work_hit, work_miss)
#
# # save hit and miss result as image:
# work_hit_and_miss_to_255 = verification.binary_image_to_255(work_hit_and_miss)
# cv2.imwrite('image_output/morpological_operators/work_hit_and_miss_to_255_{}.png'.format(time.time()), work_hit_and_miss_to_255)
#
# # save image with target bordered:
# final_result_image = verification.print_target_hit_miss_result(work_hit_and_miss, work_image_sliced)
# cv2.imwrite('image_output/morpological_operators/final_result_image_{}.png'.format(time.time()), final_result_image)

####

# # 3.2 - find 2:
# # slice
# work_image_sliced = work_image[450:600, 75:300]
# cv2.imwrite('image_output/morpological_operators/work_image_sliced_{}.png'.format(time.time()), work_image_sliced)
#
# work_image_threshold = verification.get_thershold(work_image_sliced, thrshold=0.689)  # extract to image file
# work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)  # for file writing
# cv2.imwrite('image_output/morpological_operators/work_image_threshold_to_255_{}.png'.format(time.time()), work_image_threshold_to_255)
#
# # reverted_threshold = verification.opposite_threshold(result_threshold)
# work_hit = verification.morphological_operators("hit_and_miss",
#                                                 work_image_threshold,
#                                                 verification.structure_element_number_2,
#                                                 [10 ,5])
# work_hit_to_255 = verification.binary_image_to_255(work_hit)
# cv2.imwrite('image_output/morpological_operators/work_hit_to_255_{}.png'.format(time.time()), work_hit_to_255)
#
# work_miss = verification.morphological_operators("hit_and_miss",
#                                                 verification.opposite_threshold(work_image_threshold),
#                                                 verification.structure_element_number_general_revert,
#                                                  [10,5])
# work_miss_to_255 = verification.binary_image_to_255(work_miss)
# cv2.imwrite('image_output/morpological_operators/work_miss_to_255_{}.png'.format(time.time()), work_miss_to_255)
#
# work_hit_and_miss = verification.hit_and_miss(work_hit, work_miss)
#
# # save hit and miss result as image:
# work_hit_and_miss_to_255 = verification.binary_image_to_255(work_hit_and_miss)
# cv2.imwrite('image_output/morpological_operators/work_hit_and_miss_to_255_{}.png'.format(time.time()), work_hit_and_miss_to_255)
#
# # save image with target bordered:
# final_result_image = verification.print_target_hit_miss_result(work_hit_and_miss, work_image_sliced)
# cv2.imwrite('image_output/morpological_operators/final_result_image_{}.png'.format(time.time()), final_result_image)

#####

# # 3.3 - Dilation - test image
# test_dilation = verification.test_image_2
# dilation_test_to_255 = verification.binary_image_to_255(test_dilation)
# cv2.imwrite('image_output/morpological_operators/dilation_test_image_{}.png'.format(time.time()), dilation_test_to_255)
#
# dilation = verification.morphological_operators("dilation",
#                                                 verification.test_image_2,
#                                                 verification.test_se,
#                                                  [1, 1])
#
# dilation_test_to_255 = verification.binary_image_to_255(dilation)
# cv2.imwrite('image_output/morpological_operators/dilation_test_result_{}.png'.format(time.time()), dilation_test_to_255)


# 3.4 - Dilation - over a picture - make text in picture Bolder:


###
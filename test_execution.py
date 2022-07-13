import preprocessing
import registration
import verification
import cv2
import numpy as np
import time

work_image = cv2.imread('image_resources/sample.png', 0) # open as gray picture

# slice
work_image = work_image[450:600, 75:300]

work_image_threshold = verification.get_thershold(work_image, thrshold=0.8)  # extract to image file
print(work_image_threshold)
work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)
cv2.imwrite('image_output/morpological_operators/work_image_threshold_to_255_{}.png'.format(time.time()), work_image_threshold_to_255)

# reverted_threshold = verification.opposite_threshold(result_threshold)
work_hit = verification.morphological_operators(work_image_threshold,
                                                verification.structure_element_eifel,
                                                [10 ,5])
work_hit_to_255 = verification.binary_image_to_255(work_hit)
cv2.imwrite('image_output/morpological_operators/work_hit_to_255_{}.png'.format(time.time()), work_hit_to_255)

work_miss = verification.morphological_operators(verification.opposite_threshold(work_image_threshold),
                                                 verification.structure_element_number_general_revert,
                                                 [10,5])
work_miss_to_255 = verification.binary_image_to_255(work_miss)
cv2.imwrite('image_output/morpological_operators/work_miss_to_255_{}.png'.format(time.time()), work_miss_to_255)

work_hit_and_miss = verification.hit_and_miss(work_hit, work_miss)

# save hit and miss result as image:
work_hit_and_miss_to_255 = verification.binary_image_to_255(work_hit_and_miss)
cv2.imwrite('image_output/morpological_operators/work_hit_and_miss_to_255_{}.png'.format(time.time()), work_hit_and_miss_to_255)

# save image with target bordered:
final_result_image = verification.print_target_hit_miss_result(work_hit_and_miss, work_image)
cv2.imwrite('image_output/morpological_operators/final_result_image_{}.png'.format(time.time()), final_result_image)



#### test hit and miss:
# result_hit = verification.morphological_operators(verification.test_image,
#                                                      verification.test_se,
#                                                      [1,1])
#
# result_miss = verification.morphological_operators(verification.opposite_threshold(verification.test_image),
#                                                      verification.opposite_threshold(verification.test_se),
#                                                      [1,1])
#
# hit_and_miss_result = verification.hit_and_miss(result_hit, result_miss)

# result_target = verification.binary_image_to_255(result_target)  # set all 1 to 255
# cv2.imwrite('image_resources/result_filter/found_{}.png'.format(time.time()), result_target)

print("end")
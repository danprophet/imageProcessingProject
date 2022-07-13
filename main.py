import preprocessing  # PreProcessing module
import registration  # Registration module
import verification  # Verification module
import cv2  # open cv for file opening and writing
import time  # time module for file marking

### 0. -------- Load Image --------
work_image = cv2.imread('image_resources/sample.png', 0)  # open as gray picture
pattern_image = cv2.imread('image_resources/pattern.jpg', 0)  # open as gray picture
cv2.imwrite('image_output/original_bw.png', work_image)  # write bw image to file

### 1. -------- PreProcessing --------
result_sharp_work_image = preprocessing.my_imfilter(work_image, preprocessing.shapreningFilter)  # sharpen image
cv2.imwrite('image_output/PreProcessing/sharpenResult_{}.png'.format(time.time()), result_sharp_work_image)
result_smooth_work_image = preprocessing.my_imfilter(work_image, preprocessing.smoothFilter)  # smooth image
cv2.imwrite('image_output/PreProcessing/smoothResult_{}.png'.format(time.time()), result_smooth_work_image)

### 2. -------- Registration - Geometric operators --------
for transformation in registration.transformations.keys():
    transformatiown_result = registration.implement_geometric_transformation(work_image, registration.transformations[transformation], transformation)
    cv2.imwrite('image_output/geometric_operators/transformation_{}_{}.png'.format(transformation, time.time()), transformatiown_result)

### 3. -------- Verification --------
############ 3.1 - find number 1 in sign:
work_image_sliced = work_image[450:600, 75:300]  # slice image
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_image_sliced_{}.png'.format(time.time()), work_image_sliced)

work_image_threshold = verification.get_thershold(work_image_sliced, thrshold=0.689)  # extract to image file
work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)  # thresh to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_image_threshold_to_255_{}.png'.format(time.time()), work_image_threshold_to_255)  # to file

work_hit = verification.morphological_operators("hit_and_miss",
                                                work_image_threshold,
                                                verification.structure_element_number_1,
                                                [10, 5])  # search hit
work_hit_to_255 = verification.binary_image_to_255(work_hit)  # hit to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_hit_to_255_{}.png'.format(time.time()), work_hit_to_255)  # hit thresh to file

work_miss = verification.morphological_operators("hit_and_miss",
                                                 verification.opposite_threshold(work_image_threshold),
                                                verification.structure_element_number_general_revert,
                                                 [10, 5])  # search miss
work_miss_to_255 = verification.binary_image_to_255(work_miss)  # miss thresh to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_miss_to_255_{}.png'.format(time.time()), work_miss_to_255)  # to file

work_hit_and_miss = verification.hit_and_miss(work_hit, work_miss)  # AND operation between hit and miss

work_hit_and_miss_to_255 = verification.binary_image_to_255(work_hit_and_miss)  # hit and miss to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_hit_and_miss_to_255_{}.png'.format(time.time()), work_hit_and_miss_to_255)  # to file

final_result_image = verification.print_target_hit_miss_result(work_hit_and_miss, work_image_sliced)  # print target result on original image
cv2.imwrite('image_output/morpological_operators/hit_and_miss/final_result_image_{}.png'.format(time.time()), final_result_image)  # to file

############ 3.2 - find 2:
work_image_sliced = work_image[450:600, 75:300]  # slice
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_image_sliced_{}.png'.format(time.time()), work_image_sliced)  # to file

work_image_threshold = verification.get_thershold(work_image_sliced, thrshold=0.689)  # threshold
work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)  # threshold to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_image_threshold_to_255_{}.png'.format(time.time()), work_image_threshold_to_255)  # to file

work_hit = verification.morphological_operators("hit_and_miss",
                                                work_image_threshold,
                                                verification.structure_element_number_2,
                                                [10, 5])  # search hit
work_hit_to_255 = verification.binary_image_to_255(work_hit)  # hit to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_hit_to_255_{}.png'.format(time.time()), work_hit_to_255)  # to file

work_miss = verification.morphological_operators("hit_and_miss",
                                                verification.opposite_threshold(work_image_threshold),
                                                verification.structure_element_number_general_revert,
                                                 [10, 5])  # search miss
work_miss_to_255 = verification.binary_image_to_255(work_miss)  # miss to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_miss_to_255_{}.png'.format(time.time()), work_miss_to_255)  # to file

work_hit_and_miss = verification.hit_and_miss(work_hit, work_miss)  # AND operation between hit and miss

work_hit_and_miss_to_255 = verification.binary_image_to_255(work_hit_and_miss)  # hit and miss to 255
cv2.imwrite('image_output/morpological_operators/hit_and_miss/work_hit_and_miss_to_255_{}.png'.format(time.time()), work_hit_and_miss_to_255)  # to file

final_result_image = verification.print_target_hit_miss_result(work_hit_and_miss, work_image_sliced)  # print targets on image
cv2.imwrite('image_output/morpological_operators/hit_and_miss/final_result_image_{}.png'.format(time.time()), final_result_image)  # to file

############ 3.3 - Dilation - test image
test_dilation = verification.test_image_2  # test_dilation image
dilation_test_to_255 = verification.binary_image_to_255(test_dilation)  # to 255
cv2.imwrite('image_output/morpological_operators/Dilation/dilation_test_image_{}.png'.format(time.time()), dilation_test_to_255)  # to file

dilation = verification.morphological_operators("dilation",
                                                verification.test_image_2,
                                                verification.test_se,
                                                [1, 1])  # get resulted dilation

dilation_test_to_255 = verification.binary_image_to_255(dilation)  # to 255
cv2.imwrite('image_output/morpological_operators/Dilation/dilation_test_result_{}.png'.format(time.time()), dilation_test_to_255)  # to file

############ 3.4 - Dilation - over a picture - make text in picture Bolder:
work_image_threshold = verification.get_thershold(work_image, thrshold=0.755)  # threshold
work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)  # threshold to 255
cv2.imwrite('image_output/morpological_operators/Dilation/work_image_dilation_threshold_{}.png'.format(time.time()), work_image_threshold_to_255)  # to file

result_dilation_mask = verification.morphological_operators("dilation",
                                                            work_image_threshold,
                                                            verification.test_se,
                                                            [1, 1])  # dilation mask result
result_dilation_mask_to_255 = verification.binary_image_to_255(result_dilation_mask)  # dilation mask to 255
cv2.imwrite('image_output/morpological_operators/Dilation/result_dilation_mask{}.png'.format(time.time()), result_dilation_mask_to_255)  # to file

smeared_image = verification.dilation_implemented_on_image(work_image_threshold,
                                                           work_image,
                                                           verification.test_se,
                                                           [1, 1])  # smear text over image
cv2.imwrite('image_output/morpological_operators/Dilation/smeared_image_{}.png'.format(time.time()), smeared_image)  # to file

############ 3.5 - over a picture - print pattern on image:
resulted_image_with_pattern = verification.print_pattern_on_image(result_dilation_mask,
                                                                  work_image,
                                                                  pattern_image)  # print pattern inside text:
cv2.imwrite('image_output/morpological_operators/pattern_printing/with_pattern_{}.png'.format(time.time()), resulted_image_with_pattern)  # to file

work_image_threshold = verification.get_thershold(work_image, thrshold=0.6)  # different threshold mask
work_image_threshold_to_255 = verification.binary_image_to_255(work_image_threshold)  # threshold to 255
cv2.imwrite('image_output/morpological_operators/pattern_printing/work_image_threshold_{}.png'.format(time.time()), work_image_threshold_to_255)  # to file

resulted_image_with_pattern = verification.print_pattern_on_image(work_image_threshold,
                                                                  work_image,
                                                                  pattern_image)  # image with pattern result
cv2.imwrite('image_output/morpological_operators/pattern_printing/with_pattern_{}.png'.format(time.time()), resulted_image_with_pattern)  # to file
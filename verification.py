"""This module handles verification morphological operations"""
import numpy as np
import skimage


# Image threshold
def get_thershold_to_image(image, thrshold = 0.7):
    # blur the image to denoise
    blurred_image = skimage.filters.gaussian(image, sigma=1.0)
    binary_mask = blurred_image > thrshold
    binary_mask = binary_mask.astype(int)
    height,width = binary_mask.shape

    for i in range(height):
        for j in range(width):
            if binary_mask[i,j] >= 1:
                binary_mask[i,j] = 255


def get_thershold(image, thrshold=0.7):
    # blur the image to denoise
    blurred_image = skimage.filters.gaussian(image, sigma=1.0)
    binary_mask = blurred_image > thrshold
    binary_mask = binary_mask.astype(int)
    return binary_mask


def morphological_operators(binary_image, se, se_center=[0, 0]):
    cols, rows = binary_image.shape
    se_cols, se_rows = se.shape
    center_value = se[se_center[0], se_center[1]]
    result = np.zeros([rows, cols], dtype=np.bool)

    for i in range (rows):
        for j in range(cols):
            try:
                if binary_image[i, j] == center_value:  # found matching value
                    startPosRow = i - se_center[0]
                    startPosColumn = j - se_center[1]
                    # get relevant image piece for current scan
                    image_search_zone = binary_image[startPosRow:startPosRow+se_rows, startPosColumn:startPosColumn+se_cols]

                    noMatchFlag = False  # indicates if SE matches current scanned zone
                    for se_i in range(se_rows):
                        for se_j in range(se_cols):
                            if (image_search_zone[se_i,se_j] == 1 and se[se_i,se_j] != 1) or (image_search_zone[se_i,se_j] != 1 and se[se_i,se_j] == 1):
                                noMatchFlag = True
                                break

                    if noMatchFlag:
                        result[i, j] = 0  # no match
                    else:
                        result[i, j] = 1  # match

            except Exception as e:
                print("out of bounds")
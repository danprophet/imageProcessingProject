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
    """This function executes the morphological_operators according to the passed stracture element se.
    se_center = [x_pos, y_pos] coordinates of the se
    se = stracture element
    binary_image = passed binary image (thresholded)
    """
    cols, rows = binary_image.shape
    se_cols, se_rows = se.shape
    center_value = se[se_center[0], se_center[1]]
    result = np.zeros([rows+(2*se_rows), cols+(2*se_cols)], dtype=np.bool)  # result with pad

    for i in range(rows):
        for j in range(cols):
            se_forward_distance_row = ((se_rows-1) - se_center[0])
            se_forward_distance_col = ((se_cols-1) - se_center[1])

            # edge cases:
            # 1. 4 edges
            if (i-se_center[0] < 0) and (j-se_center[1] < 0):  # top left corner
                sliced_se = se[se_center[0]:, se_center[1]:]
                sliced_binary = binary_image[i:se_rows - se_center[0], j:se_cols - se_center[1]]

            elif (i+se_forward_distance_row >= rows) and (j+se_forward_distance_col >= cols):  # bottom right corner
                se_row_diff = i + se_forward_distance_row - (rows-1)  # row se pixels out of border
                se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                sliced_se = se[se_center[0]:se_rows - se_row_diff, se_center[1]:se_cols - se_col_diff]
                sliced_binary = binary_image[i:, j:]

            elif (i-se_center[0] < 0) and (j + se_forward_distance_col >= cols):  # top right corner
                se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                sliced_se = se[se_center[0]:, se_center[1]:se_cols - se_col_diff]
                sliced_binary = binary_image[i:se_rows - se_center[0], j:]

            elif (i+se_forward_distance_row >= rows) and (j-se_center[1] < 0):  # bottom left corner
                se_row_diff = i + se_forward_distance_row - (rows-1)  # row se pixels out of border
                sliced_se = se[se_center[0]:se_rows - se_row_diff, se_center[1]:]
                sliced_binary = binary_image[i:, j:se_cols - se_center[1]]

            # 2. just borders
            elif i-se_center[0] < 0:  # only rows upper part
                sliced_se = se[se_center[0]:, :]
                sliced_binary = binary_image[i:se_rows - se_center[0], j - se_center[1]:j + se_forward_distance_col+1]

            elif j-se_center[1] < 0:  # only cols left part
                pass
            elif i+se_forward_distance_row >= rows:  # only rows bottom part
                pass
            elif j + se_forward_distance_col >= cols:  # only rows bottom part
                pass
            else:  # middle matrix
                pass

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
"""This module handles verification morphological operations"""
import numpy as np
import skimage

# test patterns:
test_image = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])

test_se = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])

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

    return binary_mask

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
    rows, cols = binary_image.shape
    se_rows, se_cols = se.shape
    center_value = se[se_center[0], se_center[1]]
    # result = np.zeros([rows+(2*se_rows), cols+(2*se_cols)], dtype=np.bool)  # result with pad
    result = np.zeros([rows, cols], dtype=np.uint8)  # result with pad

    se_forward_distance_row = ((se_rows - 1) - se_center[0])
    se_forward_distance_col = ((se_cols - 1) - se_center[1])

    for i in range(rows):
        for j in range(cols):
            # edge cases:
            # 1. 4 edges
            if (i-se_center[0] < 0) and (j-se_center[1] < 0):  # top left corner v v
                se_col_diff = abs(j - se_center[1])
                se_row_diff = abs(i - se_center[0])
                sliced_se = se[se_row_diff:, se_row_diff:]
                sliced_binary = binary_image[:se_rows - se_row_diff, :se_cols - se_col_diff]

            elif (i+se_forward_distance_row >= rows) and (j+se_forward_distance_col >= cols):  # bottom right corner v v
                se_row_diff = i + se_forward_distance_row - (rows-1)  # row se pixels out of border
                se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                sliced_se = se[:se_rows - se_row_diff, :se_cols - se_col_diff]
                sliced_binary = binary_image[i - se_center[0]:, j - se_center[1]:]

            elif (i-se_center[0] < 0) and (j + se_forward_distance_col >= cols):  # top right corner v v
                se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                se_row_diff = abs(i-se_center[0])  # exact amount of out se pixels
                sliced_se = se[se_row_diff:, :se_cols - se_col_diff - 1]
                sliced_binary = binary_image[:se_rows - se_row_diff, j-se_center[1]:]

            elif (i+se_forward_distance_row >= rows) and (j-se_center[1] < 0):  # bottom left corner v v
                se_col_diff = abs(j-se_center[1])  # exact amount of out se pixels
                sliced_se = se[:se_rows - se_row_diff, se_col_diff:]
                sliced_binary = binary_image[i-se_center[0]:, :se_cols-se_center[1]]

            # 2. just borders
            elif i-se_center[0] < 0:  # only rows upper part
                sliced_se = se[se_center[0]:, :]
                sliced_binary = binary_image[i:se_rows - se_center[0], j - se_center[1]:j + se_forward_distance_col+1]

            elif j-se_center[1] < 0:  # only cols left part
                sliced_se = se[:, se_center[1]:]
                sliced_binary = binary_image[i - se_center[0]:i + se_forward_distance_row +1, j:se_cols - se_center[1]]

            elif i+se_forward_distance_row >= rows:  # only rows bottom part
                se_row_diff = i + se_forward_distance_row - (rows-1)  # row se pixels out of border
                sliced_se = se[se_center[0]:se_rows - se_row_diff, :]
                sliced_binary = binary_image[i:, j - se_center[1]:j + se_forward_distance_col+1]

            elif j + se_forward_distance_col >= cols:  # only cols right part
                se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                sliced_se = se[:, :se_cols - se_col_diff+1]
                sliced_binary = binary_image[i - se_center[0]:i + se_forward_distance_row +1, j:]

            else:  # middle zone in the matrix
                sliced_se = se[:, :]
                sliced_binary = binary_image[i - se_center[0]:i + se_forward_distance_row +1, j - se_center[1]:j + se_forward_distance_col+1]

            # run over the 2 matrix and check match:
            sliced_se_rows, sliced_se_cols = sliced_se.shape
            noMatchFlag = False  # indicates if SE matches current scanned zone
            for se_sliced_row in range(sliced_se_rows):
                for se_sliced_col in range(sliced_se_cols):
                    if sliced_binary[se_sliced_row, se_sliced_col] == 1 and sliced_se[se_sliced_row, se_sliced_col] != 1:
                        noMatchFlag = True
                        break

            if noMatchFlag:
                result[i, j] = 0  # no match
            else:
                result[i, j] = 1  # match

    return result

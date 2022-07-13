"""This module handles verification Morphological Operators"""
import numpy as np
import skimage

# test patterns to be used in the program:
test_image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

test_image_2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

test_se = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])


structure_element_number_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                       [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                       [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                                       [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       ])  # represent number '2'

structure_element_number_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                       ])  # represent vertical line, good for finding number '1'


structure_element_number_general_revert = np.array([
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # opposite threshold to isolate numbers


## Module Functions: ##
def get_thershold_to_image(image, thrshold = 0.7):
    """
    This function gets an image, and threshold value - if threshold value is not set, the default is 0.7
    The function executes threshold over given image and returns 0 to 255 values mask of the original image after threshold,
    good for to file writing.
    :param image: given image
    :param thrshold: threshold value
    :return: binary_mask
    """
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
    """
    This function gets an image, and threshold value - if threshold value is not set, the default is 0.7
    The function executes threshold over given image and returns binary mask of the original image after threshold
    :param image: given image
    :param thrshold: threshold value
    :return: binary_mask
    """
    blurred_image = skimage.filters.gaussian(image, sigma=1.0)
    binary_mask = blurred_image > thrshold
    binary_mask = binary_mask.astype(int)
    return binary_mask


def binary_image_to_255(image):
    """
    This function gets binary image and returns the image with 0 to 255 values
    :param image:
    :return: return_image - 0 to 255 values image
    """
    rows,cols = image.shape
    return_image = np.zeros(shape=(rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if image[row,col] > 0:
                return_image[row,col] = 255

    return return_image


def opposite_threshold(image):
    """
    This function gets threshold image and reverts it - 0 instead or 1 and vice versa
    :param image:
    :return: return_image - binary opposite of given image
    """
    rows, cols = image.shape
    return_image = np.zeros(shape=(rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if image[row,col] == 1:
                return_image[row, col] = 0
            elif image[row, col] == 0:
                return_image[row, col] = 1
    return return_image


def hit_and_miss(hit, miss):
    """
    This function executes AND operation with 2 hit and miss threshold images
    :param hit: binary hit map
    :param miss: binary miss map
    :return: hit_and_miss_result - binary hit AND miss after AND operation
    """
    row_hits, cols_hit = hit.shape
    row_miss, cols_miss = miss.shape

    if row_miss != row_hits or cols_hit != cols_miss:  # matrix does not match
        return False
    else:
        rows, cols = hit.shape
        hit_and_miss_result = np.zeros(shape=(rows, cols), dtype=np.uint8)

        for row in range(rows):
            for col in range(cols):
                if hit[row, col] == 1 and miss[row, col] == 1:
                    hit_and_miss_result[row, col] = 1

        return hit_and_miss_result


def print_target_hit_miss_result(hit_and_miss, image):
    """
    This function gets hit and miss map result and original image, and prints borders around targets
    Returns image copy with targets
    :param hit_and_miss: binary hit and miss map
    :param image: given image 0 to 255 values
    :return: image_copy - image with targets around targets
    """
    rows, cols = hit_and_miss.shape
    image_rows, image_cols = image.shape
    image_copy = image.copy()
    # hit_and_miss_result = np.zeros(shape=(image_rows, image_cols), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            if hit_and_miss[row,col] == 1:

                # print border around target in white
                try:  # ignore close to border exceptions
                    for i in range(5):
                        image_copy[row+i, col+5] = 255
                        image_copy[row-i, col-5] = 255
                        image_copy[row+5, col+i] = 255
                        image_copy[row-5, col-i] = 255
                except Exception as e:
                    pass

    return image_copy


def dilation_implemented_on_image(binary_threshold, image, se, se_center):
    """
    This function smears pixels as the structure element according to binary threshold over the image
    :param binary_threshold: threshold of give image.
    :param image: give image
    :param se: structure element
    :param se_center: structure element center coordinates
    :return:
    """
    image_row, image_col = image.shape
    se_row, se_col = se.shape

    image_copy = image.copy()

    for row in range(image_row):
        for col in range(image_col):
            current_value = image[row,col]  # save value for smearing

            if binary_threshold[row, col] == 1:  # smear here
                for current_se_row in range(se_row):
                    for current_se_col in range(se_col):
                        try:
                            # find i:
                            if current_se_row < se_center[0]:
                                current_i = row - se_center[0]
                            elif current_se_row > se_center[0]:
                                current_i = row + se_center[0]
                            else:  # in center
                                current_i = row
                            # find j:
                            if current_se_col < se_center[1]:
                                current_j = col - se_center[1]
                                pass
                            elif current_se_col > se_center[1]:
                                current_j = col + se_center[1]
                                pass
                            else:
                                current_j = col
                            image_copy[current_i, current_j] = current_value

                        except Exception as e:  # out of borders
                            pass

    return image_copy


def print_pattern_on_image(binary, image, pattern):
    """
    This function gets binary map, image 0 to 255, and pattern 0 to 255, and prints pattern over the image at positions
    where map is 1
    :param binary: binary map o or 1
    :param image: given image 0 to 255
    :param pattern: given pattern 0 to 255
    :return: image_copy - image with pattern
    """
    image_row, image_col = image.shape
    pattern_row, pattern_col = pattern.shape
    image_copy = image.copy()

    for row in range(image_row):
        for col in range(image_col):
            if binary[row, col] == 1:  # print here
                image_copy[row, col] = pattern[row % pattern_row, col % pattern_col]

    return image_copy


def morphological_operators(mode, binary_image, se, se_center=[0, 0]):
    """
    This function executes the morphological_operators according to the passed Structure Element se.
    :param mode: hit_and_miss or dilation
    :param binary_image: passed binary image (thresholded)
    :param se: Structure Element
    :param se_center: [x_pos, y_pos] coordinates of the Structure Element
    :return: result - binary map after execution of Structure Element
    """
    rows, cols = binary_image.shape
    se_rows, se_cols = se.shape
    result = np.zeros([rows, cols], dtype=np.uint8)  # result with pad

    se_forward_distance_row = ((se_rows - 1) - se_center[0])
    se_forward_distance_col = ((se_cols - 1) - se_center[1])

    for i in range(rows):
        for j in range(cols):
            if mode == 'hit_and_miss':
                # edge cases:
                # 1. 4 edges
                if (i-se_center[0] < 0) and (j-se_center[1] < 0):  # top left corner v v
                    se_col_diff = abs(j - se_center[1])
                    se_row_diff = abs(i - se_center[0])
                    sliced_se = se[se_row_diff:, se_col_diff:]
                    sliced_binary = binary_image[:se_rows - se_row_diff, :se_cols - se_col_diff]

                elif (i+se_forward_distance_row >= rows) and (j+se_forward_distance_col >= cols):  # bottom right corner v v
                    se_row_diff = i + se_forward_distance_row - (rows-1)  # row se pixels out of border
                    se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                    sliced_se = se[:se_rows - se_row_diff, :se_cols - se_col_diff]
                    sliced_binary = binary_image[i - se_center[0]:, j - se_center[1]:]

                elif (i-se_center[0] < 0) and (j + se_forward_distance_col >= cols):  # top right corner v v
                    se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                    se_row_diff = abs(i-se_center[0])  # exact amount of out se pixels
                    real_forward_aviable_pixels = se_cols - se_col_diff - 1  # amount of pixels until out of border
                    sliced_se = se[se_row_diff:, :real_forward_aviable_pixels if real_forward_aviable_pixels > 0 else 1]
                    sliced_binary = binary_image[:se_rows - se_row_diff, j-se_center[1]:]

                elif (i+se_forward_distance_row >= rows) and (j-se_center[1] < 0):  # bottom left corner v v
                    se_col_diff = abs(j-se_center[1])  # exact amount of out se pixels
                    # se_row_diff = abs(i-se_center[0])
                    se_row_diff = i + se_forward_distance_row - (rows - 1)  # row se pixels out of border
                    sliced_se = se[:se_rows - se_row_diff, se_col_diff:]
                    sliced_binary = binary_image[i-se_center[0]:, :se_cols - se_col_diff]

                # 2. just borders
                elif i-se_center[0] < 0:  # only rows upper part
                    se_row_diff = abs(i - se_center[0])  # exact amount of out se pixels
                    sliced_se = se[se_row_diff:, :]
                    sliced_binary = binary_image[:se_rows - se_row_diff, j - se_center[1]:j + se_forward_distance_col+1]

                elif j-se_center[1] < 0:  # only cols left part
                    se_col_diff = abs(j - se_center[1])
                    sliced_se = se[:, se_col_diff:]
                    sliced_binary = binary_image[i - se_center[0]:i + se_forward_distance_row + 1, :se_cols - se_col_diff]

                elif i+se_forward_distance_row >= rows:  # only rows bottom part
                    se_row_diff = i + se_forward_distance_row - (rows-1)  # row se pixels out of border
                    sliced_se = se[:se_rows - se_row_diff, :]
                    sliced_binary = binary_image[i - se_center[0]:, j - se_center[1]:j + se_forward_distance_col+1]

                elif j + se_forward_distance_col >= cols:  # only cols right part
                    se_col_diff = j + se_forward_distance_col - (cols-1)  # col se pixels out of border
                    sliced_se = se[:, :se_cols - se_col_diff]
                    sliced_binary = binary_image[i - se_center[0]:i + se_forward_distance_row +1, j-se_center[1]:]

                else:  # middle zone in the matrix
                    sliced_se = se[:, :]
                    sliced_binary = binary_image[i - se_center[0]:i + se_forward_distance_row +1, j - se_center[1]:j + se_forward_distance_col+1]

                # run over the 2 matrix and check match:
                sliced_se_rows, sliced_se_cols = sliced_se.shape
                matchFlag = True  # indicates if SE matches current scanned zone
                for se_sliced_row in range(sliced_se_rows):
                    for se_sliced_col in range(sliced_se_cols):
                        if sliced_se[se_sliced_row, se_sliced_col] == 1 and sliced_binary[se_sliced_row, se_sliced_col] != 1:
                            matchFlag = False
                    if not matchFlag:
                        break

                if matchFlag:
                        result[i, j] = 1  # match
                else:
                    result[i, j] = 0  # no match

            elif mode == 'dilation':
                # print in the result 1's as the SE:
                if binary_image[i, j] == 1:
                    for r in range(se_rows):
                        for c in range(se_cols):
                            if se[r, c] == 1:  # need to put 1
                                try:  # won't execute if out of bound
                                    # find i:
                                    if r < se_center[0]:
                                        current_i = i - se_center[0]

                                    elif r > se_center[0]:
                                        current_i = i + se_center[0]

                                    else:  # in center
                                        current_i = i
                                    # find j:
                                    if c < se_center[1]:
                                        current_j = j - se_center[1]
                                        pass
                                    elif c > se_center[1]:
                                        current_j = j + se_center[1]
                                        pass
                                    else:
                                        current_j = j

                                    result[current_i, current_j] = 1
                                except Exception as e:
                                    pass

    return result


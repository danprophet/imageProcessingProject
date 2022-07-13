"""This module handles Registration"""
import numpy as np

#################################
# Geometric Transformation dictionary:
transformations = {
    "transformation_identity": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ]
    ),

    "transformation_scale_filter_2x": np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
        ]
    ),

    "transformation_scale_filter_halfx": np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0,   1]
        ]
    ),

    "transformation_scale_horizontal_shear": np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
        ]
    ),

    "transformation_scale_vertical_shear": np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1]
        ]
    ),

    "transformation_translation": np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ]
    ),

    "transformation_rotation": np.array([
        [0.9, -0.7, 0],
        [0.7, 0.9, 0],
        [0, 0, 1]
    ]
    ),
}

#################################


def implement_geometric_transformation(image, transformation, transformation_name):
    """
    This function implements geometric transformation over a given image
    :param image: given image to execute over the geometric transformation
    :param transformation: given geometric transformation
    :param transformation_name: passed transformation according to transformations dictionary in this module
    :return: imageCanvas - image after geometric transformation
    """
    original_image_height, original_image_width = image.shape  # save original image width and height
    new_image_height, new_image_width = image.shape # size for the destination canvas

    # determine size:
    if transformation_name == "transformation_translation":
        new_image_height = int(original_image_height + transformation[0, 2])
        new_image_width = int(original_image_width + transformation[1, 2])
    elif transformation_name == "transformation_identity":
        new_image_height = original_image_height
        new_image_width = new_image_width
    elif transformation_name == "transformation_scale_filter_2x" or transformation_name == "transformation_scale_filter_halfx":
        new_image_height = int(original_image_height * transformation[0, 0])+1
        new_image_width = int(original_image_width * transformation[1, 1])+1
    elif transformation_name == "transformation_scale_horizontal_shear":
        new_image_height = original_image_height
        new_image_width = int(original_image_width+transformation[0, 1]*original_image_height) + 1
    elif transformation_name == "transformation_scale_vertical_shear":
        new_image_height = int(original_image_height+transformation[1, 0]*original_image_width) + 1
        new_image_width = original_image_width
    elif transformation_name == "transformation_rotation":
        new_image_height = 2*new_image_height
        new_image_width = 2*new_image_width
    else:
        new_image_height = 4*new_image_height
        new_image_width = 4*new_image_width

    imageCanvas = np.zeros([int(new_image_height), int(new_image_width)], dtype=np.uint8)  # new canvas

    # implement the filter:
    for i in range(original_image_height):
        for j in range(original_image_width):
            current_pixel_info = image[i, j]
            current_coordinates = np.array([i, j, 0])
            if transformation_name == 'transformation_translation':
                imageCanvas[int(i + transformation[0, 2]), int(j + transformation[1,2])] = current_pixel_info
            elif transformation_name == "transformation_rotation":
                result_multiplication = current_coordinates @ transformation # result 1x3 matrix
                if result_multiplication[0] < 0 or result_multiplication[1] < 0:  # out of bounds
                    continue
                else:
                    imageCanvas[int(result_multiplication[0]), int(result_multiplication[1])] = current_pixel_info
            else:
                result_multiplication = current_coordinates @ transformation # result 3x3 matrix
                imageCanvas[int(result_multiplication[0]), int(result_multiplication[1])] = current_pixel_info

    return imageCanvas


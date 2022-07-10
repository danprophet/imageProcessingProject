"""This module handles registration"""
import numpy as np

# Geometric Transformation:
scale_filter_2x = np.array([
    [2,0,0],
    [0,2,0],
    [0,0,1]
    ]
)

scale_filter_halfx = np.array([
    [0.5,0,0],
    [0,0.5,0],
    [0,0,1]
    ]
)

scale_filter_shear = np.array([
    [1,1,0],
    [0,1,0],
    [0,0,1]
    ]
)

def implement_geometric_transformation(image, filter):
    original_image_height, original_image_width = image.shape  # save original image width and height
    new_image_height, new_image_width = image.shape # size for the destination canvas

    # TODO: add here code to determine filter to set imageCanvas size
    new_image_height = 2*new_image_height
    new_image_width = 2*new_image_width

    imageCanvas = np.zeros([new_image_height, new_image_width], dtype=np.uint8)  # new canvas

    # implement the filter:
    for i in range(original_image_height):
        for j in range(original_image_width):
            current_pixel_info = image[i,j]
            current_coordinates = np.array([i, j, 0])
            result_multiplication = current_coordinates @ filter # result 3x3 matrix
            imageCanvas[int(result_multiplication[0]), int(result_multiplication[1])] = current_pixel_info

    return imageCanvas


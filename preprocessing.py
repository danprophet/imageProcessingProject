"""This module handles PreProcessing"""
import numpy as np

# ----- set filters: -----
# set sharpein Filter:
shapreningFilter = np.array([[-1, -1,-1],[-1,8,-1],[-1,-1,-1]]) # Laplacian, image is the function
# shapreningFilter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) # Laplacian, image is the function
# set smooth Filter:
smoothFilter = np.array([[1,1,1],[1,1,1],[1,1,1]]) # Average of the around 1/9 * sums


# image filtering function:
def my_imfilter (s, filter):
    """
    This function gets GRAYSCALE image s and filter, and returns image after filter implementation
    :param s: given image
    :param filter: given filter
    :return: resultImage or resultFilter, according to given filter
    """
    filterSizeRows, filterSizeCols = filter.shape
    imageSizeRows, imageSizeCols = s.shape

    paddingDiff=filterSizeRows-1
    resultFilter = np.zeros((imageSizeRows, imageSizeCols), dtype=np.int32)  # image to be returned at the end of process
    resultImage = np.zeros((imageSizeRows, imageSizeCols), dtype=np.int32)  # image to be returned at the end of process
    filterType = None  # will determine logic later
    filterDictionary = {'sharpening':[np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
                  np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])], 'smoothing':[np.array([[1,1,1],[1,1,1],[1,1,1]])]}

    # determing current filter, for logic to apply later
    for filterFamily in filterDictionary.keys():
        for currentFilter in filterDictionary[filterFamily]:
            if np.array_equal(currentFilter, filter, equal_nan=False):
                filterType = filterFamily
                break

    # Add padding
    imageWithPadding = np.zeros([imageSizeRows + paddingDiff, imageSizeCols + paddingDiff], dtype=np.uint8)
    for i in range(imageSizeRows):
        for j in range(imageSizeCols):
            imageWithPadding[i+int(paddingDiff/2), j+int(paddingDiff/2)] = s[i,j]

    # implement the filter:
    for i in range(imageSizeRows):
        for j in range(imageSizeCols):
            pixelCalculation = 0  # for filter calculation later
            # calculate each pixel in the image after filter
            for filterRow in range(filterSizeRows):
                for filterCol in range(filterSizeCols):
                    pixelCalculation += imageWithPadding[i+filterRow, j+filterCol] \
                                        * filter[filterRow, filterCol]

            if filterType == 'smoothing':
                resultFilter[i, j] = int(pixelCalculation / (filterSizeRows * filterSizeCols))
            elif filterType == 'sharpening':
                resultFilter[i, j] = pixelCalculation

    if filterType == 'smoothing':
        resultFilter=resultFilter.astype(dtype=np.uint8)
        return resultFilter  # end logic for smoothing filter, image is ready

    if filterType == 'sharpening':
        resultFilter = resultFilter * 100.0 / (np.amax(resultFilter)-np.amin(resultFilter)) # normalize
        resultFilter = resultFilter.astype(np.int32)
        resultImage = s + resultFilter # add filter to original image
        resultImage = abs(resultImage) # drop negative values
        resultImage = resultImage* 255.0 / np.amax(resultImage) # normalize back to 256 scale
        resultImage = resultImage.astype(np.uint8)
        return resultImage


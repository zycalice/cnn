import numpy as np


def add_padding(image, num=1):
    # get vertical zeros and concatenate
    vertical_zeros = np.zeros((len(image), num))
    padded_image = np.concatenate((vertical_zeros, image), axis=1)
    padded_image = np.concatenate((padded_image, vertical_zeros), axis=1)

    # get  horizontal zeros and concatenate
    horizontal_zeros = np.zeros((num, len(padded_image[0])))
    padded_image = np.concatenate((horizontal_zeros, padded_image), axis=0)
    padded_image = np.concatenate((padded_image, horizontal_zeros), axis=0)
    return padded_image


def convolve_greyscale(image, kernel):
    """
    convolve function
    :param image: (image_height, image_width)
    :param kernel:  (kernel_height, kernel_width)
    :return:
    """
    kernel_flip = np.fliplr(np.flipud(kernel))
    padded_image = add_padding(image)
    result = np.zeros(image.shape)

    for i in range(len(image)):
        for j in range(len(image[0])):
            result[i][j] = np.sum(kernel_flip * padded_image[i:i + 3][:, j:j + 3])

    return result


def convolve_rgb(image, kernel):
    """
    uses function convolve_greyscale to go through each depth
    :param image: (image_height, image_width, image_depth)
    :param kernel: (kernel_height, kernel_width)
    :return: the same shape with image
    """
    image = image.astype(np.float32)
    image_height, image_width, image_depth = image.shape
    for d in range(image_depth):
        image[..., d] = convolve_greyscale(image[..., d], kernel)
    return image


def max_pooling(image, kernel, stride):
    """
    function to perform max pooling on one filter of the image
    :param image: (image_height, image_width)
    :param kernel: (kernel_height, kernel_width)
    :param stride: (stride_height, stride_width)
    :return:
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride

    # start the pooling
    result = []
    for row_start in range(image_height - kernel_height + 1)[::stride_height]:
        row_result = []
        for col_start in range(image_width - kernel_width + 1)[::stride_width]:
            subarea = image[row_start:row_start + kernel_height][:, col_start:col_start + kernel_width]
            row_result.append(np.max(subarea))
        result.append(row_result)
    return np.array(result)


def average_pooling(image, kernel, stride):
    """
    function to perform average pooling on one filter of the image
    :param image: (image_height, image_width)
    :param kernel: (kernel_height, kernel_width)
    :param stride: (stride_height, stride_width)
    :return:
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride

    # start the pooling
    result = []
    for row_start in range(image_height - kernel_height + 1)[::stride_height]:
        row_result = []
        for col_start in range(image_width - kernel_width + 1)[::stride_width]:
            subarea = image[row_start:row_start + kernel_height][:, col_start:col_start + kernel_width]
            row_result.append(np.mean(subarea))
        result.append(row_result)
    return np.array(result)

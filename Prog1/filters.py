# CS510: DL and CV. Summer 2023. Prog 1 Part 1(i), (ii), and (iii). Cera Oh.
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Gaussian Filter 3x3
a = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
gf3x3 = np.dot(1/16, a)

# Gaussian Filter 5x5
b = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [
             7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
gf5x5 = np.dot(1/273, b)

# DoG gx filter
DoG_gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# DoG gy filter
DoG_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# Function to load image as grayscale array
def load_image_grayscale(file_path):
    image = cv2.imread(file_path, 0)
    return image


def show(image, title):  # Function to show image as grayscale
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.show()


# Function to preprocess images and initialize convolution matrix
def preprocess(image, dimension):

  # Initialize convoluted image with zeros
    im_row, im_col = image.shape
    conv = np.zeros((im_row, im_col))

  # Zero pad image
    if dimension == 3:
        padded = zero_pad(image, 1)
    elif dimension == 5:
        padded = zero_pad(image, 2)
    else:
        print("Dimension needs to be 3 (3x3) or 5 (5x5)")

    return conv, padded


def zero_pad(image, width):  # Function to pad images with zeros of given width
    padded = cv2.copyMakeBorder(
        image, width, width, width, width, cv2.BORDER_CONSTANT, value=0)
    return padded


def sobel(gx, gy):  # Function to apply the Sobel filter

    square_add = np.power(gx, 2) + np.power(gy, 2)
    result = np.sqrt(square_add)

    return result


def DoG(conv, padded, orientation):  # Function to apply the Derivative of Gaussian gx and gy filters
    r_count = 0
    for row in padded:
        end = len(row)
        if np.array_equal(row, padded[0]):  # zero padded row
            pass  # Do nothing
        else:
            stop = end-1
            col = 1
            if orientation == 'x':
                while col < stop:
                    x = np.array([[padded[r_count-1][col-1], padded[r_count-1][col], padded[r_count-1][col+1]],
                                  [row[col-1], row[col], row[col+1]],
                                  [padded[r_count+1][col-1], padded[r_count+1][col], padded[r_count+1][col+1]]])
                    value = np.sum(x * DoG_gx)
                    conv[r_count-1][col-1] = value
                    col = col + 1

            elif orientation == 'y':
                while col < stop:
                    x = np.array([[padded[r_count-1][col-1], padded[r_count-1][col], padded[r_count-1][col+1]],
                                  [row[col-1], row[col], row[col+1]],
                                  [padded[r_count+1][col-1], padded[r_count+1][col], padded[r_count+1][col+1]]])
                    value = np.sum(x * DoG_gy)
                    conv[r_count-1][col-1] = value
                    col = col + 1
            else:
                print("The orientation isn't correct. Please pass 'x' or 'y'.")
        r_count = r_count + 1

    return conv


def gaussian(conv, padded, dimension):  # Function to apply 3x3 or 5x5 Gaussian Filters
    r_count = 0

    for row in padded:
        end = len(row)

        if np.array_equal(row, padded[0]):  # zero padded row
            pass  # Do nothing
        else:
            if dimension == 3:
                stop = end-1
                col = 1
                while col < stop:
                    x = np.array([[padded[r_count-1][col-1], padded[r_count-1][col], padded[r_count-1][col+1]],
                                  [row[col-1], row[col], row[col+1]],
                                  [padded[r_count+1][col-1], padded[r_count+1][col], padded[r_count+1][col+1]]])
                    value = np.sum(x * gf3x3)
                    conv[r_count-1][col-1] = value
                    col = col + 1
            elif dimension == 5:
                stop = end-2
                col = 2
                while col < stop:
                    x = np.array([[padded[r_count-2][col-2], padded[r_count-2][col-1], padded[r_count-2][col], padded[r_count-2][col+1], padded[r_count-2][col+2]],
                                  [padded[r_count-1][col-2], padded[r_count-1][col-1], padded[r_count-1]
                                   [col], padded[r_count-1][col+1], padded[r_count-1][col+2]],
                                  [row[col-2], row[col-1], row[col],
                                   row[col+1], row[col+2]],
                                  [padded[r_count+1][col-2], padded[r_count+1][col-1], padded[r_count+1]
                                   [col], padded[r_count+1][col+1], padded[r_count+1][col+2]],
                                  [padded[r_count+2][col-2], padded[r_count+2][col-1], padded[r_count+2][col], padded[r_count+2][col+1], padded[r_count+2][col+2]]])
                    value = np.sum(x * gf5x5)

                    conv[r_count-2][col-2] = value
                    col = col + 1
            else:
                print("The dimension is incorrect. Please pass 3 for 3x3 and 5 for 5x5")
        r_count = r_count + 1

    return conv


def main():
    # Gaussian
    path1 = 'filter1_img.jpg'
    path2 = 'filter2_img.jpg'

    img1 = load_image_grayscale(path1)
    img2 = load_image_grayscale(path2)

    conv1_3, padded1_3 = preprocess(img1, 3)
    conv1_5, padded1_5 = preprocess(img1, 5)

    conv2_3, padded2_3 = preprocess(img2, 3)
    conv2_5, padded2_5 = preprocess(img2, 5)

    g_img1_3 = gaussian(conv1_3, padded1_3, 3)
    g_img1_5 = gaussian(conv1_5, padded1_5, 5)

    g_img2_3 = gaussian(conv2_3, padded2_3, 3)
    g_img2_5 = gaussian(conv2_5, padded2_5, 5)

    show(img1, "Original")
    show(g_img1_3, "Gaussian 3x3")
    show(g_img1_5, "Gaussian 5x5")

    show(img2, "Original")
    show(g_img2_3, "Gaussian 3x3")
    show(g_img2_5, "Gaussian 5x5")

    # DoG
    path1 = 'filter1_img.jpg'
    path2 = 'filter2_img.jpg'

    img1 = load_image_grayscale(path1)
    img2 = load_image_grayscale(path2)

    conv1x, padded1x = preprocess(img1, 3)
    conv2x, padded2x = preprocess(img2, 3)

    DoG_img1_x = DoG(conv1x, padded1x, 'x')
    DoG_img2_x = DoG(conv2x, padded2x, 'x')

    conv1y, padded1y = preprocess(img1, 3)
    conv2y, padded2y = preprocess(img2, 3)

    DoG_img1_y = DoG(conv1y, padded1y, 'y')
    DoG_img2_y = DoG(conv2y, padded2y, 'y')

    show(DoG_img1_x, "DoG gx")
    show(DoG_img1_y, "DoG gy")

    show(DoG_img2_x, "DoG gx")
    show(DoG_img2_y, "DoG gy")

    # Sobel
    sobel_1 = sobel(DoG_img1_x, DoG_img1_y)
    sobel_2 = sobel(DoG_img2_x, DoG_img2_y)

    show(sobel_1, "Sobel")
    show(sobel_2, "Sobel")


if __name__ == '__main__':
    main()

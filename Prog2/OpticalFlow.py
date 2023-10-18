# CS510 DL & CV. Summer 2023. Prog 2 Optical Flow. Cera Oh
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# Image Paths
path_f1a = './frame1_a.png'
path_flb = './frame1_b.png'
path_f2a = './frame2_a.png'
path_f2b = './frame2_b.png'

# DoG gx filter
DoG_gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# DoG gy filter
DoG_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# Function to load image as grayscale array
def load_image_grayscale(file_path):
    image = cv2.imread(file_path, 0)
    return image


def show(image, title):  # Function to show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def preprocess(image):  # Function to preprocess images and initialize convolution matrix

  # Initialize a numpy array to hold Vx and Vy values
    im_row, im_col = np.shape(image)
    V = np.zeros((im_row, im_col, 2))

  # Zero pad image
    padded = zero_pad(image, 1)

    return V, padded


def zero_pad(image, width):  # Function to pad images with zeros of given width
    padded = cv2.copyMakeBorder(
        image, width, width, width, width, cv2.BORDER_CONSTANT, value=0)
    return padded


def lucas_kanade(V, padded1, padded2):  # Lucas-Kanade Optical Flow Function
    r_count = 0

    for row in padded1:
        end = len(row)
        if np.array_equal(row, padded1[0]):  # zero padded row
            pass  # Do nothing
        else:
            stop = end-1
            col = 1
            while col < stop:
                window1 = np.array([[padded1[r_count-1][col-1], padded1[r_count-1][col], padded1[r_count-1][col+1]],
                                    [row[col-1], row[col], row[col+1]],
                                    [padded1[r_count+1][col-1], padded2[r_count+1][col], padded2[r_count+1][col+1]]])
                window2 = np.array([[padded2[r_count-1][col-1], padded2[r_count-1][col], padded2[r_count-1][col+1]],
                                    [padded2[r_count][col-1], padded2[r_count]
                                        [col], padded2[r_count][col+1]],
                                    [padded2[r_count+1][col-1], padded2[r_count+1][col], padded2[r_count+1][col+1]]])

                w1 = 1/255 * window1  # Normalization
                w2 = 1/255 * window2  # Normalization
                I_x = w1 * DoG_gx  # Element-wise multiplication
                I_y = w1 * DoG_gy  # Element-wise multiplication
                I_t = np.subtract(w2, w1)  # Element-wise subtraction

                A = np.array([[np.sum(I_x * I_x), np.sum(I_x * I_y)],
                             [np.sum(I_x * I_y), np.sum(I_y * I_y)]])
                A_inv = np.linalg.inv(A)  # A^(-1)

                b = np.zeros((2, 1))
                b[0][0] = -1 * np.sum(I_x * I_t)
                b[1][0] = -1 * np.sum(I_y * I_t)

                u_v = np.matmul(A_inv, b)

                V[r_count-1][col-1][0] = u_v[0]
                V[r_count-1][col-1][1] = u_v[1]

                col = col + 1

        r_count = r_count + 1

    return V


def graph(vectors, limit):  # Function to plot vectors

    # Graph points
    x_count = 0
    y_count = 0
    for row in vectors:
        x_count = 0
        for col in row:
            x = x_count
            y = y_count
            dx = col[0] * 10
            dy = col[1] * 10

            if x % limit == 0 and y % limit == 0:
                if (dx > -0.5 and dx < 0.5) and (dy > -0.5 and dy < 0.5):
                    pass  # do nothing
                else:
                    plt.arrow(x, y, dx, dy, head_width=1.5)

            x_count = x_count + 1
        y_count = y_count + 1

    plt.show()


def draw(image, vectors, mult, limit):  # Function to plot Vx and Vy and magnitude
    r_count = 0
    c_count = 0

    # To plot colors on top of the grayscale image
    cimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    r, c = np.shape(image)
    magnitude = np.zeros((r, c))

    for row in cimage:
        c_count = 0
        for col in row:
            v_x = vectors[r_count][c_count][0]
            v_y = vectors[r_count][c_count][1]
            magnitude[r_count][c_count] = math.sqrt(
                math.pow(v_x, 2) + math.pow(v_y, 2))

            new_x = int(r_count + round(v_x))
            new_y = int(c_count + round(v_y))

            # Vx and Vy values to ignore to lessen noise
            if (v_x > -0.3 and v_x < 0.3) and (v_y > -0.3 and v_y < 0.3):
                pass  # do nothing
            else:
                # Plot Vx and Vy as magenta lines from pixel
                cv2.arrowedLine(
                    cimage, (c_count, r_count), (new_y, new_x), (215, 67, 238), 1)

                # Plot only magnitudes that are greater than limit given
                if magnitude[r_count][c_count] > limit:

                    # For scaling
                    new_x_s = int(
                        round(r_count + v_x * magnitude[r_count][c_count] * mult))
                    new_y_s = int(
                        round(c_count + (v_y) * magnitude[r_count][c_count] * mult))

                    # Plot only magnitudes at certain pixels
                    if r_count % 3 == 0 and c_count % 4 == 0:
                        cv2.arrowedLine(
                            cimage, (c_count, r_count), (new_y_s, new_x_s), (0, 255, 0), 1)
            c_count = c_count + 1
        r_count = r_count + 1

    return cimage


def main():

    # Load image
    img1a = load_image_grayscale(path_f1a)
    img1b = load_image_grayscale(path_flb)
    img2a = load_image_grayscale(path_f2a)
    img2b = load_image_grayscale(path_f2b)

    # Zero pad
    flow1, padded1 = preprocess(img1a)
    mask1, padded2 = preprocess(img1b)
    flow3, padded3 = preprocess(img2a)
    mask3, padded4 = preprocess(img2b)

    # Image1: Optical Flow
    calculated1 = lucas_kanade(flow1, padded1, padded2)
    print(f"Image 1 Vx and Vy: \n {calculated1}")
    op_img1 = draw(img1b, calculated1, 150, 0.2)

    # Image2: Optical Flow
    calculated2 = lucas_kanade(flow3, padded3, padded4)
    print(f"Image 2 Vx and Vy:\n {calculated2}")
    op_img2 = draw(img2b, calculated2, 4, 1)

    # Show images
    show(img1a, "Image 1 Frame 1")
    show(img1b, "Image 1 Frame 2")
    show(op_img1, "Image 1 Frame 2: Optical Flow")

    show(img2a, "Image 2 Frame 1")
    show(img2b, "Image 2 Frame 2")
    show(op_img2, "Image 2 Frame 2: Optical Flow")

    # For visualizing Vx and Vy
    graph(calculated1, 10)
    graph(calculated2, 10)


if __name__ == '__main__':
    main()

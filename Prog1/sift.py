# CS510: DL and CV. Summer 2023. Prog 1 Part 3. Cera Oh.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# Image Paths
path1 = "SIFT1_img.jpg"
path2 = "SIFT2_img.jpg"


def load_image(file_path):  # Function that loads an image as an array
    image = cv2.imread(file_path)
    return image


def show(image, title):  # Function to show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def scale(image, scale):  # Function to scale image
    o_height, o_width, channels = np.shape(image)
    new_width = int(o_width * scale)
    new_height = int(o_height * scale)
    new = (new_width, new_height)

    scaled = cv2.resize(image, new, interpolation=cv2.INTER_AREA)
    return scaled


def create_SIFT(img):  # Function to create SIFT image
    sift = cv2.SIFT_create()  # make class for keypoints and description extraction
    keypoints, descriptors = sift.detectAndCompute(img, None)
    sift_img = cv2.drawKeypoints(img, keypoints, img)  # Draw keypoints on img
    return sift_img, keypoints, descriptors


# Function to match keypoints in first image with that of second with min distance
def match(keypt1, des1, keypt2, des2):
    min_match = []  # Array to store min distance matches found
    count_d1 = 0  # counter for descriptor 1

    print("Calculating match...")

    if len(keypt1) >= len(keypt2):  # Check whether there are more keypoints for img 1 or 2
        length = len(keypt1)  # 7005
    else:
        length = len(keypt2)  # 41380

    dist = np.zeros(length)

    for each in des1:  # des1 is 7005 x 128
        counter = 0

        for one in des2:  # des2 is 41380 x 128
            # Calculate L2 distance
            dist[counter] = np.sqrt(np.sum(np.square(np.subtract(each, one))))
            counter = counter + 1
        minimum = np.argmin(dist)  # Find index of min distance
        min_match.append((count_d1, minimum, dist[minimum]))

        print(f'Processing Point {count_d1}')
        count_d1 = count_d1 + 1

    print("Done")
    return min_match


def top_10(keypt1, desc1, keypt2, desc2):  # Function to find top 10 matches
    ten_percent = math.ceil(len(keypt1) * 0.10)

    matched = match(keypt1, desc1, keypt2, desc2)
    # Sort by distance, smallest to largest
    matched = sorted(matched, key=lambda x: x[2])
    top_matched = matched[:ten_percent]  # Save only the first 10%

    return top_matched


# Function to draw lines between the top matching keypoints
def draw_Match(img1, keypt1, img2, keypt2, top):
    toMatch = []  # An array for type DMatch
    for item in top:
        toMatch.append((cv2.DMatch(item[0], item[1], item[2])))

    img_matched = cv2.drawMatches(img1, keypt1, img2, keypt2,
                                  toMatch, None, flags=2, matchesThickness=1)  # Draw lines showing matches
    show(img_matched, 'Matched')


def main():
    img1 = load_image(path1)
    img2 = load_image(path2)

    # resized1 = scale(img1, 0.75)
    # resized2 = scale(img2, 0.60)

    sift_img1, keypt1, desc1 = create_SIFT(img1)
    sift_img2, keypt2, desc2 = create_SIFT(img2)

    show(sift_img1, 'SIFT')
    show(sift_img2, 'SIFT')

    top = top_10(keypt1, desc1, keypt2, desc2)
    draw_Match(img1, keypt1, img2, keypt2, top)


if __name__ == '__main__':
    main()

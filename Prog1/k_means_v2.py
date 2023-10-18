# CS510: DL and CV. Summer 2023. Prog 1 Part 2(i) and (ii). Cera Oh.
import numpy as np
import random
from matplotlib import pyplot as plt
import cv2

# Data Path
data_path = './510_cluster_dataset.txt'

# Image Path
img_path1 = './Kmean_img1.jpg'
img_path2 = './Kmean_img2.jpg'


def load_data(file_path):  # Function that loads the txt data as an 2D array
    with open(file_path, 'r') as data:
        data = np.genfromtxt(file_path, delimiter='  ')
    return data


def load_image(file_path):  # Function that loads an image as an 3D array
    image = cv2.imread(file_path)
    return image


# Function to randomly select k data points/pixels as initial means
def initialize(k, data, dimension):

    if dimension == 2:  # picks k random points from the data set as initial means for 2D
        k_means = np.random.permutation(data)[:k]
    elif dimension == 3:  # picks k random points from the data set as initial means for 3D
        k_means = np.zeros((k, 3))
        r_max, c_max, ch_max = data.shape
        counter = 0
        while counter < k:
            row = random.randrange(r_max-1)
            col = random.randrange(c_max-1)
            k_means[counter] = data[row][col]
            counter = counter + 1
    else:  # Error handling
        print("Wrong dimension was given. It must be 2 or 3.")

    print(f"initial chosen: \n {k_means}")
    return k_means


# Function to assign each point/pixel to the closest mean
def assignment(k, k_means, data, dimension):
    r_count = 0
    make_list = 0

    dist = np.zeros(k)  # array to keep track of Euclidean distances

    # empty list to keep track of point row number for 2D and (row, col) pixel for 3D
    cluster = []

    while make_list < k:
        new = []  # empty list to add
        cluster.append(new)
        make_list = make_list + 1

    if dimension == 2:  # Calculate Euclidean distance and find minimum for 2D
        for row in data:
            counter = 0
            for k in k_means:
                # Euclidean distance between current point and a mean
                dist[counter] = np.sqrt(np.sum(np.square(np.subtract(k, row))))
                counter = counter + 1
            minimum = np.argmin(dist)  # Find index of min distance
            cluster[minimum].append((r_count))
            r_count = r_count + 1

    elif dimension == 3:  # Calculate Euclidean distance and find minimum for 3D
        print("Assigning...")
        for row in data:
            c_count = 0
            for pixel in row:
                counter = 0
                for k in k_means:
                    # Euclidean distance between current pixel and a mean
                    dist[counter] = np.sqrt(
                        np.sum(np.square(np.subtract(k, pixel))))
                    counter = counter + 1
                minimum = np.argmin(dist)  # Find index of min distance
                cluster[minimum].append((r_count, c_count))
                c_count = c_count + 1
            r_count = r_count + 1
    else:  # Error handling
        print("Error assigning")

    return cluster


def recompute(k, data, cluster, dimension):  # Function to recompute means

    if dimension == 3:
        print("Recomputing...")

    # array to hold newly calculated means
    new_k_means = np.zeros((k, dimension))

    list_count = 0

    for lst in cluster:
        if dimension == 2:
            add = [0, 0]
            for item in lst:
                add = np.add(add, data[item])
        elif dimension == 3:
            add = [0, 0, 0]
            for item in lst:
                row = item[0]
                col = item[1]
                add = np.add(add, data[row][col])
        length = len(lst)

        if length != 0:
            add[0] = add[0]/length
            add[1] = add[1]/length
            if dimension == 3:
                add[2] = add[2]/length
        else:
            add[0] = 0
            add[1] = 0
            if dimension == 3:
                add[2] = 0

        new_k_means[list_count][0] = add[0]
        new_k_means[list_count][1] = add[1]
        if dimension == 3:
            new_k_means[list_count][2] = add[2]
        list_count = list_count + 1

    return new_k_means


def k_means_alg(k, k_means, data, dimension):  # K-Means Recursive Algorithm
    print("Running alg...")
    old_k = np.copy(k_means)

    cluster = assignment(k, k_means, data, dimension)
    new_k_means = recompute(k, data, cluster, dimension)

    # Stopping Condition
    comparison = old_k == new_k_means
    equal = comparison.all()

    if (equal == False):
        # Recursively run K-Means Algorithm
        new_k_means, cluster, E = k_means_alg(k, new_k_means, data, dimension)
    else:
        print("K_means finished. The final centroids are:")
        print(new_k_means)

        E = SSE(k, new_k_means, cluster, data, dimension)
        print(f"SSE: {E}")

        # Halt until user inputs 1 to graph
        plot = input("Enter 1 to print:\n")
        plot = int(plot)

        if plot == 1 and dimension == 3:
            colorize(k, new_k_means, cluster, data)
        elif plot == 1 and dimension == 2:
            graph(k, new_k_means, cluster, data)
        else:
            print("Continuing without showing graph...")

    return new_k_means, cluster, E


# Function to run K-Means r times from r differently chosen initial means and find lowest SSE
def run_k_means(data, dimension):
    run = 0
    r = input("Please enter the number of times to run K-Means:\n")
    r = int(r)
    k = input("Please enter the number of clusters:\n")
    k = int(k)

    lowest_SSE = np.zeros(r)  # array to keep track of SSE values
    # array to keep track of final means
    best_cluster = np.zeros((r, k, dimension))

    while (r != 0):
        print(f"Run {run + 1}:")
        k_means = initialize(k, data, dimension)
        final, cluster, E = k_means_alg(k, k_means, data, dimension)
        lowest_SSE[run] = E
        best_cluster[run] = final
        r -= 1
        run += 1

    minimum = np.argmin(lowest_SSE)  # Find index min distance
    print(
        f"Lowest SSE was run {minimum + 1} with value {lowest_SSE[minimum]} with clusters:")
    print(f"{best_cluster[minimum]}")


# Function to calculate Sum of Square Error for K-Means
def SSE(k, k_means, cluster, data, dimension):
    rss = np.zeros(k)
    E = 0

    index = 0
    for c in cluster:
        if dimension == 2:  # for 2D
            for pt in c:
                rss[index] += np.square(
                    np.sum(np.subtract(k_means[index], data[pt])))
            index = index + 1

        if dimension == 3:  # for 3D
            for pixel in c:
                row = pixel[0]
                col = pixel[1]
                rss[index] += np.square(np.sum(np.subtract(
                    k_means[index], data[row][col])))
            index = index + 1

    for each in rss:
        E += each

    return E


def show(image, title):  # Function to show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def graph(k, k_means, cluster, data):  # Function to plot K-Means clusters
    colors = np.zeros((k, 3))  # array to hold colors

    # Assign random colors per cluster
    counter = 0
    while counter < k:
        colors[counter, 0] = random.random()
        colors[counter, 1] = random.random()
        colors[counter, 2] = random.random()
        counter += 1

    # Graph points
    counter = 0
    for c in cluster:
        for row in c:
            x = data[row][0]
            y = data[row][1]
            r = colors[counter][0]
            b = colors[counter][1]
            g = colors[counter][2]
            color = (r, b, g)
            plt.scatter(x, y, c=np.array([color]))
        counter = counter + 1

    # Plot means
    c_x, c_y = k_means.T
    plt.scatter(c_x, c_y, c='k')

    plt.show()


def colorize(k, k_means, cluster, data):  # Function to render image
    colors = np.zeros((k, 3))  # array to hold colors

    # Copy original image for recoloring
    new_img = np.copy(data)

    counter = 0
    while counter < k:
        colors[counter][0] = k_means[counter][0]
        colors[counter][1] = k_means[counter][1]
        colors[counter][2] = k_means[counter][2]
        counter += 1

    count = 0
    for c in cluster:
        for pixel in c:
            row = pixel[0]
            col = pixel[1]
            new_img[row][col][0] = colors[count][0]
            new_img[row][col][1] = colors[count][1]
            new_img[row][col][2] = colors[count][2]
        count = count + 1

    show(new_img, f"K-Means k={k}")  # Function to show image


def main():
    # 2D
    data = load_data(data_path)
    run_k_means(data, 2)
    run_k_means(data, 2)
    run_k_means(data, 2)

    # 3D
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    show(img1, "Original")
    run_k_means(img1, 3)
    run_k_means(img1, 3)

    show(img2, "Original")
    run_k_means(img2, 3)
    run_k_means(img2, 3)


if __name__ == '__main__':
    main()

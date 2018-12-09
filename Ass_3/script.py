from __future__ import division
import cv2
import numpy as np
import math

# Mohab Amr Abdelfatah T07  34-5862
# Abdelrahman Hisham   T07  34-1203

left_img = None
right_img = None
new_img = None
H = 4
W = 4


def read_images():
    left_img = cv2.imread('./left.jpg', 0)
    right_img = cv2.imread('./right.jpg', 0)

    # identical imgs in size
    H, W = left_img.shape[:2]
    H2, W2 = right_img.shape[:2]
    print(H, W)
    print(H2, W2)
    W = W - 1
    new_img = [[0 for x in range(W)] for y in range(H)]
    # H = int (H/10)
    # W = int (W/10)
    print(H, W)

    # left_img = [[9, 8, 9, 2], [8, 7, 9, 3], [9, 9, 2, 2], [5, 8, 1, 1]]
    # right_img = [[2, 3, 1, 3], [2, 8, 7, 8], [1, 9, 8, 9], [1, 9, 8, 2]]

    # disparity_window(left_img, right_img, 1, 1, (0, 1), (0, 1), 3)

    for y in range(0, H):
        for x in range(0, W):
            # for each pixel, use the limits to get all pixels in this range and use the min desparity
            # disp range func missing
            x = 1
            disparity_window(left_img, right_img, new_img, y, x, (0, 1), (0, 1), 7)
    print "finished disp"
    # print new_img
    write_img(new_img)
    print "finished"


def disparity_window(left_img, right_img, new_img, y, x, x_range, y_range, kernel_size):
    curr_min = None
    choosen_vector = None
    for j in range(y_range[0], y_range[1]+1):
        for i in range(x_range[0], x_range[1]+1):
            right_y = y + j
            right_x = x + i
            curr_disp = get_ssd(left_img, right_img, y, x,
                                right_y, right_x, kernel_size)
            if(curr_disp == None):
                continue
            if(curr_min == None):
                curr_min = curr_disp
                choosen_vector = {'x': i, 'y': j}
            else:
                if curr_disp < curr_min:
                    curr_min = curr_disp
                    choosen_vector = {'x': i, 'y': j}
    if(curr_min != None):
        pix_val = np.clip(curr_min, 0, 255)
        print pix_val
        new_img[y][x] = pix_val 

    print(choosen_vector)
    print "\n\n"


def write_img(new_img):
    imgToWrite = np.array(new_img)
    cv2.imwrite('./disparity.jpg', imgToWrite)


def get_ssd(left_img, right_img, left_y, left_x, right_y, right_x, kernel_size):
    diff_sum = 0
    kernel_limit = int(math.floor(kernel_size / 2))
    for j in range(-kernel_limit, kernel_limit+1):
        for i in range(-kernel_limit, kernel_limit+1):
            left_y_calc = left_y + j
            left_x_calc = left_x + i
            right_y_calc = right_y + j
            right_x_calc = right_x + i
            # if(left_y_calc < 0 or left_x_calc < 0 or left_y_calc > H or left_x_calc > W or
            #    right_y_calc < 0 or right_x_calc < 0 or right_y_calc > H or right_x_calc > W):
            #     return None
            try:
                # print left_img[left_y_calc][left_x_calc]
                # print right_img[right_y_calc][right_x_calc]
                # print "\n"
                int_diff = left_img[left_y_calc][left_x_calc] - \
                    right_img[right_y_calc][right_x_calc]
                diff_squared = int_diff ** 2
                diff_sum += diff_squared
            except IndexError:
                return None
    return diff_sum


if __name__ == '__main__':
    print("Starting..\n")
    read_images()

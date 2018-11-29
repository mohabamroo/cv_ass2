from __future__ import division
import cv2
import numpy as np
import math
from scipy.signal import convolve2d
from skimage.restoration import estimate_sigma

# Mohab Amr Abdelfatah T07  34-5862
# Abdelrahman Hisham   T07  34-1203

original_noise = 0
original_blurry = 0
noise_index = [2, 5, 6, 8]
blur_index = [3, 5, 7, 8]
color_index = [4, 6, 7, 8]
# Noise tut
# http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html#sphx-glr-auto-examples-filters-plot-denoise-py

# Smoothing tutorial
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html


def estimate_noise(img, i=0):
    noise = estimate_sigma(img, multichannel=True, average_sigmas=True)
    if(noise > original_noise):
        return smooth_img(img, i)
    return img


def measure_noise(img, i=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    print "Img #" + str(i) +" noise: " + str(original_noise)
    return sigma


def smooth_img(img, i=0):
    kernel = np.ones((5, 5), np.float32)/25
    dst = cv2.filter2D(img, -1, kernel)
    # TODO: play with kernel size
    median = cv2.medianBlur(img, 7)
    cv2.imwrite('./imagesA2/'+str(i)+'_1smoothed.jpg', median)
    return median

def unsharp_filter(img, k=0.5, i=0):
    gaussian_3 = cv2.GaussianBlur(img, (11, 11), 50.0)
    edges = cv2.subtract(img, gaussian_3)
    sharped = cv2.addWeighted(img, 1, edges, k, 0)
    cv2.imwrite('./imagesA2/'+str(i)+'_sharpedWEdges.jpg', sharped)
    return sharped

def detect_sobel(img, i=0):
    blurry = blur_measure(img, i)
    print "Image #" + str(i) + " blur: " + str(blurry)
    if(blurry > original_blurry):
        # blurred imgs
        gaussian_3 = cv2.GaussianBlur(img, (11, 11), 50.0)
        unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 1)

        # Calculate the Laplacian
        # lap = cv2.Laplacian(img, cv2.CV_64F)
        # Calculate the sharpened image
        # sharp = img - 0.5*lap

        cv2.imwrite('./imagesA2/'+str(i)+'_sharped.jpg', unsharp_image)
        return unsharp_image
    return img


def blur_measure(im, i=0):
    """ See cv::videostab::calcBlurriness """
    # the lesser value means more sharpness
    H, W = im.shape[:2]
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    cv2.imwrite('./imagesA2/'+str(i)+'_sobelX.jpg', gx)
    cv2.imwrite('./imagesA2/'+str(i)+'_sobelY.jpg', gy)
    GXY = np.add(gx, gy)
    print("Blur of GXY: " + str(i))
    for y in range(0, H):
        for x in range(0, W):
            GXY[y][x] = np.clip(GXY[y][x], 0, 255)
    cv2.imwrite('./imagesA2/'+str(i)+'_sobelGXY.jpg', GXY)
    print(GXY.var())
    return GXY.var()
    # norm_gx, norm_gy = cv2.norm(gx), cv2.norm(gy)
    # print("edg value: " + str(np.sqrt((gx ** 2 + gy ** 2))))
    # return 1.0 / ((norm_gx ** 2 + norm_gy ** 2) / (H * W + 1e-6))


def getIntensity(original_img):
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    w = img.shape[1]
    maxI = img[0][0]
    minI = img[0][0]
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            currentMax = img[y][x]
            currentMin = img[y][x]
            if currentMax > maxI:
                maxI = currentMax
            if currentMin < minI:
                minI = currentMin
    return maxI, minI


def set_original_attrs():
    original_img = cv2.imread('./imagesA2/1.jpg')

    global original_noise
    original_noise = estimate_sigma(
        original_img, multichannel=True, average_sigmas=True)
    print "Original noise: " + str(original_noise)
    original_noise = measure_noise(original_img, 1)
    print "Original noise(variance): " + str(original_noise)

    global original_blurry
    original_blurry = blur_measure(original_img, 1)
    print "Original blur: " + str(original_blurry)

    print "\n"


# contrast stretching
def correct_color(original_img, i=0):
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    maxI, minI = getIntensity(original_img)
    range_covered = ((maxI - minI) / 255) * 100
    print("Range covered by #" + str(i) + ": " + str(range_covered))
    if(range_covered > 70):
        return img
    else:
        if(i not in color_index):
            print("Img #" + str(i) + " mismatch color")
    h = img.shape[0]
    w = img.shape[1]
    newImg = [[0 for x in range(w)] for y in range(h)]
    for y in range(0, h):
        for x in range(0, w):
            newImg[y][x] = ((img[y][x] - minI) / (maxI - minI)) * 255
            newImg[y][x] = np.clip(newImg[y][x], 0, 255)
    imgToWrite = np.array(newImg)
    cv2.imwrite('./imagesA2/'+str(i)+'_corrected.jpg', imgToWrite)
    return imgToWrite


def main():
    set_original_attrs()
    for i in range(2, 9):
        img = cv2.imread('./imagesA2/'+str(i)+'.jpg')
        imgN = estimate_noise(img, i)
        if(measure_noise(img, i) > 10):
            if i not in noise_index:
                print("Img #" + str(i) + " mismatch noise")
            img = smooth_img(img)
        if(blur_measure(img, i) < 700):
            img = unsharp_filter(img, 0.6, i)
            if(i not in blur_index):
                print("Img #" + str(i) + " mismatch blurry")
        # imgS = detect_sobel(imgN, i)
        imgCont = correct_color(img, i)
        cv2.imwrite('./imagesA2/'+str(i)+'_final.jpg', imgCont)
        print "\n"


if __name__ == '__main__':
    print("Starting..\n")
    main()

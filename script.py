from __future__ import division
import cv2
import numpy as np
from skimage.restoration import estimate_sigma

original_noise = 0
original_blurry = 0

# Noise tut
# http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html#sphx-glr-auto-examples-filters-plot-denoise-py

# Smoothing tutorial
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html


def estimate_noise(img, i=0):
    noise = estimate_sigma(img, multichannel=True, average_sigmas=True)
    print "Image #" + str(i) + " noise: " + str(noise)
    if(noise > original_noise):
        return smooth_img(img, i)
    return img


def smooth_img(img, i=0):
    kernel = np.ones((5, 5), np.float32)/25
    dst = cv2.filter2D(img, -1, kernel)
    # TODO: play with kernel size
    median = cv2.medianBlur(img, 7)
    cv2.imwrite('./imagesA2/'+str(i)+'_smoothed.jpg', median)
    return median


def detect_sobel(img, i=0):
    blurry = blur_measure(img)
    print "Image #" + str(i) + " blur: " + str(blurry)
    if(blurry > original_blurry):
        # blurred imgs
        gaussian_3 = cv2.GaussianBlur(img, (11, 11), 50.0)
        unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 1)

        # Calculate the Laplacian
        # lap = cv2.Laplacian(img, cv2.CV_64F)
        # Calculate the sharpened image
        # sharp = img - 0.5*lap

        cv2.imwrite('./imagesA2/'+str(i)+'_sharpened.jpg', unsharp_image)
        return unsharp_image
    return img

def blur_measure(im):
    """ See cv::videostab::calcBlurriness """
    # the lesser value means more sharpness
    H, W = im.shape[:2]
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    norm_gx, norm_gy = cv2.norm(gx), cv2.norm(gy)
    return 1.0 / ((norm_gx ** 2 + norm_gy ** 2) / (H * W + 1e-6))


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

    global original_blurry
    original_blurry = blur_measure(original_img)
    print "Original blur: " + str(original_blurry)

    print "\n"


# contrast stretching
def correct_color(original_img, i=0):
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    maxI, minI = getIntensity(original_img)
    h = img.shape[0]
    w = img.shape[1]
    newImg = [[0 for x in range(w)] for y in range(h)]
    for y in range(0, h):
        for x in range(0, w):
            newImg[y][x] = ((img[y][x] - minI) / (maxI - minI)) * 255
    imgToWrite = np.array(newImg)
    cv2.imwrite('./imagesA2/'+str(i)+'_corrected.jpg', imgToWrite)
    return imgToWrite


def main():
    set_original_attrs()
    for i in range(2, 9):
        img = cv2.imread('./imagesA2/'+str(i)+'.jpg')
        imgN = estimate_noise(img, i)
        imgS = detect_sobel(imgN, i)
        imgCont = correct_color(imgS, i)
        cv2.imwrite('./imagesA2/'+str(i)+'_final.jpg', imgCont)
        print "\n"


if __name__ == '__main__':
    print("Starting..\n")
    main()

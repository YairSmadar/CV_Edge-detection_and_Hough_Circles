from ex2_utils import *
import matplotlib.pyplot as plt
import cv2 as cv
import time


def conv1Demo():
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    kernel = np.array([0, 1, 2, 3, 4])
    arr = conv1D(signal, kernel)
    print(arr)


def conv2Demo():
    inImage = np.array([[0.1, 0.2, 0.3, 0.1],
                        [0.4, 0.5, 0.6, 0.1],
                        [0.7, 0.8, 0.9, 0.1]])
    # print(len(inImage))
    # print(len(inImage[0]))
    # kernel = np.array([[0.1, 0.2, 0.1]])
    # kernel = np.array([[0.1],
    #                    [0.1],
    #                    [0.1]])
    kernel = np.array([[0.1, 0.2],
                       [0.4, 0.5],
                       [0.7, 0.8]])

    new_img = conv2D(inImage, kernel)
    print(new_img)


def derivDemo():
    im = cv.imread("beach.jpg", cv.IMREAD_GRAYSCALE)
    # cv.imshow("a", im)
    # cv.waitKey(0)
    im = im/255
    DiractionG, MagG, dir_img_X, dir_img_Y = convDerivative(im)
    cv.imshow("a", MagG)
    cv.waitKey(0)


def edgeDemo():
    im = cv.imread("boxman.jpg", cv.IMREAD_GRAYSCALE)
    # im = im / 255
    # openCV, mySol = edgeDetectionSobel(im, 0.5)
    # cv.imshow('my solution', mySol)
    # cv.imshow('openCV', openCV)
    # cv.waitKey(0)

    # log_im = edgeDetectionZeroCrossingLOG(im)
    # cv.imshow('log_im', log_im)
    # cv.waitKey(0)
    openCV, mySol = edgeDetectionCanny(im, 0.4, 0.7)
    cv.imshow('my solution', mySol)
    cv.imshow('openCV', openCV)
    cv.waitKey(0)
    pass


def houghDemo():
    im = cv.imread("coins.jpg", cv.IMREAD_GRAYSCALE)
    lst = houghCircle(im, 50, 120)
    for i in lst:
        cv.circle(im, (i[1], i[2]), i[0], color=(150, 40, 100), thickness=1)
    print(lst)
    cv.imshow("after", im)
    cv.waitKey(0)




def main():
    # conv1Demo()
    # conv2Demo()
    # derivDemo()
    # blurDemo()
    # edgeDemo()
    houghDemo()
    # inImage = np.array([[0.1, 0.2, 0.3, 0.1],
    #                     [0.4, 0.5, 0.6, 0.1],
    #                     [0.7, 0.8, 0.9, 0.1]])
    # print(len(inImage))
    # print(len(inImage[0]))
    #


if __name__ == '__main__':
    main()

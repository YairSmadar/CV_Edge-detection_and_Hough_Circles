import numpy as np
import cv2 as cv


##
# Convolve a 1-D array with a given kernel
# :param inSignal: 1-D array
# :param kernel1: 1-D array as a kernel
# :return: The convolved array
##
def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    flip_kernel = np.flip(kernel1)

    if len(inSignal) < len(kernel1):
        kernel1, inSignal = inSignal, kernel1

    if len(inSignal) == 0 or len(kernel1) == 0:
        raise ValueError("Cannot convolve when array length = 0")

    output = np.arange(len(inSignal) + len(kernel1) - 1, dtype='float')
    kernel1_runner = len(flip_kernel) - 1
    start_from_end = 1
    start_kernel = len(flip_kernel) - 1
    for i in range(len(inSignal) + len(kernel1) - 1):
        if i < len(flip_kernel):
            a = inSignal[:i + 1]
            b = flip_kernel[kernel1_runner:]
            num = a * b
            new_pixel = np.sum(num)
        elif i < len(inSignal):
            a = inSignal[start_from_end:i + 1]
            b = flip_kernel
            num = a * b
            new_pixel = np.sum(num)
        else:
            a = inSignal[start_from_end:i + 1]
            b = flip_kernel[:start_kernel]
            num = a * b
            new_pixel = np.sum(num)
        output[i] = new_pixel

        if len(flip_kernel) <= i < len(inSignal):
            start_from_end += 1
            kernel1_runner -= 1
        elif i < len(flip_kernel):
            kernel1_runner -= 1
        elif i >= len(inSignal):
            start_from_end += 1
            start_kernel -= 1

    return output


##
# Convolve a 2-D array with a given kernel
# :param inImage: 2D image
# :param kernel2: A kernel
# :return: The convolved image
##
def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    # print(cv.filter2D(src=inImage, ddepth=-1, kernel=kernel2, borderType=cv.BORDER_REPLICATE))
    # print(len(kernel2[0]))
    # print(len(kernel2))
    if kernel2.ndim == 1:
        up = down = np.floor(len(kernel2) / 2).astype(int)
        left = right = 0
    elif kernel2.ndim == 2:
        if inImage.shape[1] == 1:  # 1D vector up to down
            up = down = np.floor(len(kernel2) / 2).astype(int)
            left = right = 0
        elif len(kernel2) == len(kernel2[0]):
            if len(kernel2) % 2 == 0:  # even
                up = left = np.floor(len(kernel2) / 2).astype(int)
                down = right = np.floor(len(kernel2) / 2).astype(int) - 1
            else:  # odd
                up = down = np.floor(len(kernel2) / 2).astype(int)
                left = right = np.floor(len(kernel2[0]) / 2).astype(int)
        elif len(kernel2[0]) >= len(kernel2):
            if len(kernel2[0]) % 2 == 0:  # right even
                left = np.floor(len(kernel2[0]) / 2).astype(int)
                right = np.floor(len(kernel2[0]) / 2).astype(int) - 1
            else:  # right odd
                left = np.floor(len(kernel2[0]) / 2).astype(int)
                right = np.floor(len(kernel2[0]) / 2).astype(int)
            if len(kernel2) % 2 == 0:  # down even
                up = np.floor(len(kernel2) / 2).astype(int)
                down = np.floor(len(kernel2) / 2).astype(int) - 1
            else:
                up = np.floor(len(kernel2) / 2).astype(int)
                down = np.floor(len(kernel2) / 2).astype(int)
        else:  # len(kernel2[0]) < len(kernel)
            if len(kernel2[0]) % 2 == 0:  # right even
                left = np.floor(len(kernel2[0]) / 2).astype(int)
                right = np.floor(len(kernel2[0]) / 2).astype(int) - 1
            else:  # right odd
                left = np.floor(len(kernel2[0]) / 2).astype(int)
                right = np.floor(len(kernel2[0]) / 2).astype(int)
            if len(kernel2) % 2 == 0:  # down even
                up = np.floor(len(kernel2) / 2).astype(int)
                down = np.floor(len(kernel2) / 2).astype(int) - 1
            else:
                up = np.floor(len(kernel2) / 2).astype(int)
                down = np.floor(len(kernel2) / 2).astype(int)

    paddingImg = np.pad(inImage, ((up, down), (left, right)), 'edge')
    # print(paddingImg)
    # print()
    output = np.zeros_like(inImage)
    output = output.flatten()
    if kernel2.ndim == 1:
        row_start = 0
        row_end = len(kernel2)
        col_start = 0
        col_end = 1
    else:
        row_start = 0
        row_end = len(kernel2)
        col_start = 0
        col_end = len(kernel2[0])
    for i in range(len(output)):
        sub_mat = paddingImg[row_start:row_end, col_start:col_end]
        if kernel2.ndim == 1:
            temp_sub = np.zeros(len(sub_mat), dtype='float')
            for j in range(len(sub_mat)):
                temp_sub[j] = sub_mat[j][0]
            sub_mat = temp_sub
        # print(sub_mat)
        if kernel2.ndim == 1:
            temp_mat = sub_mat * kernel2
        else:
            temp_mat = sub_mat * kernel2
        output[i] = np.sum(temp_mat)
        if col_end <= len(paddingImg[0]) - 1:
            col_end += 1
            col_start += 1
        else:
            if kernel2.ndim == 1:
                col_end = 1
            else:
                col_end = len(kernel2[0])
            col_start = 0
            if row_end <= len(paddingImg) - 1:
                row_end += 1
                row_start += 1
    return output.reshape(inImage.shape)


##
# Calculate gradient of an image
# :param inImage: Grayscale iamge
# :return: (directions, magnitude,x_der,y_der)
##
def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    dir = np.array([[1, 0, -1]])
    dir_T = dir.T
    # dir_img_X = conv2D(inImage, dir)
    # dir_img_Y = conv2D(inImage, dir_T)
    dir_img_X = cv.filter2D(src=inImage, ddepth=-1, kernel=dir, borderType=cv.BORDER_REPLICATE)
    dir_img_Y = cv.filter2D(src=inImage, ddepth=-1, kernel=dir_T, borderType=cv.BORDER_REPLICATE)

    MagG = np.sqrt(np.power(dir_img_X, 2) + np.power(dir_img_Y, 2))
    DiractionG = np.arctan2(dir_img_Y, dir_img_X)
    return DiractionG, MagG, dir_img_X, dir_img_Y


##
# Detects edges using the Sobel method
# :param img: Input image
# :param thresh: The minimum threshold for the edge response
# :return: opencv solution, my implementation
##
def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    # open CV
    img = img * 255
    x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3, scale=1, borderType=cv.BORDER_REPLICATE)
    y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3, scale=1, borderType=cv.BORDER_REPLICATE)
    absx = cv.convertScaleAbs(x)
    absy = cv.convertScaleAbs(y)
    openCV = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
    boolmat = openCV < thresh
    openCV[boolmat] = 0
    # mine
    img = img / 255
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = Gx.T
    imgX = conv2D(img, Gx)
    imgY = conv2D(img, Gy)
    mySol = np.sqrt(np.power(imgX, 2) + np.power(imgY, 2))
    boolmat = mySol < thresh
    mySol[boolmat] = 0

    return openCV, mySol


#  get GaussianKernel
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


##
# Detecting edges using the "ZeroCrossingLOG" method
# :param img: Input image
# :return: :return: Edge matrix
##
def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    Gaus_kernel = gkern()
    gaus_img = conv2D(img, Gaus_kernel)

    sec_derv_kernel = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]])
    derv_gaus_img = conv2D(gaus_img, sec_derv_kernel)
    plus = derv_gaus_img > 0
    minus = derv_gaus_img < 0
    zero = derv_gaus_img == 0

    ans = np.zeros_like(derv_gaus_img)
    for i in range(len(derv_gaus_img)):
        for j in range(len(derv_gaus_img[0]) - 1):
            if zero[i][j] and plus[i][j - 1] and minus[i][j + 1]:
                ans[i][j] = 1  # img[i][j]
            if plus[i][j] and minus[i][j + 1]:
                ans[i][j] = 1  # img[i][j]

    return ans


def getSovelDirAndImage(img):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = Gx.T
    # imgX = conv2D(img, Gx)
    # imgY = conv2D(img, Gy)
    imgX = cv.filter2D(src=img, ddepth=-1, kernel=Gx, borderType=cv.BORDER_REPLICATE)
    imgY = cv.filter2D(src=img, ddepth=-1, kernel=Gy, borderType=cv.BORDER_REPLICATE)

    mySol = np.sqrt(np.power(imgX, 2) + np.power(imgY, 2))
    DiractionG = np.arctan2(imgY, imgX)
    return DiractionG, mySol


##
# Detecting edges usint "Canny Edge" method
# :param img: Input image
# :param thrs_1: T1
# :param thrs_2: T2
# :return: opencv solution, my implementation
##
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    OpenCV = cv.Canny(img, int(thrs_1 * 255), int(thrs_2 * 255))

    #  step a
    img = img / 255
    Gaus_kernel = gkern()
    gaus_img = cv.filter2D(src=img, ddepth=-1, kernel=Gaus_kernel, borderType=cv.BORDER_REPLICATE)
    # gaus_img = conv2D(img, Gaus_kernel)

    # step b + c
    # DiractionG, MagG, imgX, imgY, = convDerivative(gaus_img)
    DiractionG, MagG = getSovelDirAndImage(gaus_img)
    DiractionG_180 = DiractionG * 180. / np.pi
    DiractionG_180[DiractionG_180 < 0] += 180

    # Note: I got really tangled up with this part so I got some help from the internet to implement it
    ans = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            try:
                right_check = 255
                left_check = 255
                # 135
                if DiractionG_180[i][j] >= 112.5 and DiractionG_180[i][j] < 157.5:
                    right_check = MagG[i - 1][j - 1]
                    left_check = MagG[i + 1][j + 1]
                # 90
                elif DiractionG_180[i][j] >= 67.5 and DiractionG_180[i][j] < 112.5:
                    right_check = MagG[i + 1][j]
                    left_check = MagG[i - 1][j]
                # 45
                elif DiractionG_180[i][j] >= 22.5 and DiractionG_180[i][j] < 67.5:
                    right_check = MagG[i + 1][j - 1]
                    left_check = MagG[i - 1][j + 1]
                # 0
                elif (DiractionG_180[i][j] >= 0 and DiractionG_180[i][j] < 22.5) or (
                        DiractionG_180[i][j] >= 157.5 and DiractionG_180[i][j] <= 180):
                    right_check = MagG[i][j + 1]
                    left_check = MagG[i][j - 1]

                max_check = np.maximum(right_check, left_check)
                if MagG[i][j] >= max_check:
                    ans[i][j] = MagG[i][j]
                else:
                    ans[i][j] = 0
            except:
                continue

    # Hysteresis
    # cv.imshow("aa", ans)
    # cv.waitKey(0)
    mat_strong = ans > thrs_1
    mat_weak = np.logical_and(thrs_2 <= ans, ans <= thrs_1)
    finel_ans = np.zeros_like(ans)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if mat_weak[i][j]:
                if (mat_strong[i + 1][j - 1] or (mat_strong[i + 1][j]) or (mat_strong[i + 1][j + 1])
                        or (mat_strong[i][j - 1]) or (mat_strong[i][j + 1])
                        or (mat_strong[i - 1][j - 1]) or (mat_strong[i - 1][j]) or (mat_strong[i - 1][j + 1])):
                    finel_ans[i][j] = 1
                    mat_strong[i][j] = True
                else:
                    finel_ans[i][j] = 0
    finel_ans[mat_strong] = 1
    return OpenCV, finel_ans


##
# Find Circles in an image using a Hough Transform algorithm extension
# :param I: Input image
# :param minRadius: Minimum circle radius
# :param maxRadius: Maximum circle radius
# :return: A list containing the detected circles,
# [(x,y,radius),(x,y,radius),...]
##
def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    imgCannyEdge = cv.Canny(img, int(100), int(200))
    # cv.imshow("aa", imgCannyEdge)
    # cv.waitKey(0)

    houghSpace = np.zeros((int(max_radius + 0.5), img.shape[0] + 2 * max_radius, img.shape[1] + 2 * max_radius), dtype='uint8')
    # B = np.zeros((int(max_radius + 0.5), img.shape[0] + 2 * max_radius, img.shape[1] + 2 * max_radius), dtype='uint8')

    theta_arr = np.arange(0, 360) * np.pi / 180  # thetas array

    edges_arr = np.argwhere(imgCannyEdge[:, :])  # edges array (the position of all turn on pixels)

    lst = []
    for current_radius in range(int(min_radius), int(max_radius + 0.5)):
        circle_print_arr = np.zeros((2 * int((current_radius + 1)), 2 * int((current_radius + 1))), dtype='uint16')  # circle according to the radius

        # set circle with the current radius
        for angle in theta_arr:
            x = int(np.round(current_radius * np.cos(angle)))
            y = int(np.round(current_radius * np.sin(angle)))
            circle_print_arr[current_radius + x + 1, current_radius + y + 1] = 1

        num_of_pixel_in_circle = np.argwhere(circle_print_arr).shape[0]
        for x, y in edges_arr:
            X = [x - current_radius + max_radius - 1, x + current_radius + max_radius + 1]
            Y = [y - current_radius + max_radius - 1, y + current_radius + max_radius + 1]
            houghSpace[current_radius, X[0]:X[1], Y[0]:Y[1]] += np.uint16(circle_print_arr)
        threshold = 0.5*num_of_pixel_in_circle
        houghSpace[current_radius][houghSpace[current_radius] < threshold] = 0
        print("R = ", current_radius)

    # get final list of (radius, X, Y)
    for r, x, y in np.argwhere(houghSpace):
        lst.append((r, x, y - max_radius))

    return lst

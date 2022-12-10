import threading
from math import sqrt

import cv2  # opencv
import time

import numpy as np

from FingerTips import fingertips
from LandMark import find_Landmark, connectFinger

checkerfordoc = 0


def resize_image(image):
    # cropped_image = image[0:430, 110:530]
    # Burası ROI islemin yapılması gereken yer
    cropped_image = image[40:390, 115:530]
    return cropped_image


def ShowImagev2(image, count, name):
    image = resize_image(image)
    start = time.time()
    end = time.time()
    while end - start < count:
        cv2.imshow(name, image)
        cv2.waitKey(1)
        end = time.time()


def ShowImage(image, count):
    start = time.time()
    end = time.time()
    while end - start < count:
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        end = time.time()


def DoctorChecker(FingerTips):
    i = 0
    if len(FingerTips) > 5:
        while i < 5:
            x1, y1 = FingerTips[i]
            j = 5
            while j < len(FingerTips):
                x2, y2 = FingerTips[j]
                results = int((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5)
                if results <= 5:
                    print(i, results)
                    return i
                j = j + 1
            i = i + 1
    return -1


def inc(x, y, x1, y1):
    x2 = x1 - x
    y2 = y1 - y
    if x2 == 0:
        return [1, 0]
    elif y2 == 0:
        return [0, 1]
    elif y2 / x2 < 0:
        if x2 > y2:
            r_x = 1
            r_y = -1 * (x2 / y2)
        else:
            r_x = y2 / x2
            r_y = -1
        return checkInc(r_x, r_y)
    else:
        if x2 > y2:
            r_x = -1
            r_y = x2 / y2
        else:
            r_x = -1 * (y2 / x2)
            r_y = 1
        return checkInc(r_x, r_y)


def checkInc(plus_x, plus_y):
    if plus_y > 2:
        while plus_y > 2:
            plus_y = plus_y / 2
            plus_x = plus_x / 2
    elif plus_y < -2:
        while plus_y < -2:
            plus_y = plus_y / 2
            plus_x = plus_x / 2
    if plus_x > 2:
        while plus_x > 2:
            plus_y = plus_y / 2
            plus_x = plus_x / 2
    elif plus_x < -2:
        while plus_x < -2:
            plus_y = plus_y / 2
            plus_x = plus_x / 2
    return [plus_x, plus_y]


def give_point_root(image, x, y, x1, y1, direction):
    stop = 0
    if direction == 1:
        x1 = -1 * x1
        y1 = -1 * y1
    count = 0
    x_r = x + x1
    y_r = y + y1
    while True:
        a = int(x_r)
        s = int(y_r)
        b, g, r = (image[s, a])
        # print("bgr:", b, g, r)
        if r < 100:
            return [a, s]
        elif count >= 1 and r < 100:  # buraya artık hiç girmiyor
            x_r = int(x_r)
            y_r = int(y_r)
            return [x_r, y_r]
        elif r < 100:  # buraya artık hiç girmiyor
            count = count + 1
        x_r = x_r + x1
        y_r = y_r + y1
        stop = stop + 1
        if stop == 200:
            return [x, y]


def give_point_root_up(x, y, x1, y1, x2, y2, direction):
    metre_x = x - x1
    metre_y = y - y1
    return x2 - metre_x, y2 - metre_y


def find_boolean(point):
    if point[0][0] > point[2][0]:
        return 1
    return 2


def find_Y1(denklem, x1):
    Y1 = denklem[1] - ((denklem[0] - x1) * denklem[2])
    return Y1


def egim_bulma(x, y, x1, y1):
    egim = (y1 - y) / (x1 - x)
    return egim


def find_X1(denklem, y1):
    X1 = denklem[0] - ((denklem[1] - y1) / denklem[2])
    return X1


def root_paralel(image, x, y):
    check = True
    yedek = x
    count = 0
    look = 0

    while check:
        b, g, r = (image[y, x])
        # print("bgr:", b, g, r)
        if r < 100 and count < 1:
            count = count + 1
        elif count >= 1 and r < 80:
            left = x
            check = False
        if look > 20:
            return [yedek - 15, y, yedek + 15]
        x = x - 1
        look = look + 1
    x = yedek
    check = True
    count = 0
    look = 0
    while check:
        b, g, r = (image[y, x])
        # print("bgr:", b, g, r)
        if r < 100 and count < 1:
            count = count + 1
        elif count >= 1 and r < 80:
            right = x
            check = False
        if look > 20:
            return [yedek - 15, y, yedek + 15]
        x = x - 1
        look = look + 1
    # print(left, y, right)
    # cv2.circle(image, (right, y), 1, (255, 0, 255), cv2.FILLED)
    # cv2.circle(image, (left, y), 1, (255, 0, 255), cv2.FILLED)
    return [left, y, right]


def find_linear_denklem(x, y, x1, y1):
    m = egim_bulma(x, y, x1, y1)
    denklem = [x, y, m]
    return denklem


def bilinear_interpolation(image, x, y):
    x = x

    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width) / (x)
    scale_y = (height) / (y)

    new_image = np.zeros((int(y), int(x), image.shape[2]))

    for k in range(3):
        for i in range(int(y)):
            for j in range(int(x)):
                x = (j + 0.5) * (scale_x) - 0.5
                y = (i + 0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width - 2)
                y_int = min(y_int, height - 2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int + 1, k]
                c = image[y_int + 1, x_int, k]
                d = image[y_int + 1, x_int + 1, k]

                pixel = a * (1 - x_diff) * (1 - y_diff) + b * (x_diff) * \
                        (1 - y_diff) + c * (1 - x_diff) * (y_diff) + d * x_diff * y_diff

                new_image[i, j, k] = pixel.astype(int)

    return new_image


def start_finger_try(img, point, growth, end, Time_count, count):
    if growth > end:
        pass
        # end of
    else:
        image = img.copy()
        boolean = find_boolean(point)
        # başlangıç noktası ile bitiş noktası arasındaki doğrunun denklemi
        denklem_long = find_linear_denklem(point[4][0], point[4][1], point[5][0], point[5][1])

        # başlangıç noktası ve x'e paralel köşe noktaları bulundu
        root_connect = root_paralel(image, point[4][0], point[4][1])

        # print(root_connect)
        # print(point[4])

        # orta noktadan köşelere doğruı olan doğru için kullanılacak 1-left   2-right
        fark_left = point[4][0] - root_connect[0]
        fark_right = root_connect[2] - point[4][0]

        # parmağın başlangıç noktasından başlarak ilerler
        if boolean == 1:
            y = int(point[3][1] + (point[3][1] - point[4][1]) * (growth / 100))
        else:
            y = int(point[2][1] + (point[2][1] - point[4][1]) * (growth / 100))
        # print(y, point[4][1], point[5][1])

        while y >= point[4][1]:
            x = find_X1(denklem_long, y)
            # parmağın başlangıç noktasından başlarak ilerler
            if boolean == 1:
                long_last_y = int(point[4][1] + ((y - point[4][1]) * 100 / (100 + growth)))
                long_last_x = find_X1(denklem_long, long_last_y)
            else:
                long_last_y = int(point[4][1] + ((y - point[4][1]) * 100 / (100 + growth)))
                long_last_x = find_X1(denklem_long, long_last_y)

            long_yedek = long_last_x
            yedek = x

            # sol tarafa bakıyorsa
            b, g, r = img[int(long_last_y)][int(long_last_x)]
            while r > 80:
                # cv2.circle(image, (int(x), int(y_dene)), 1, (255, 0, 255), cv2.FILLED)
                image[int(y)][int(x)] = img[int(long_last_y)][int(long_last_x)]
                # b, g, r = interPlasyon(img, long_last_x, long_last_y)
                b, g, r = bilinearv2(img, long_last_x, long_last_y)
                image[int(y), int(x), 0] = b
                image[int(y), int(x), 1] = g
                image[int(y), int(x), 2] = r
                x = x - 1
                long_last_x = long_last_x - 1
                b, g, r = img[int(long_last_y)][int(long_last_x)]

            long_last_x = long_yedek + 1
            x = yedek + 1
            # sağ tarafa bakıyorsa
            b, g, r = img[int(long_last_y)][int(long_last_x)]
            while r > 80:
                # cv2.circle(image, (int(x), int(y_dene)), 1, (255, 0, 255), cv2.FILLED)
                image[int(y)][int(x)] = img[int(long_last_y)][int(long_last_x)]
                # b, g, r = interPlasyon(img, long_last_x, long_last_y)
                b, g, r = bilinearv2(img, long_last_x, long_last_y)
                image[int(y), int(x), 0] = b
                image[int(y), int(x), 1] = g
                image[int(y), int(x), 2] = r

                x = x + 1
                long_last_x = long_last_x + 1
                b, g, r = img[int(long_last_y)][int(long_last_x)]
            # image[y, int(x + 5)] = (image[y][int(x-1)])
            y = y - 1
        ShowImage(image, Time_count)
        start_finger_try(img, point, growth + count, end, Time_count, count)
        ShowImage(image, Time_count + 0.1)


def start_finger_with_angle(img, point, growth, end, Time_count, count):
    if growth > end:
        pass
        # end of
    else:
        image = img.copy()
        boolean = find_boolean(point)
        # başlangıç noktası ile bitiş noktası arasındaki doğrunun denklemi
        denklem_long = find_linear_denklem(point[4][0], point[4][1], point[5][0], point[5][1])

        # başlangıç noktası ve x'e paralel köşe noktaları bulundu
        root_connect = root_paralel(image, point[4][0], point[4][1])

        # print(root_connect)
        # print(point[4])

        # orta noktadan köşelere doğruı olan doğru için kullanılacak 1-left   2-right
        fark_left = point[4][0] - root_connect[0]
        fark_right = root_connect[2] - point[4][0]

        # parmağın başlangıç noktasından başlarak ilerler
        if boolean == 1:
            y = int(point[3][1] + (point[3][1] - point[4][1]) * (growth / 100))
        else:
            y = int(point[2][1] + (point[2][1] - point[4][1]) * (growth / 100))
        # print(y, point[4][1], point[5][1])

        while y >= point[4][1]:
            x = find_X1(denklem_long, y)
            # parmağın başlangıç noktasından başlarak ilerler
            if boolean == 1:
                long_last_y = int(point[4][1] + ((y - point[4][1]) * 100 / (100 + growth)))
                long_last_x = find_X1(denklem_long, long_last_y)
            else:
                long_last_y = int(point[4][1] + ((y - point[4][1]) * 100 / (100 + growth)))
                long_last_x = find_X1(denklem_long, long_last_y)

            long_yedek = long_last_x
            yedek = x

            # sol tarafa bakıyorsa
            b, g, r = img[int(long_last_y)][int(long_last_x)]
            while r > 80:
                # cv2.circle(image, (int(x), int(y_dene)), 1, (255, 0, 255), cv2.FILLED)
                image[int(y)][int(x)] = img[int(long_last_y)][int(long_last_x)]
                # b, g, r = interPlasyon(img, long_last_x, long_last_y)
                b, g, r = bilinearv2(img, long_last_x, long_last_y, 1)
                image[int(y), int(x), 0] = b
                image[int(y), int(x), 1] = g
                image[int(y), int(x), 2] = r

                x = x - 1
                long_last_x = long_last_x - 1
                b, g, r = img[int(long_last_y)][int(long_last_x)]

            long_last_x = long_yedek + 1
            x = yedek + 1
            # sağ tarafa bakıyorsa
            b, g, r = img[int(long_last_y)][int(long_last_x)]
            while r > 80:
                # cv2.circle(image, (int(x), int(y_dene)), 1, (255, 0, 255), cv2.FILLED)
                image[int(y)][int(x)] = img[int(long_last_y)][int(long_last_x)]
                # b, g, r = interPlasyon(img, long_last_x, long_last_y)
                b, g, r = bilinearv2(img, long_last_x, long_last_y, 1)
                image[int(y), int(x), 0] = b
                image[int(y), int(x), 1] = g
                image[int(y), int(x), 2] = r

                x = x + 1
                long_last_x = long_last_x + 1
                b, g, r = img[int(long_last_y)][int(long_last_x)]
            # image[y, int(x + 5)] = (image[y][int(x-1)])
            y = y - 1
        ShowImage(image, Time_count)
        start_finger_try(img, point, growth + count, end, Time_count, count)
        ShowImage(image, Time_count + 0.1)


def findDistance(x, y, x1, y1):
    return sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1))


def interPlasyon(image, x, y):
    effect = list()
    bgr = list()

    count = 0

    # [0] self , [1] right , [2] left , [3] up , [4] down
    # take distance and b,g,r values
    b, g, r = (image[int(y), int(x)])
    if r > 60:
        effect.append(findDistance(x, y, int(x), int(y)))
        bgr.append([b, g, r])
        count = count + 1
    b, g, r = (image[int(y), int(x + 1)])
    if r > 60:
        effect.append(findDistance(x, y, int(x + 1), int(y)))
        bgr.append([b, g, r])
        count = count + 1
    b, g, r = (image[int(y), int(x - 1)])
    if r > 60:
        effect.append(findDistance(x, y, int(x - 1), int(y)))
        bgr.append([b, g, r])
        count = count + 1
    b, g, r = (image[int(y + 1), int(x)])
    if r > 60:
        effect.append(findDistance(x, y, int(x), int(y + 1)))
        bgr.append([b, g, r])
        count = count + 1
    b, g, r = (image[int(y - 1), int(x)])
    if r > 60:
        effect.append(findDistance(x, y, int(x), int(y - 1)))
        bgr.append([b, g, r])
        count = count + 1

    if count < 2:
        b, g, r = (image[int(y), int(x)])
        return [b, g, r]

    adder = 0
    i = 0
    while i < len(effect):
        adder = adder + effect[i]
        i = i + 1

    i = 0
    while i < len(effect):
        effect[i] = (1 - (effect[i] / adder)) / (count - 1)
        i = i + 1

    b = 0
    g = 0
    r = 0
    i = 0
    while i < len(effect):
        b = b + (bgr[i][0] * effect[i])
        g = g + (bgr[i][1] * effect[i])
        r = r + (bgr[i][2] * effect[i])
        i = i + 1

    return [b, g, r]


def bilinearv2(image, x, y):
    x1 = int(x) - 1
    x2 = int(x) + 1
    y1 = int(y) - 1
    y2 = int(y) + 1
    a = image[y1, x1]
    b = image[y2, x1]
    c = image[y1, x2]
    d = image[y2, x2]

    pixel = 1 / ((x2 - x1) * (y2 - y1))
    pixel1 = a * (x2 - x) * (y2 - y)
    pixel2 = c * (x - x1) * (y2 - y)
    pixel3 = b * (x2 - x) * (y - y1)
    pixel4 = d * (x - x1) * (y - y1)
    pixel_total = pixel * (pixel1 + pixel2 + pixel3 + pixel4)

    b_pi, g, r = pixel_total.astype(np.uint8)

    return [b_pi, g, r]


def bilinear(image, x, y):
    height = image.shape[0]
    width = image.shape[1]
    scale_x = (width) / (x)
    scale_y = (height) / (y)
    temp_x = x
    temp_y = y
    x_int = int(temp_x)
    y_int = int(temp_y)
    x_diff = x - x_int
    y_diff = y - y_int
    a = image[y_int + 1, x_int + 1]
    b = image[y_int - 1, x_int + 1]
    c = image[y_int + 1, x_int - 1]
    d = image[y_int - 1, x_int - 1]

    pixel_b = a * (1 - x_diff) * (1 - y_diff) + b * (x_diff) * \
              (1 - y_diff) + c * (1 - x_diff) * (y_diff) + d * x_diff * y_diff

    b_pi, g, r = pixel_b.astype(np.uint8)

    return [b_pi, g, r]


def start_finger(img, point, growth, end, Time_count):
    if growth > end:
        pass
    else:
        image = img.copy()
        denklem_long = find_linear_denklem(point[4][0], point[4][1], point[5][0], point[5][1])
        # distance from mid point to edges
        fark1_x, fark1_y = (point[4][0] - point[0][0]), (point[4][1] - point[0][1])
        fark2_x, fark2_y = (point[1][0] - point[4][0]), (point[1][1] - point[4][1])
        # finger's start point
        end_x = int(point[5][0] + (point[5][0] - point[4][0]) * (growth / 100))
        end_y = int(point[5][1] + (point[5][1] - point[4][1]) * (growth / 100))
        last_y = point[5][1]
        y = end_y
        while y >= point[4][1]:
            x = find_X1(denklem_long, y)
            denklem_left = find_linear_denklem(x, y, x - fark1_x, y - fark1_y)
            denklem_right = find_linear_denklem(x, y, x + fark2_x, y + fark2_y)
            # pixel copy
            long_last_y = int(point[4][1] + ((y - point[4][1]) * 100 / (100 + growth)))
            long_last_x = int(find_X1(denklem_long, long_last_y))
            long_denklem_left = find_linear_denklem(long_last_x, long_last_y, long_last_x - fark1_x,
                                                    long_last_y - fark1_y)
            long_denklem_right = find_linear_denklem(long_last_x, long_last_y, long_last_x - fark2_x,
                                                     long_last_y - fark2_y)
            long_yedek = long_last_x
            yedek = x
            while x >= yedek - fark1_x:
                y_dene = int(find_Y1(denklem_left, x))
                y_long = int(find_Y1(long_denklem_left, long_last_x))
                b, g, r = interPlasyon(img, long_last_x, y_long)

                image[y_dene, int(x), 0] = b
                image[y_dene, int(x), 1] = g
                image[y_dene, int(x), 2] = r
                if (image[y_dene, int(x - 1), 2] < 60) and (image[y_dene, int(x - 2), 2] > 60):
                    image[y_dene, int(x - 1)] = image[y_dene, int(x - 2)]
                x = x - 1
                long_last_x = long_last_x - 1
            long_last_x = long_yedek + 1
            x = yedek + 1
            while x <= yedek + fark1_x:
                y_dene = int(find_Y1(denklem_right, x))
                y_long = int(find_Y1(long_denklem_right, long_last_x))
                # image[y_dene][int(x)] = img[y_long][int(long_last_x)]
                b, g, r = interPlasyon(img, long_last_x, y_long)
                image[y_dene, int(x), 0] = b
                image[y_dene, int(x), 1] = g
                image[y_dene, int(x), 2] = r
                if (image[y_dene, int(x + 1), 2] < 60) and (image[y_dene, int(x + 2), 2] > 60):
                    image[y_dene, int(x + 1)] = image[y_dene, int(x + 2)]
                x = x + 1
                long_last_x = long_last_x + 1
            y = y - 1
        ShowImage(image, Time_count)
        start_finger(img, point, growth + 5, end, Time_count)
        ShowImage(image, Time_count)


def parmakcekme(finger):
    """print(finger)"""


def checkersıfırlama():
    global checkerfordoc
    checkerfordoc = 0


def parmakgenel(img, lmList):
    _, x1, y1 = lmList[8]
    _, x2, y2 = lmList[4]
    yon = 1
    if (x1 > x2):
        tempx1 = x2
        x2 = x1
        x1 = tempx1
        tempy1 = y2
        y2 = y1
        y1 = tempy1
        yon = 2
        print("girdi")

    denklem = find_linear_denklem(x1, y1, x2, y2)
    i = int(x1)
    flag = -2
    finger = -1
    while (i < x2):
        y = find_Y1(denklem, i)
        b, g, r = (img[int(y), int(i)])
        if r < 80:
            flag = -1
        if i == lmList[25][1]:
            finger = 0
        if i == lmList[29][1]:
            finger = 1
        if i == lmList[33][1]:
            finger = 2
        if i == lmList[37][1]:
            finger = 3
        if i <= lmList[41][1] + 5 and i >= lmList[41][1] - 5:
            finger = 4
        i = i + 1
    return flag, finger, yon


def doktor(img, lmList):
    flag, finger, yon = parmakgenel(img, lmList)
    if flag == -2:
        # cv2.circle(img, (lmList[24+finger*4][1], lmList[24+finger*4][2]), 2, (255, 0, 255), cv2.FILLED)
        parmakcekme(finger)
    return img, flag, finger, yon


def start_mask(img, lmList, finger, yon):
    _, x1, y1 = lmList[24 + finger * 4]
    xL = x1
    xR = x1
    b, g, r = (img[int(y1), int(xL)])
    array = list()
    alt2 = 460
    alt1 = 165

    while (r > 80):
        xL = xL - 1
        b, g, r = (img[int(y1), int(xL)])
    temp = 0
    array.append([xL - 5, y1])

    while (r < 80) and (temp < 50):
        xL = xL - 1
        b, g, r = (img[int(y1), int(xL - 10)])
        temp = temp + 1
    b, g, r = (img[int(y1), int(xR)])
    while (r > 80):
        xR = xR + 1
        b, g, r = (img[int(y1), int(xR)])
    temp = 0
    array.append([xR, y1])
    while (r < 80) and (temp < 50):
        xR = xR + 1
        b, g, r = (img[int(y1), int(xR + 10)])
        temp = temp + 1
    ust1 = xL
    ust2 = xR
    if yon == 1:
        if (finger == 4):
            b1, g1, r1 = img[int(400), int(430)]
            i = 430
            while (r1 < 80):
                i = i - 1
                b1, g1, r1 = img[int(400), int(i)]
            alt2 = i + 5
            alt1 = 5
            ust1 = 0
        elif finger == 0:
            b1, g1, r1 = img[int(400), int(170)]
            i = 170
            while (r1 < 80):
                i = i + 1
                b1, g1, r1 = img[int(400), int(i)]
            alt1 = i - 25
            alt2 = img.shape[1] - 5
            ust2 = img.shape[1] - 1

    elif yon == 2:
        if (finger == 0):
            b1, g1, r1 = img[int(400), int(430)]
            i = 430
            while (r1 < 80):
                i = i - 1
                b1, g1, r1 = img[int(400), int(i)]
            alt2 = i + 5
            alt1 = 5
            ust1 = 0
        elif finger == 4:
            b1, g1, r1 = img[int(400), int(170)]
            i = 170
            while (r1 < 80):
                i = i + 1
                b1, g1, r1 = img[int(400), int(i)]
            alt1 = i - 25
            alt2 = img.shape[1] - 5
            ust2 = img.shape[1] - 1
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(ust1, y1), (ust2, y1), (alt2, img.shape[0] - 2), (alt1, img.shape[0] - 2)]],
                           dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count

    ignore_mask_color_2 = (0,) * channel_count

    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    img_2 = img.copy()
    cv2.fillPoly(img_2, roi_corners, ignore_mask_color_2)
    # apply the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, img_2, xL, xR, y1, array, alt1, alt2


def start_maskv2(img, lmList, finger):
    _, x1, y1 = lmList[24 + finger * 4]
    xL = x1
    xR = x1
    b, g, r = (img[int(y1), int(xL)])
    array = list()

    while (r > 80):
        xL = xL - 1
        b, g, r = (img[int(y1), int(xL)])
    temp = 0
    array.append([xL - 5, y1])

    while (r < 80) and (temp < 50):
        xL = xL - 1
        b, g, r = (img[int(y1), int(xL - 10)])
        temp = temp + 1
    b, g, r = (img[int(y1), int(xR)])
    while (r > 80):
        xR = xR + 1
        b, g, r = (img[int(y1), int(xR)])
    temp = 0
    array.append([xR, y1])
    while (r < 80) and (temp < 50):
        xR = xR + 1
        b, g, r = (img[int(y1), int(xR + 10)])
        temp = temp + 1

    mask = np.zeros(img.shape, dtype=np.uint8)

    roi_corners = np.array([[(xL, y1), (xR, y1), (460, img.shape[0] - 2), (165, img.shape[0] - 2)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count

    ignore_mask_color_2 = (0,) * channel_count

    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    img_2 = img.copy()
    cv2.fillPoly(img_2, roi_corners, ignore_mask_color_2)
    # apply the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, img_2, xL, xR, y1, array


def image_shift(masked_image, croped_image, shift_count):
    M = np.float32([[1, 0, 0], [0, 1, shift_count]])
    dst = cv2.warpAffine(masked_image, M, (masked_image.shape[1], masked_image.shape[0]))
    return dst


def image_shiftangle(masked_image, croped_image, shift_count, x):
    M = np.float32([[1, 0, int(x * shift_count)], [0, 1, shift_count]])
    dst = cv2.warpAffine(masked_image, M, (masked_image.shape[1], masked_image.shape[0]))
    return dst


def shiftfillangle(image, masked_image, croped_image, sol_kose, sag_kose, y, count, array, alt1, alt2, finger,yon):
    global checkerfordoc
    if checkerfordoc > 0:
        pass
    else:
        rotatevalue = 0.0
        if array[0][0] < array[2][0]:
            rotatevalue = float((array[2][0] - array[0][0]) / (array[0][1] - array[2][1])) * -1
        elif array[0][0] > array[2][0]:
            rotatevalue = float((array[0][0] - array[2][0]) / (array[0][1] - array[2][1]))
        elif array[0][0] == array[2][0]:
            rotatevalue = 0
        if (rotatevalue <= 0.2 and rotatevalue >= -0.2) and not (finger == 4 or finger == 0):
            shiftfill(image, masked_image, croped_image, sol_kose, sag_kose, y, count)
            return
        if yon ==1:
            if (finger == 4):
                alt1 = 80
            elif (finger == 0):
                alt1 = alt1 - 100
                alt2 = image.shape[1] - 2
        elif yon==2:
            if (finger == 0):
                alt1 = 80
            elif (finger == 4):
                alt1 = alt1 - 100
                alt2 = image.shape[1] - 2
        shift_image = image_shiftangle(masked_image, croped_image, count, rotatevalue)
        img = croped_image.copy()
        i = 1
        old_solkose = sol_kose
        old_sagkose = sag_kose
        while (i <= count):
            j = old_solkose + int(i * rotatevalue)
            while (j <= int(old_sagkose + (i * rotatevalue))):
                img[int(y + i - 1), int(j + int(i * rotatevalue))] = croped_image[int(y - 1), int(j)]
                j = j + 1
            i = i + 1
        temp_image = img + shift_image
        i = y - 4
        while (i <= image.shape[0] - 1):
            j = alt1 - 1
            while (j <= alt2 + 1):
                b, g, r = (temp_image[int(i), int(j)])
                if b == 0 and g == 0 and r == 0:
                    temp_image[int(i), int(j)] = temp_image[int(i - 2), int(j)]
                j = j + 1
            i = i + 1
        ShowImagev2(temp_image, 0.2, "image" + str(y))
        shiftfillangle(image, masked_image, croped_image, old_solkose, old_sagkose, y, count + 1, array, alt1, alt2,
                       finger,yon)
        ShowImagev2(temp_image, 0.2, "image" + str(y))
    checkerfordoc = 0


def shiftfill(image, masked_image, croped_image, sol_kose, sag_kose, y, count):
    global checkerfordoc
    if checkerfordoc > 0:
        pass
    else:
        shift_image = image_shift(masked_image, croped_image, count)
        img = croped_image.copy()
        i = 1
        while (i <= count):
            j = sol_kose
            while (j <= sag_kose):
                img[int(y + i - 1), int(j)] = croped_image[int(y - 1), int(j)]
                j = j + 1
            i = i + 1
        temp_image = img + shift_image
        i = y - 2
        while (i <= image.shape[0] - 1):
            j = 150
            while (j <= 470):
                b, g, r = (temp_image[int(i), int(j)])
                if b == 0 and g == 0 and r == 0:
                    temp_image[int(i), int(j)] = temp_image[int(i), int(j - 4)]
                j = j + 1
            i = i + 1

        ShowImagev2(temp_image, 0.2, "image" + str(y))
        shiftfill(image, masked_image, croped_image, sol_kose, sag_kose, y, count + 1)
        ShowImagev2(temp_image, 0.2, "image" + str(y))
    checkerfordoc = 0


def düzparmak():
    print("düzparmak")


def lenghtline(image, count, hands, mpDraw, Present, Time_count, Time_end):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    HandLandmark = find_Landmark(image, results, mpDraw)
    FingerStart = connectFinger(HandLandmark, image)
    FingerTips = fingertips(image, HandLandmark, FingerStart)
    x, y = FingerStart[count]
    x, y = int(x), int(y)
    end_x, end_y = FingerTips[count]
    tempcount = count
    end_x, end_y = int(end_x), int(end_y)
    plus_x, plus_y = inc(x, y, end_x, end_y)
    x_A, y_A = give_point_root(image, x, y, plus_x, plus_y, 0)
    x_B, y_B = give_point_root(image, x, y, plus_x, plus_y, 1)
    x_C, y_C = give_point_root_up(x, y, x_A, y_A, end_x, end_y, 0)
    x_D, y_D = give_point_root_up(x, y, x_B, y_B, end_x, end_y, 1)

    point = list()
    point.append([x_A, y_A])
    point.append([x_B, y_B])
    point.append([x_C, y_C])
    point.append([x_D, y_D])
    point.append([x, y])
    point.append([end_x, end_y])
    count = Present / (Time_end / Time_count)

    temp_x = x - end_x

    if abs(temp_x) >= 30 and (tempcount == 0 or tempcount == 4):
        """cv2.circle(image, (point[2][0], point[2][1]), 2, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (point[3][0], point[3][1]), 2, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (point[5][0], point[5][1]), 2, (255, 0, 255), cv2.FILLED)
        cv2.imshow("sfg", image)
        time.sleep(3)"""
        düzparmak()
        print(temp_x)
        boolean = find_boolean(point)
        if boolean == 1:
            point[2][1] = point[2][1] + ((point[2][1] - point[0][1]) * 0.2)
            point[2][0] = point[2][0] + ((point[2][0] - point[0][0]) * 0.2)
            point[3][1] = point[3][1] + ((point[3][1] - point[1][1]) * 0.2)
            point[3][0] = point[3][0] + ((point[3][0] - point[1][0]) * 0.2)
        else:
            point[3][1] = point[3][1] + ((point[3][1] - point[1][1]) * 0.2)
            point[3][0] = point[3][0] + ((point[3][0] - point[1][0]) * 0.2)
            point[2][1] = point[2][1] + ((point[2][1] - point[0][1]) * 0.2)
            point[2][0] = point[2][0] + ((point[2][0] - point[0][0]) * 0.2)
        print("angle")
        """if count==0:
            point[2][1]=point[3][1]+8
        elif count==4:
            point[3][1] = point[2][1]+8"""
        start_finger_try(image, point, count, Present, Time_count, count)
    else:
        print(temp_x)
        print("normal")
        start_finger_try(image, point, count, Present, Time_count, count)


def forchecker(cap, check, start, end, count, detector):
    global checkerfordoc
    checkerfordoc = 0
    while True:
        img = cap.read()[1]
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, flag, finger, lmList,yon = detector.findDoktor(img, check, start, end, count)
        if flag == -1:
            checkerfordoc = 1
            exit()


def take_mask(image, point):
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[point[0], point[1], point[3], point[2]]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    x_A, y_A = point[0]
    x_B, y_B = point[1]
    x_C, y_C = point[2]
    x_D, y_D = point[3]

    if y_A < y_B:  # sola bakan
        # masks = masked_image[y_A - 50:y_D + 50, x_C - 50:x_B + 50]
        # return masks, 1
        return masked_image, 1
    else:  # sağa bakar
        # masks = masked_image[y_B - 50:y_C + 50, x_A - 50:x_D + 50]
        # return masks, 2
        return masked_image, 2

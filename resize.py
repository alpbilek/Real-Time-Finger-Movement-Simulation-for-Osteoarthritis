import cv2
import time
from FingerTips import fingertips
def ShowImage(image, count):
    start = time.time()
    end = time.time()
    while end - start < count:
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        end = time.time()
def TouchChecker(FingerTips):
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
def Inc(x, y, x1, y1):
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
        if r < 100:
            return [a, s]
        x_r = x_r + x1
        y_r = y_r + y1
        stop = stop + 1
        if stop == 200:
            return [x, y]

def give_point_root_up(x, y, x1, y1, x2, y2):
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
    egim = (y1 -y) / (x1 - x)
    return egim


def find_X1(denklem, y1):
    X1 = denklem[0] - ((denklem[1] - y1) / denklem[2])
    return X1

def find_equation(x, y, x1, y1):
    m = egim_bulma(x, y, x1, y1)
    denklem = [x, y, m]
    return denklem

def start_finger(img, point, growth, end, Time_count):
    if growth > end:
        pass
    else:
        image = img.copy()
        denklem_long = find_equation(point[4][0], point[4][1], point[5][0], point[5][1])

        fark1_x, fark1_y = (point[4][0] - point[0][0]), (point[4][1] - point[0][1])
        fark2_x, fark2_y = (point[1][0] - point[4][0]), (point[1][1] - point[4][1])

        end_x = int(point[5][0] + (point[5][0] - point[4][0]) * (growth / 100))
        end_y = int(point[5][1] + (point[5][1] - point[4][1]) * (growth / 100))
        last_y = point[5][1]
        y = end_y
        while y >= point[4][1]:
            x = find_X1(denklem_long, y)
            denklem_left = find_equation(x, y, x - fark1_x, y - fark1_y)
            denklem_right = find_equation(x, y, x + fark2_x, y + fark2_y)
            long_last_y = int(point[4][1] + ((y - point[4][1]) * 100 / (100 + growth)))
            long_last_x = int(find_X1(denklem_long, long_last_y))
            long_denklem_left = find_equation(long_last_x, long_last_y, long_last_x - fark1_x, long_last_y - fark1_y)
            long_denklem_right = find_equation(long_last_x, long_last_y, long_last_x - fark2_x, long_last_y - fark2_y)
            long_yedek = long_last_x
            yedek = x
            while x >= yedek - fark1_x:
                y_dene = int(find_Y1(denklem_left, x))
                y_long = int(find_Y1(long_denklem_left, long_last_x))
                image[y_dene][int(x)] = img[y_long][int(long_last_x)]
                x = x - 1
                long_last_x = long_last_x - 1
            long_last_x = long_yedek + 1
            x = yedek + 1
            while x <= yedek + fark1_x:
                y_dene = int(find_Y1(denklem_right, x))
                y_long = int(find_Y1(long_denklem_right, long_last_x))
                image[y_dene][int(x)] = img[y_long][int(long_last_x)]
                image[y_dene, int(x), 0] = img[y_long, int(long_last_x), 0]
                image[y_dene, int(x), 1] = img[y_long, int(long_last_x), 1]
                image[y_dene, int(x), 2] = img[y_long, int(long_last_x), 2]
                x = x + 1
                long_last_x = long_last_x + 1
            y = y - 1
        ShowImage(image, Time_count)
        start_finger(img, point, growth + 5, end, Time_count)
        ShowImage(image, Time_count)


def lenghtline(image, count, hands, mpDraw, Present, Time_count, Time_end):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    HandLandmark = find_Landmark(image, results, mpDraw)
    FingerStart = connectFinger(HandLandmark, image)
    FingerTips = fingertips(image, HandLandmark, FingerStart)
    x, y = FingerStart[count]
    x, y = int(x), int(y)
    end_x, end_y = FingerTips[count]
    end_x, end_y = int(end_x), int(end_y)
    plus_x, plus_y = Inc(x, y, end_x, end_y)
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
    start_finger(image, point, 5, 40, 1)

from math import sqrt
def fingertips(image, HandLandmark, FingerStart):
    hand = list()
    if len(HandLandmark) == 21:  # only
        hand = fingertipsOne(image, HandLandmark, FingerStart)
        return hand
    if len(HandLandmark) == 42:
        hand = fingertipsTwo(image, HandLandmark, FingerStart)
        return hand
    else:
        return hand
def fingertipsTwo(image, HandLandmark, FingerStart):
    Two_Hand = list()
    x, y = FingerStart[0]
    _, x25, y25 = HandLandmark[25]
    Two_Hand.append(findPoint(image, x25, y25, x, y, 1))
    x, y = FingerStart[1]
    _, x29, y29 = HandLandmark[29]
    Two_Hand.append(findPoint(image, x29, y29, x, y, 1))
    x, y = FingerStart[2]
    _, x33, y33 = HandLandmark[33]
    Two_Hand.append(findPoint(image, x33, y33, x, y, 1))
    x, y = FingerStart[3]
    _, x37, y37 = HandLandmark[37]
    Two_Hand.append(findPoint(image, x37, y37, x, y, 1))
    x, y = FingerStart[4]
    _, x41, y41 = HandLandmark[41]
    Two_Hand.append(findPoint(image, x41, y41, x, y, 1))
    _, x, y = HandLandmark[3]
    _, x4, y4 = HandLandmark[4]
    Two_Hand.append(findPoint(image, x4, y4, x, y, 2))
    _, x, y = HandLandmark[7]
    _, x8, y8 = HandLandmark[8]
    Two_Hand.append(findPoint(image, x8, y8, x, y, 2))
    _, x, y = HandLandmark[11]
    _, x12, y12 = HandLandmark[12]
    Two_Hand.append(findPoint(image, x12, y12, x, y, 2))
    _, x, y = HandLandmark[15]
    _, x16, y16 = HandLandmark[16]
    Two_Hand.append(findPoint(image, x16, y16, x, y, 2))
    _, x, y = HandLandmark[19]
    _, x20, y20 = HandLandmark[20]
    Two_Hand.append(findPoint(image, x20, y20, x, y, 2))
    return Two_Hand

def fingertipsOne(image, HandLandmark, FingerStart):
    One_Hand = list()
    x, y = FingerStart[0]
    _, x25, y25 = HandLandmark[4]
    One_Hand.append(findPoint(image, x25, y25, x, y, 1))
    x, y = FingerStart[1]
    _, x29, y29 = HandLandmark[8]
    One_Hand.append(findPoint(image, x29, y29, x, y, 1))
    x, y = FingerStart[2]
    _, x33, y33 = HandLandmark[12]
    One_Hand.append(findPoint(image, x33, y33, x, y, 1))
    x, y = FingerStart[3]
    _, x37, y37 = HandLandmark[16]
    One_Hand.append(findPoint(image, x37, y37, x, y, 1))
    x, y = FingerStart[4]
    _, x41, y41 = HandLandmark[20]
    One_Hand.append(findPoint(image, x41, y41, x, y, 1))
    return One_Hand

def findPoint(image, x, y, x1, y1, position):
    if position == 1:
        if y - y1 == 0 and x < x1:
            return x - 10, y
        elif y - y1 == 0 and x > x1:
            return x + 10, y
        elif x - x1 == 0:

            return x, y + 10
        else:
            plus_x, plus_y = 1,2
            check = True
            count = 0
            while check:
                x = x + plus_x
                y = y + plus_y
                b, g, r = (image[int(y), int(x)])
                if r < 100:
                    check = False
                count += 1
                if count > 10:
                    return x, y + 10
                last_x = int(x)
                last_y = int(y)

            return x, y

    else:
        if y - y1 == 0 and x < x1:

            return x - 6, y
        elif y - y1 == 0 and x > x1:

            return x + 6, y
        elif x - x1 == 0:

            return x, y - 6
        elif x - x1 < 0:
            long = int(sqrt(((x - x1) ** 2) + ((y - y1) ** 2)))
            newLong = long + 6
            newX = int(x1 - ((x1 - x) * newLong / long))
            newY = int(y1 - ((y1 - y) * newLong / long))

            return newX, newY
        elif x - x1 > 0:  # x - x1 > 0
            long = int(sqrt(((x - x1) ** 2) + ((y1 - y) ** 2)))
            newLong = long + 6
            newX = int(((x - x1) * newLong / long) + x1)
            newY = int(y1 - ((y1 - y) * newLong / long))

            return newX, newY
        else:
            return x, y - 6


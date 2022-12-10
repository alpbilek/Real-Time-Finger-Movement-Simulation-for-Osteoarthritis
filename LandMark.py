import cv2  # opencv


def find_Landmark(image, results, mpDraw):
    Point_Landmark = list()  # handlandmark point keep in there
    index = 0
    if results.multi_hand_landmarks:  # if we see a hand
        for handLms in results.multi_hand_landmarks:  # find all hand . For each hand landmark
            if index == 0:
                for id, lm in enumerate(handLms.landmark):  # landmark x,y,z cordinat     id between 0-20
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # landmark point cordinat
                    Point_Landmark.append([id, cx, cy])
                    index = 1
            else:
                for id, lm in enumerate(handLms.landmark):  # landmark x,y,z cordinat     id between 0-20
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # landmark point cordinat
                    Point_Landmark.append([id + 21, cx, cy])
            #mpDraw.draw_landmarks(image, handLms)  # all handmark show
            # mpDraw.draw_landmarks(image, handLms ,mpHands.HAND_CONNECTIONS)  # all handmark show with connections
    if len(Point_Landmark) == 42:
        _, _, y1 = Point_Landmark[0]
        _, _, y2 = Point_Landmark[21]
        if y1 < y2:
            newLandmark = list()
            i = 0
            while i < 21:
                newLandmark.append(Point_Landmark[i + 21])
                i = i + 1
            i = 0
            while i < 21:
                newLandmark.append(Point_Landmark[i])
                i = i + 1
            return newLandmark
        else:
            pass
    return Point_Landmark


# there we show connected point
def between(image, connect):
    cv2.circle(image, (int((connect[0] + connect[2]) / 2), int((connect[1] + connect[3]) / 2)), 2, (255, 0, 255),
               cv2.FILLED)
    cv2.circle(image, (int((connect[2] + connect[4]) / 2), int((connect[3] + connect[5]) / 2)), 2, (255, 0, 255),
               cv2.FILLED)
    cv2.circle(image, (int((connect[4] + connect[6]) / 2), int((connect[5] + connect[7]) / 2)), 2, (255, 0, 255),
               cv2.FILLED)
    cv2.circle(image, (int((connect[6] + connect[8]) / 2), int((connect[7] + connect[9]) / 2)), 2, (255, 0, 255),
               cv2.FILLED)


# hasta parmağının başlangıç kordinatları bulunur .
def connectFinger(LandMarks, image):
    connect = list()
    if len(LandMarks) > 0:
        i = 0
        if len(LandMarks) == 21:
            size = 2
        else:
            size = 23
        while i < 5:
            _, x, y = LandMarks[size]
            _, x1, y1 = LandMarks[size + 1]
            x2, y2 = int((x + (2 * x1)) / 3), int((y + (2 * y1)) / 3)
            plus_x, plus_y = increase(x, y, x1, y1)  # artış miktarları bulundu
            x_l, y_l = give_point_root(image, x2, y2, plus_x, plus_y, 0)
            x_r, y_r = give_point_root(image, x2, y2, plus_x, plus_y, 1)
            x2, y2 = int((x_l + x_r) / 2), int((y_l + y_r) / 2)
            if i == 0:
                size = size + 3
            else:
                size = size + 4
            # cv2.circle(image, (x2, y2), 2, (255, 0, 255), cv2.FILLED)
            connect.append([x2, y2])
            i = i + 1
        return connect
    else:  # size == 0
        return connect


# x,y kök noktası ; x1,y1 artış miktarı ; direction hangi tarafı
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
        if count >= 2 and r < 100:
            x_r = int(x_r)
            y_r = int(y_r)
            return [x_r, y_r]
        if r < 100:
            count = count + 1
        x_r = x_r + x1
        y_r = y_r + y1
        stop = stop + 1
        if stop == 40:
            return [x, y]


# o parmakta noktaların kayma mislini verir , açıya göre
def increase(x, y, x1, y1):
    # artış miktarı    x,y = parmak başlangıç yeri     x1, y1 = en uç nokta
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
        return check_increase(r_x, r_y)
    else:
        if x2 > y2:
            r_x = -1
            r_y = x2 / y2
        else:
            r_x = -1 * (y2 / x2)
            r_y = 1
        return check_increase(r_x, r_y)


# o parmakta noktaların kayma mislini verir , ama dik açı değik aynı doğrultuda
def increase_for_same(x, y, x1, y1):
    # artış miktarı    x,y = parmak başlangıç yeri     x1, y1 = en uç nokta
    x2 = x - x1
    y2 = y - y1
    return check_increase(x2, y2)


# bir sayının çok büyük yada küçük olmasına kontrol
def check_increase(plus_x, plus_y):
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
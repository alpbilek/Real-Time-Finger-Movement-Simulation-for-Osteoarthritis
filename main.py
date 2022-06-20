
from tkinter import ttk
import cv2
import mediapipe as mp
import time
import FingerTips
import resize
from tkinter import *
fingerTips = FingerTips
line = resize
root = Tk()
f1 = LabelFrame(root, bg="red")
Label(root, text="Real Time Finger \nMovement Simulation for \nOsteoarthritis", font=("times new roman", 20, "bold"), bg="white", fg="red").pack()
vlist = ["Thumb", "Index Finger", "Middle Finger",
             "Ring Finger", "Little Finger"]
Combo = ttk.Combobox(f1, values=vlist)
my_text=Text(root,width=4,height=1)
my_text.pack(padx=19)


def FingerSelection():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    count = Combo.current()
    while True:
        img = cap.read()[1]
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector.findhandsHardcoded(img, count)
        break
    cap.release()

def DoctorTouch():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    check = True
    start = 0
    end = 0
    count = 0
    while True:
        img = cap.read()[1]
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Video", img)
        img, check, start, end, count, checker = detector.findHands(img, check, start, end, count)
        if checker == 1:
            break
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_Landmark(image, results):
        Point_Landmark = list()
        index = 0
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if index == 0:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        Point_Landmark.append([id, cx, cy])
                        index = 1
                else:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        Point_Landmark.append([id + 21, cx, cy])

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
    def findhandsHardcoded(self, img, count, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        lmList = self.find_Landmark(img, self.results, self.mpDraw)
        if len(lmList) > 0:
            FingerStart = self.connectFinger(lmList, img)
            Finger_tips = fingerTips.fingertips(img, lmList, FingerStart)
        img = line.lenghtline(img, count, self.mpHands.Hands(), self.mpHands, 20, 0.3, 3)
        return img

    def findHands(self, img, check, start, end, count, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        checker=0
        self.results = self.hands.process(imgRGB)
        lmList = self.find_Landmark(img, self.results, self.mpDraw)
        if len(lmList) > 0:
            FingerStart = self.connectFinger(lmList, img)
            Finger_tips = fingerTips.fingertips(img, lmList, FingerStart)

            if check:
                if len(lmList) == 42:
                    print("The doctor hand detected")
                    Finger_tips = fingerTips.fingertips(img, lmList, FingerStart)
                    count = line.TouchChecker(Finger_tips)
                    if count >= 0:
                        start = time.time()
                        print("The doctor must withdraw hand")
                        print(count)
                        end = time.time()
                        check = False

            else:
                end = time.time()

                if end - start > 3:
                    line.lenghtline(img, count, self.mpHands.Hands(), self.mpHands, 20, 0.3, 3)
                    check = True
                    checker=1

        return img, check, start, end, count,checker

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])

        return lmlist


def main():

    root.geometry("480x480")
    root.configure(bg="white")



    f1.pack()



    Combo.set("Select a Finger")

    Combo.pack(padx=5, pady=5)
    btn1 = Button(root, text="Doctor Touch", font=("times new roman", 20, "bold"), bg="black", fg="red",
                  command=DoctorTouch)
    btn1.place(x=5, y=270)
    button3 = Button(root, text="Finger Selection", font=("times new roman", 20, "bold"), bg="black", fg="red",
                     command=FingerSelection)
    button3.place(x=5, y=200)
    while True:
        root.update()


if __name__ == "__main__":
    main()

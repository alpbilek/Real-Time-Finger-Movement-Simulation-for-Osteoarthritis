import threading
from tkinter import ttk
import cv2
import mediapipe as mp
import time
import LandMark
import FingerTips
import linebyline
from tkinter import *
countfoo=0
landMark = LandMark
fingerTips = FingerTips

line = linebyline
root = Tk()
f1 = LabelFrame(root, bg="red")
Label(root, text="Real Time Finger \nMovement Simulation for \nOsteoarthritis", font=("times new roman", 20, "bold"), bg="white", fg="red").pack()
vlist = ["Thumb", "Index Finger", "Middle Finger",
             "Ring Finger", "Little Finger"]
Combo = ttk.Combobox(f1, values=vlist)
a=Entry(root, width=35)
a.pack()

def forThread(img,lmList,finger,yon):
    masked_image, croped_image, xL, xR, y,array,alt1,alt2 = line.start_mask(img, lmList, finger,yon)
    if finger==0:
        _, xt, yt = lmList[22 + (finger * 4)]
        array.append([xt-15, yt])
    else:
        _,x,y1=lmList[23+(finger*4)]
        b, g, r = (img[int(y), int(x)])
        while (r > 80):
            x = x - 1
            b, g, r = (img[int(y1), int(x)])
        x=x-5
        array.append([x,y1])

    line.shiftfillangle(img, masked_image, croped_image, xL, xR, y, 1,array,alt1,alt2,finger,yon)


def DoctorCekim():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    check = True
    start = 0
    end = 0
    count = 0
    while True:
        img = cap.read()[1]
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img,flag,finger,lmList,yon = detector.findDoktor(img, check, start, end, count)
        if flag == -2 and finger>=0:
            t1=threading.Thread(target=forThread, args=(img,lmList,finger,yon,))
            t2=threading.Thread(target=line.forchecker,args=(cap,check, start, end, count,detector,))
            t1.start()
            t2.start()

            t1.join()
            t2.join()
            line.checkersıfırlama()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()

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

    def findhandsHardcoded(self, img, count, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        lmList = landMark.find_Landmark(img, self.results, self.mpDraw)
        if len(lmList) > 0:
            FingerStart = landMark.connectFinger(lmList, img)
            Finger_tips = fingerTips.fingertips(img, lmList, FingerStart)
        img = line.lenghtline(img, count, self.mpHands.Hands(), self.mpHands, int(a.get()), 0.3, 3)
        return img

    def findHands(self, img, check, start, end, count, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        checker=0
        self.results = self.hands.process(imgRGB)
        lmList = landMark.find_Landmark(img, self.results, self.mpDraw)
        if len(lmList) > 0:
            FingerStart = landMark.connectFinger(lmList, img)
            Finger_tips = fingerTips.fingertips(img, lmList, FingerStart)

            if check:
                if len(lmList) == 42:
                    print("The doctor hand detected")
                    Finger_tips = fingerTips.fingertips(img, lmList, FingerStart)
                    count = line.DoctorChecker(Finger_tips)
                    if count >= 0:
                        start = time.time()
                        print("The doctor must withdraw hand")
                        print(count)
                        end = time.time()
                        check = False

            else:
                end = time.time()
                if end - start > 3:
                    line.lenghtline(img, count, self.mpHands.Hands(), self.mpHands, int(a.get()), 0.3, 3)
                    check = True
                    checker=1

        return img, check, start, end, count,checker
    def finddoc(self,img ,check,start,end,count,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        checker = 0
        self.results = self.hands.process(imgRGB)
        lmList = landMark.find_Landmark(img, self.results, self.mpDraw)
        return lmList
    def findDoktor(self,img ,check,start,end,count,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        checker = 0
        self.results = self.hands.process(imgRGB)
        lmList = landMark.find_Landmark(img, self.results, self.mpDraw)
        if len(lmList) != 42:
            return img,-1,-1,lmList,0
        else :
            img,flag,finger,yon=line.doktor(img,lmList)
            return img,flag,finger,lmList,yon

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
    button3 = Button(root, text="Doktor Çekim", font=("times new roman", 20, "bold"), bg="black", fg="red",
                     command=DoctorCekim)
    button3.place(x=5, y=340)
    while True:
        root.update()



if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import time
import numpy as np
import osascript

cap = cv2.VideoCapture(0)
cVol = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTIme = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    List = [0,0,0,0]
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx ,cy = int(lm.x*w), int(lm.y*h)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (200,100,100), cv2.FILLED)
                    List[0], List[1] = cx,cy
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (200,100,100), cv2.FILLED)
                    List[2], List[3] = cx,cy

            cv2.line(img, (List[0],List[1]), (List[2],List[3]), (255,0,0),1)

            cVol = ((List[0] - List[2])**2 + (List[1] - List[3])**2)**(1/2) / 3.3

            osascript.osascript("set volume output volume " + str(cVol))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            

    cTIme = time.time()
    fps = 1/(cTIme - pTime)
    pTime = cTIme

    cv2.putText(img, "Fps : " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN , 3 , (0,0,0) , 3)
    cv2.putText(img, "Volume : " + str(int(cVol)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 3 ,(0,0,0), 4)

    cv2.imshow("Camera", img)
    cv2.waitKey(1)
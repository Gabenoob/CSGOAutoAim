from ultralytics import YOLO
from PIL import ImageGrab
import cv2
import pyautogui
import torch
import threading
import numpy
import time


model = YOLO("best.pt")
# result = model.predict("test2.png")
pyautogui.moveTo(0,0)

while 1:
    image = ImageGrab.grab()
    result = model.predict(image)
    if len(result==0):
        continue
    else:
        box = result[0].boxes.xyxy[0].numpy()
        print(box)
        target = ((box[0]+box[2])/2,abs(box[1]-box[3])*0.10+box[1])
        pyautogui.mouseInfo()
        # TODO move mouse to the head
        # TODO measure the distance
        pyautogui.click()
        result.clear()


time.sleep(10)
img = cv2.imread("test2.png")
img[int(target[1])][int(target[0])] = [255,255,255]
img = cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
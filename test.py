import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
handDetector = HandDetector(maxHands=1)  # Initialize hand detector
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  # Initialize classifier

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = handDetector.findHands(img)  # Find hands in the frame
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get the bounding box coordinates of the hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white canvas
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand region

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            newW = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (newW, imgSize))  # Resize the cropped image while maintaining aspect ratio
            wGap = math.ceil((imgSize - newW) / 2)
            imgWhite[:, wGap:newW + wGap] = imgResize
            predictedLabel, predictedIndex = classifier.getPrediction(imgWhite, draw=False)  # Get the predicted label and index
            print(labels[predictedIndex])

        else:
            k = imgSize / w
            newH = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, newH))  # Resize the cropped image while maintaining aspect ratio
            hGap = math.ceil((imgSize - newH) / 2)
            imgWhite[hGap:newH + hGap, :] = imgResize
            predictedLabel, predictedIndex = classifier.getPrediction(imgWhite, draw=False)  # Get the predicted label and index
            print(labels[predictedIndex])

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (0, 255, 0), cv2.FILLED)  # Draw a rectangle for displaying the predicted label
        cv2.putText(imgOutput, labels[predictedIndex], (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)  # Put the predicted label text
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)  # Draw a rectangle around the hand region

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("h"):
        break

cv2.destroyAllWindows()
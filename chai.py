import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time

# Initialize webcam capture
cap = cv2.VideoCapture(0)
# Initialize hand detector with a maximum of 1 hand
detector = HandDetector(maxHands=1)
# Offset for cropping
offset = 10
# Size of the white image
imgSize = 300
counter = 0
imgWhite=None

Folder = "data/A"

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        # Get the bounding box of the first hand
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # Create a white image of size imgSize x imgSize
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the cropping coordinates are within the image bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Crop the image using the adjusted coordinates
        imgCrop = img[y1:y2, x1:x2]

        # Get the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Resize and center the cropped image in imgWhite
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Display the cropped and white images
        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    # Display the original image
    cv2.imshow("Image", img)
    key =cv2.waitKey(1)
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f'{Folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)


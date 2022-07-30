# import the necessary packages
# https://github.com/DEBANJANAB/Face-Blur-and-Anonymization-using-OpenCV/blob/master/blur_techniques/face_blurring.py

import numpy as np
import time
import cv2

# defining prototext and caffemodel paths
caffeModel = "models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "models/deploy.prototxt.txt"

# Load Model
net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

# initialize the video stream to get the video frames
print("starting video stream...")
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("data/video/college.mp4")
time.sleep(2.0)

def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)

    # return the pixelated blurred image
    return image


# loop the frams from the  VideoStream
while True:
    # Get the frams from the video stream and resize to 400 px
    _, frame = cap.read()
    # frame = imutils.resize(frame,width=400)

    warn = "press q to exit"
    cv2.putText(frame, warn, (640, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    (h, w) = frame.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # passing blob through the network to detect and pridiction
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence and prediction
        confidence = detections[0, 0, i, 2]
        # filter detections by confidence greater than the minimum confidence
        if confidence < 0.5:
            continue

        # Determine the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # print(startX, startY, endX, endY)

        # print(confidence)
        # draw the bounding box of the face along with the associated
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        faces = frame[startY:endY, startX:endX]
        pixel_img = anonymize_face_pixelate(faces, blocks=8)

        frame[startY:endY, startX:endX] = cv2.GaussianBlur(frame[startY:endY, startX:endX], (101, 101),
                                                           cv2.BORDER_DEFAULT)
        #warn="press q to exit"
        #cv2.putText(frame, warn, (640,30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        """some exploration"""
        # if you want to show rectangle uncomment it
        #cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

        # if you need to put confidence value uncomment it
        #cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()


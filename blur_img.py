# import the necessary packages
# https://github.com/DEBANJANAB/Face-Blur-and-Anonymization-using-OpenCV/blob/master/blur_techniques/face_blurring.py

import numpy as np
import cv2

# defining prototext and caffemodel paths
caffeModel = "models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "models/deploy.prototxt.txt"

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

# Load Model
net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)
frame = cv2.imread('data/image/id4.jpg')

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
    if confidence < 0.25:
        continue

    # Determine the (x, y)-coordinates of the bounding box for the
    # object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    y = startY - 10 if startY - 10 > 10 else startY + 10

    faces = frame[startY:endY, startX:endX]
    pixel_img = anonymize_face_pixelate(faces, blocks=8)
    frame[startY:endY, startX:endX] = cv2.GaussianBlur(frame[startY:endY, startX:endX], (101, 101),
                                                       cv2.BORDER_DEFAULT)

cv2.imshow('image',frame)
status = cv2.imwrite('/output/python_grey.jpg', frame)
print("Image written to file-system : ", status)
cv2.waitKey(0)
cv2.destroyAllWindows()





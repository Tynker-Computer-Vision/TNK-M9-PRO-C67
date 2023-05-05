import numpy as np
import cv2

# Set confidence and NMS thresholds (Non Maximum Suppression)
confidenceThreshold = 0.3
NMSThreshold = 0.1

# Path to model configuration and weights files
modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

# Path to labels file
labelsPath = 'coco.names'

# Load labels from file
labels = open(labelsPath).read().strip().split('\n')

# Load YOLO object detection network
yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Read the input video file
video = cv2.VideoCapture("bb2.mp4")

while True:
    # Read the first frame of the video
    readVideo = video.read()
    check = readVideo[0]
    image = readVideo[1]

    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    # Get image dimensions
    dimensions = image.shape[:2]
    H = dimensions[0]
    W = dimensions[1]

    # Create blob from image and set input for YOLO network
    # Syntax: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size)
    # 1/255 is takes to normalise the pixel value from 0-255 to 0-1 as the yolo (other models also) require the pixel to be in range 0 to 1.
    # 416,416 is size of images taken by yolo model
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
    yoloNetwork.setInput(blob)

    # Get names of unconnected output layers
    layerName = yoloNetwork.getUnconnectedOutLayersNames()
    # Forward pass through network
    layerOutputs = yoloNetwork.forward(layerName)

    # Initialize lists to store bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIds = []

    # Process each output from YOLO network
    for output in layerOutputs:
        for detection in output:
            # Get class scores and ID of class with highest score
            scores = detection[5:]
            # np.argmax gives indice of maximum score value
            classId = np.argmax(scores)
            confidence = scores[classId]

            # If confidence threshold is met, save bounding box coordinates and class Id
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)

    # Apply Non Maxima Suppression to remove overlapping bounding boxes
    detectionNMS = cv2.dnn.NMSBoxes(
        boxes, confidences, confidenceThreshold, NMSThreshold)

    # If at least one detection remains after NMS
    if (len(detectionNMS) > 0):
        # Process each remaining detection
        for i in detectionNMS.flatten():

            if labels[classIds[i]] == "sports ball":
                # Get bounding box coordinates and class color
                x = boxes[i][0]
                y = boxes[i][1]
                w = boxes[i][2]
                h = boxes[i][3]
                # Red color
                color = (255, 0, 0)
                # Draw bounding box and label on image
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Display image with bounding boxes and labels
    cv2.imshow('Image', image)
    cv2.waitKey(1)

    # Quit the display window when the spacebar key is pressed
    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break

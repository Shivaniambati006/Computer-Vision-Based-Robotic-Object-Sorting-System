import cv2
import numpy as np

# Load the SSD MobileNet model with frozen weights and config
config_file = 'c:\\Users\\ambat\\Downloads\\Object_Detection_Files\\Object_Detection_Files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'c:\\Users\\ambat\\Downloads\\Object_Detection_Files\\Object_Detection_Files\\frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
classlabels = []
filename = 'c:\\Users\\ambat\\Downloads\\Object_Detection_Files\\Object_Detection_Files\\coco.names'
with open(filename, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

# Set model input parameters
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Define a function to classify color based on HSV values
def get_color_name(h, s, v):
    if s < 20:
        if v < 20:
            return "Black"
        elif v > 90:
            return "White"
        else:
            return "Gray"
    if h < 15:
        return "Red"
    elif h < 35:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 125:
        return "Cyan"
    elif h < 170:
        return "Blue"
    elif h < 255:
        return "Magenta"
    else:
        return "Red"

# Define a function to detect shapes and their dimensions
def detect_shape_and_dimensions(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Ignore small contours
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            return "Triangle", w, h
        elif len(approx) == 4:
            aspect_ratio = w / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "Square", w, h
            else:
                return "Rectangle", w, h
        elif len(approx) > 4:
            if 0.85 <= circularity <= 1.15:
                radius = int(np.sqrt(area / np.pi))
                return "Ecllipse", radius * 2, radius * 2
            else:
                return "Circle", w, h

    return "Unknown", 0, 0

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Perform object detection
    confThreshold = 0.5
    class_index, confidence, bbox = model.detect(frame, confThreshold=confThreshold)

    # Process each detected object
    if len(class_index) > 0:
        for class_ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
            (startX, startY, width, height) = boxes
            endX, endY = startX + width, startY + height

            # Extract the region of interest (ROI)
            roi = frame[startY:endY, startX:endX]

            # Get the color at the center of the bounding box
            color_hsv = hsv[(startY + endY) // 2, (startX + endX) // 2]
            color_name = get_color_name(color_hsv[0], color_hsv[1], color_hsv[2])

            # Detect shape and dimensions
            shape, length, width = detect_shape_and_dimensions(roi)

            # Draw bounding box and label
            label = (f'{classlabels[class_ind - 1]}: {color_name}, {shape}, ' \
                     f'Length: {length}, Width: {width}, Conf: {conf * 100:.2f}%')
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Feed with Shapes', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

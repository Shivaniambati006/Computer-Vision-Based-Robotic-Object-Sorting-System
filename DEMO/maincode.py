import cv2
import numpy as np

# Load the SSD MobileNet model with frozen weights and configuration
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

# Define a function for color classification based on HSV values
def get_color_name(h, s, v):
    if s < 50:
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"
    elif h < 10 or h >= 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 125:
        return "Cyan"
    elif 125 <= h < 170:
        return "Blue"
    else:
        return "Unknown"

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

            # Extract ROI and calculate attributes
            roi = frame[startY:endY, startX:endX]
            centerX = (startX + endX) // 2
            centerY = (startY + endY) // 2
            aspect_ratio = width / height

            # Get the color at the center of the bounding box
            color_hsv = hsv[centerY, centerX]
            color_name = get_color_name(color_hsv[0], color_hsv[1], color_hsv[2])

            # Extract shape (optional)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            shape = "Unknown"
            if contours:
                contour = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                shape = "Circle" if len(approx) > 8 else "Polygon"

            # Print object details to the terminal
            print(f"Detected: {classlabels[class_ind - 1]}\n"
                  f"Confidence: {conf * 100:.2f}%\n"
                  f"Position: ({centerX}, {centerY})\n"
                  f"Width: {width}, Height: {height}\n"
                  f"Aspect Ratio: {aspect_ratio:.2f}\n"
                  f"Color: {color_name}\n"
                  f"Shape: {shape}\n")

            # Draw bounding box and label on the frame
            label = (f'{classlabels[class_ind - 1]}: {color_name}, Conf: {conf * 100:.2f}%, '
                     f'W:{width}, H:{height}, AR:{aspect_ratio:.2f}, Pos:({centerX},{centerY})')
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

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

# Refined function for more accurate color classification based on HSV values
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
            (startX, startY, endX, endY) = boxes
            width = endX - startX
            height = endY - startY

            # Get the color at the center of the bounding box
            color_hsv = hsv[(startY + endY) // 2, (startX + endX) // 2]
            color_name = get_color_name(color_hsv[0], color_hsv[1], color_hsv[2])

            # Draw bounding box and label
            label = f'{classlabels[class_ind - 1]}: {color_name}, {conf * 100:.2f}%, W:{width}, H:{height}'
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

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

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Dictionary to hold last known positions of detected objects
tracked_objects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Perform object detection
    confThreshold = 0.5
    class_index, confidence, bbox = model.detect(frame, confThreshold=confThreshold)

    # Update tracked objects
    current_objects = {}

    if len(class_index) > 0:
        for class_ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
            (startX, startY, endX, endY) = boxes
            
            # Get the color at the center of the bounding box
            color_hsv = hsv[(startY + endY) // 2, (startX + endX) // 2]
            color_name = get_color_name(color_hsv[0], color_hsv[1], color_hsv[2])

            # Generate a unique ID for each detected object
            object_id = f"{class_ind}-{conf:.2f}"
            current_objects[object_id] = (startX, startY, endX, endY, color_name, conf)

            # Draw bounding box and label
            label = f'{classlabels[class_ind - 1]}: {color_name}, {conf * 100:.2f}%'
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Smooth the positions of tracked objects
    for object_id, (startX, startY, endX, endY, color_name, conf) in list(tracked_objects.items()):
        if object_id in current_objects:
            # Get the current position
            new_startX, new_startY, new_endX, new_endY, _, _ = current_objects[object_id]
            # Smooth the positions
            startX = int((startX + new_startX) / 2)
            startY = int((startY + new_startY) / 2)  # Fixed variable naming
            endX = int((endX + new_endX) / 2)
            endY = int((endY + new_endY) / 2)
            tracked_objects[object_id] = (startX, startY, endX, endY, color_name, conf)
        else:
            # Remove the object if it's no longer detected
            del tracked_objects[object_id]

    # Update the tracked objects with the current objects
    tracked_objects.update(current_objects)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

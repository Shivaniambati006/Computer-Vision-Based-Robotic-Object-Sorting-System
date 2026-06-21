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

    # Store detected objects for sorting
    detected_objects = []

    if len(class_index) > 0:
        for class_ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
            (startX, startY, width, height) = boxes
            endX, endY = startX + width, startY + height

            # Calculate attributes
            centerX = (startX + endX) // 2
            centerY = (startY + endY) // 2
            area = width * height
            aspect_ratio = width / height

            # Get the color at the center of the bounding box
            color_hsv = hsv[centerY, centerX]
            color_name = get_color_name(color_hsv[0], color_hsv[1], color_hsv[2])

            # Append object data to list
            detected_objects.append({
                "class": classlabels[class_ind - 1],
                "confidence": conf,
                "position": (centerX, centerY),
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "color": color_name,
                "bbox": boxes
            })

    # Sort objects by confidence (descending)
    sorted_objects = sorted(detected_objects, key=lambda x: x["confidence"], reverse=True)

    # Display sorted objects
    for obj in sorted_objects:
        startX, startY, width, height = obj["bbox"]
        endX, endY = startX + width, startY + height
        label = (f'{obj["class"]}: {obj["color"]}, Conf: {obj["confidence"] * 100:.2f}%, '
                 f'W:{obj["width"]}, H:{obj["height"]}, Pos:{obj["position"]}')
        
        # Draw bounding box and label
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print sorted object details to the terminal
        print(f"Class: {obj['class']}, Confidence: {obj['confidence'] * 100:.2f}%, "
              f"Position: {obj['position']}, Width: {obj['width']}, Height: {obj['height']}")

    # Display the resulting frame
    cv2.imshow('Webcam Feed with Sorted Objects', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

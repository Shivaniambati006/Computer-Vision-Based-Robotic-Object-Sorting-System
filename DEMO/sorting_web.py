import cv2
import numpy as np

# Load the SSD MobileNet model
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

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    confThreshold = 0.5
    class_index, confidence, bbox = model.detect(frame, confThreshold=confThreshold)

    # Check if any objects are detected
    if len(class_index) > 0:
        # Get the index of the most confident detected object
        max_conf_index = np.argmax(confidence)
        index = class_index[max_conf_index]
        box = bbox[max_conf_index]
        conf = confidence[max_conf_index]
        
        # Get bounding box coordinates
        (startX, startY, endX, endY) = box

        # Draw bounding box and label
        label = f'{classlabels[index - 1]}: {conf * 100:.2f}%'
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

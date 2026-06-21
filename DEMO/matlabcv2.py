import cv2
import numpy as np
import serial
import time

# Initialize Arduino connection
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to be established

# Load the model and configuration files
config_file = 'c:\\Users\\ambat\\Downloads\\Object_Detection_Files\\Object_Detection_Files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'c:\\Users\\ambat\\Downloads\\Object_Detection_Files\\Object_Detection_Files\\frozen_inference_graph.pb'
labels_path = 'c:\\Users\\ambat\\Downloads\\Object_Detection_Files\\Object_Detection_Files\\coco.names'

# Load class labels
with open(labels_path, 'r') as f:
    class_names = f.read().strip().split('\n')

# Load the pre-trained neural network
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# Initialize video capture (0 = default camera)
cap = cv2.VideoCapture(0)

# Define function to send commands to Arduino
def send_to_arduino(angle1, angle2, angle3, gripper):
    command = f'{angle1},{angle2},{angle3},{gripper}\n'
    arduino.write(command.encode())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for the deep learning model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, crop=False)
    net.setInput(blob)
    detections = net.forward()

    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (left, top, right, bottom) = box.astype("int")
            
            # Draw bounding box and label on the frame
            label = f'{class_names[class_id]}: {confidence:.2f}'
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Calculate angles for the robot arm and send commands to Arduino
            # Replace the following lines with your kinematics calculations
            angle1 = 90  # Example value
            angle2 = 90  # Example value
            angle3 = 90  # Example value
            gripper = 90  # Example value (0 for open, 90 for closed)

            send_to_arduino(angle1, angle2, angle3, gripper)

    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()

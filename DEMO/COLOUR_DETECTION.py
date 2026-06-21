import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

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
        return "black"
    elif h < 125:
        return "Cyan"
    elif h < 170:
        return "Blue"
    elif h < 255:
        return "Magenta"
    else:
        return "Red"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for different colors (you can add more ranges if needed)
    color_ranges = {
        "Red": ([0, 50, 50], [10, 255, 255]),
        "Green": ([36, 50, 50], [85, 255, 255]),
        "Blue": ([110, 50, 50], [130, 255, 255]),
        # Add other colors here
    }

    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # You can adjust this threshold
                x, y, w, h = cv2.boundingRect(cnt)
                color_hsv = hsv[y + h // 2, x + w // 2]
                color_detected = get_color_name(color_hsv[0], color_hsv[1], color_hsv[2])

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_detected, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

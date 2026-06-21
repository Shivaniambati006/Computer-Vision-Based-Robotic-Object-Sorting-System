import cv2
import numpy as np

# Function to find the dominant color in an image
def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = palette[np.argmax(np.bincount(labels.flatten()))]
    return dominant_color

# Function to identify the color name based on RGB values
def identify_color(r, g, b):
    if r > g and r > b:
        return 'Red'
    elif g > r and g > b:
        return 'Green'
    elif b > r and b > g:
        return 'Blue'
    else:
        return 'Unknown'

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Starting camera feed. Press 'q' to quit.")

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Resize the frame to a smaller size for faster processing
    resized_frame = cv2.resize(frame, (300, 300))

    # Find the dominant color in the frame
    dominant_color = get_dominant_color(resized_frame)
    r, g, b = dominant_color
    color_name = identify_color(r, g, b)

    # Display the dominant color name on the frame
    cv2.putText(frame, f"Detected Color: {color_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Create a display image with the detected color
    display_color = np.zeros((100, 300, 3), dtype=np.uint8)
    display_color[:] = (int(b), int(g), int(r))  # OpenCV uses BGR format
    cv2.putText(display_color, color_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the camera feed and detected color
    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Detected Color', display_color)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

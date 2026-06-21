import cv2
import numpy as np
import os

# Function to find the dominant color in an image
def get_dominant_color(image):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = np.float32(image.reshape(-1, 3))
    
    # Number of clusters (K)
    n_colors = 1
    
    # Criteria for K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    
    # Apply K-means clustering
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Get the dominant color
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

# Folder containing images
image_folder = 'path_to_your_image_folder'

# Iterate through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Get the dominant color
        dominant_color = get_dominant_color(image)
        r, g, b = dominant_color
        
        # Identify the color name
        color_name = identify_color(r, g, b)
        
        # Create a display image with the detected color
        display_image = np.zeros((100, 300, 3), dtype=np.uint8)
        display_image[:] = (int(b), int(g), int(r))  # OpenCV uses BGR format

        # Add text to the display image
        cv2.putText(display_image, f"Detected Color: {color_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the original image and the display image
        cv2.imshow('Original Image', image)
        cv2.imshow('Detected Color', display_image)

        # Wait for a key press and then close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("Processing complete.")
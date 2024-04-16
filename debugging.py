import cv2
import numpy as np

# Load the CT scan lung image
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# Preprocess the image (optional)
# Example: Gaussian blur to reduce noise
image_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Thresholding to segment potential tumor regions
# Adjust the threshold values according to your dataset
_, binary_image = cv2.threshold(image_blur, 100, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour
for contour in contours:
    # Calculate contour area
    area = cv2.contourArea(contour)
    
    # Filter out small areas (noise)
    if area > 100:  # Adjust this threshold according to your dataset
        # Draw contour on the original image
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Display the segmented image with contours
cv2.imshow('Segmented Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
